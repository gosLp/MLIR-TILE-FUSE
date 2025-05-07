#include "mlir/Dialect/GPU/Transforms/TileSelection.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include <cmath>
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"   // helper
#include "mlir/Interfaces/TilingInterface.h"                  // TilingInterface
#include "mlir/IR/PatternMatch.h"                             // PatternRewriter

using namespace mlir;
// using namespace mlir::gpu;

namespace {


static bool hasReuseDimension(AffineExpr expr, unsigned dim) {
  // Helper to check if a dimension contributes to an affine expression
  // if (auto dimExpr = expr.dyn_cast<AffineDimExpr>())
  if (auto dimExpr = llvm::dyn_cast<AffineDimExpr>(expr))
    return dimExpr.getPosition() == dim;
  else if (auto binExpr = llvm::dyn_cast<AffineBinaryOpExpr>(expr))
    return hasReuseDimension(binExpr.getLHS(), dim) || 
           hasReuseDimension(binExpr.getRHS(), dim);
  return false;
}

/// Calculate dimensional reuse (γ) for each dimension in a linalg operation
static SmallVector<double> calculateDimensionalReuse(Operation *op) {
  SmallVector<double> reuse;
  
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    unsigned numDims = linalgOp.getNumLoops();
    reuse.resize(numDims, 0.0);
    
    // Analyze each operand's access patterns
    for (OpOperand &opOperand : linalgOp->getOpOperands()) {
      // Skip operands that don't have affine access patterns
      if (!linalgOp.hasPureBufferSemantics() && !linalgOp.hasPureTensorSemantics())
        continue;

      AffineMap accessMap = linalgOp.getMatchingIndexingMap(&opOperand);
        
      // Count temporal reuse along each dimension
      for (unsigned dim = 0; dim < numDims; ++dim) {
        // Check if this dimension contributes to indexing this operand
        bool hasReuse = false;
        for (AffineExpr expr : accessMap.getResults()) {
          // Check if dimension appears in the expression
          if (hasReuseDimension(expr, dim)) {
            hasReuse = true;
            break;
          }
        }
        if (hasReuse)
          reuse[dim] += 1.0;
      }
    }
    
    // Normalize reuse factors if needed
    double maxReuse = 0.0;
    for (double r : reuse)
      maxReuse = std::max(maxReuse, r);
    
    if (maxReuse > 0.0) {
      for (double &r : reuse)
        r = std::max(0.5, r / maxReuse * 2.0); // Scale between 0.5 and 2.0
    }
  }
  
  return reuse;
}




/// Solve the reuse polynomial to determine tile sizes
static SmallVector<int64_t> computeTileSizes(ArrayRef<double> reuse, int64_t cacheSize) {
  unsigned numDims = reuse.size();
  SmallVector<int64_t> tileSizes(numDims);
  
  // Construct the reuse expression (polynomial)
  double reuseCoefficient = 0.0;
  
  // Sum of pairwise products (i<j) of reuse factors: γi * γj
  for (unsigned i = 0; i < numDims; ++i) {
    for (unsigned j = i + 1; j < numDims; ++j) {
      reuseCoefficient += reuse[i] * reuse[j];
    }
  }
  
  // If there's only one dimension or no reuse detected
  if (reuseCoefficient < 0.1) {
    // Fallback to simple distribution
    double tileVolume = std::sqrt(cacheSize);
    for (unsigned i = 0; i < numDims; ++i) {
      tileSizes[i] = std::max<int64_t>(16, std::min<int64_t>(256, 
                      static_cast<int64_t>(tileVolume * reuse[i] / numDims)));
    }
    return tileSizes;
  }
  
  // Solve for τ using the polynomial: (γi * γj + γj * γk + γk * γi) * τ² = CacheSize
  double tau = std::sqrt(cacheSize / reuseCoefficient);
  
  // Calculate tile sizes: ti = γi * τ
  for (unsigned i = 0; i < numDims; ++i) {
    double rawSize = reuse[i] * tau;
    
    // Apply practical constraints
    int64_t tileSize = static_cast<int64_t>(rawSize);
    
    // Ensure minimum and maximum sizes
    tileSize = std::max<int64_t>(16, std::min<int64_t>(512, tileSize));
    
    // Power of 2 adjustment (optional)
    // tileSize = 1 << (32 - llvm::countLeadingZeros(static_cast<uint32_t>(tileSize - 1)));
    
    tileSizes[i] = tileSize;
  }
  
  return tileSizes;
}

// Function to detect if two operations have a producer-consumer relationship
bool hasProducerConsumerRelation(Operation *producer, Operation *consumer) {
  // Check if any result of producer is used by consumer
  for (Value result : producer->getResults()) {
    for (OpOperand &use : result.getUses()) {
      if (use.getOwner() == consumer)
        return true;
    }
  }
  return false;
}




void addPrefetching(Operation *tiledOp) {
  // Locate the innermost tiled loops
  
  // For each memory access in the compute region:
  // 1. Create shared memory allocation
  // 2. Insert async copy operations before the compute
  // 3. Add synchronization to ensure data is ready
  
  // Example:
  /*
  OpBuilder builder(tiledOp);
  Location loc = tiledOp->getLoc();
  
  // Create shared memory allocation
  Value sharedMem = builder.create<gpu::AllocSharedMemOp>(loc, memRefType);
  
  // Create async copy from global to shared memory
  builder.create<nvgpu::DeviceAsyncCopyOp>(loc, 
                                          globalMem, 
                                          sharedMem, 
                                          ... indices ...);
  
  // Create async wait group
  Value group = builder.create<nvgpu::DeviceAsyncCreateGroupOp>(loc);
  
  // Insert wait operation
  builder.create<nvgpu::DeviceAsyncWaitOp>(loc, group);
  
  // Replace accesses to global memory with accesses to shared memory
  */
}
} // end of namespace

SmallVector<int64_t> mlir::gpu::calculateOptimalTileSizes(Operation *op) {
  auto linalgOp = dyn_cast_if_present<linalg::LinalgOp>(op);
  if (!linalgOp)            // <-- extra safety
    return {64, 64, 64};    // or whatever your default is
  // Get reuse factors for each dimension
  // SmallVector<double> reuse = calculateDimensionalReuse(op);
  
  // 1. Use 64 KB by default (or let the user pass it via module attr)
  int64_t sharedMemBytes = 64 * 1024;   // 65 536 B
  if (auto attr = op->getParentOfType<ModuleOp>()
                 ->getAttrOfType<IntegerAttr>("gpu.smem_kb"))
    sharedMemBytes = attr.getInt() * 1024;                  // allow override
  // return computeTileSizes(reuse, gpuL1CacheSize);
  constexpr int64_t bytesPerElem = 4;                 // f32
  int64_t cacheElems = sharedMemBytes / bytesPerElem; // ***convert***

  // ────────────────────────────────────────────────────────────────
  // 2.  Solve for τ and get the raw tile sizes
  // ────────────────────────────────────────────────────────────────
  SmallVector<double> reuse = calculateDimensionalReuse(op);
  SmallVector<int64_t> sizes = computeTileSizes(reuse, cacheElems);
  //  (computeTileSizes now gets a *capacity‑in‑elements*, so no change there)

  /// ────────────────────────────────────────────────────────────────
  // 3.  Warp‑friendly round‑down (multiple of 8) and clamp
  //     to the static problem shape so we never exceed 64×64 etc.
  // ────────────────────────────────────────────────────────────────
  auto staticShape = linalgOp.getStaticLoopRanges();  // {64,64,128} for GEMM
  for (unsigned i = 0; i < sizes.size(); ++i) {
    sizes[i] = (sizes[i] / 8) * 8;                    // align
    sizes[i] = std::max<int64_t>(8, sizes[i]);
    if (staticShape[i] > 0)                           // ‑1 == dynamic
      sizes[i] = std::min<int64_t>(sizes[i], staticShape[i]);
  }

  // ────────────────────────────────────────────────────────────────
  // 4.  If footprint > shared‑mem, shrink all dims together
  // ────────────────────────────────────────────────────────────────
  auto footprintBytes = [&](ArrayRef<int64_t> ts) {
    int64_t ij = ts[0] * ts[1];
    int64_t jk = (ts.size() > 2 ? ts[1] * ts[2] : 0);
    int64_t ki = (ts.size() > 2 ? ts[2] * ts[0] : 0);
    return (ij + jk + ki) * bytesPerElem;    // 4 B/float
  };

  while (footprintBytes(sizes) > sharedMemBytes) {
    for (auto &s : sizes) s = std::max<int64_t>(8, s - 8);            // shrink one “chunk”
  }
  return sizes;
}

namespace mlir {
  namespace gpu {
  
  // SmallVector<int64_t>
  // fitTileSizesToOp(Operation *op, ArrayRef<int64_t> sizes) {
  //   auto linalgOp = cast<linalg::LinalgOp>(op);
  //   unsigned loops = linalgOp.getNumLoops();
  //   SmallVector<int64_t> trimmed;
  //   trimmed.assign(sizes.begin(), sizes.begin() + loops);
  //   return trimmed;
  // }
SmallVector<int64_t>
fitTileSizesToOp(Operation *op, ArrayRef<int64_t> sizes) {
  unsigned nLoops =
      cast<TilingInterface>(op).getLoopIteratorTypes().size();
  SmallVector<int64_t> out;

  // copy as many as we have loops
  out.append(sizes.begin(),
             sizes.begin() + std::min<size_t>(nLoops, sizes.size()));

  // pad with zeros for “no‑tile” on leftover loops
  out.resize(nLoops, /*zero‑tile =*/0);
  return out;
}

  // SmallVector<int64_t> fitTileSizesToOp(Operation *op, ArrayRef<int64_t> tileSizes) {
  //   if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
  //     unsigned numLoops = linalgOp.getNumLoops();
  //     SmallVector<int64_t> result;
  //     result.reserve(numLoops);
      
  //     // Copy tile sizes up to the number of loops
  //     for (unsigned i = 0; i < std::min(numLoops, static_cast<unsigned>(tileSizes.size())); ++i) {
  //       result.push_back(tileSizes[i]);
  //     }
      
  //     // Pad with default sizes if needed
  //     while (result.size() < numLoops) {
  //       result.push_back(64); // Default tile size
  //     }
      
  //     return result;
  //   }
  //   return SmallVector<int64_t>(tileSizes);
  // }

//   bool fuseProducerConsumer(PatternRewriter &rewriter,
//     Operation *producer, 
//     Operation *consumer,
//     ArrayRef<int64_t> tileSizes) {
// // First tile the producer
// rewriter.setInsertionPoint(producer);
// auto producerOp = dyn_cast<linalg::LinalgOp>(producer);
// if (!producerOp) return false;

// linalg::LinalgTilingOptions producerOptions;
// producerOptions = producerOptions.setTileSizes(tileSizes);

// auto tiledProducer = linalg::tileLinalgOp(rewriter, producerOp, producerOptions);
// if (failed(tiledProducer)) return false;

// // Get the new producer operation after tiling
// Operation *newProducer = tiledProducer->op;

// // Now tile the consumer with compatible tile sizes
// rewriter.setInsertionPoint(consumer);
// auto consumerOp = dyn_cast<linalg::LinalgOp>(consumer);
// if (!consumerOp) return false;

// // Adapt tile sizes for the consumer
// SmallVector<int64_t> consumerTileSizes = fitTileSizesToOp(consumer, tileSizes);

// linalg::LinalgTilingOptions consumerOptions;
// consumerOptions = consumerOptions.setTileSizes(consumerTileSizes);

// auto tiledConsumer = linalg::tileLinalgOp(rewriter, consumerOp, consumerOptions);
// if (failed(tiledConsumer)) return false;

// // Instead of directly trying to fuse, which may cause issues with the current setup,
// // we'll simply ensure both operations are tiled with compatible tile sizes
// // and let the compiler's other passes handle the fusion

// return true;
// }
  /// Implementation of tileAndFuseConsumerWithProducers
//   bool fuseProducerConsumer(PatternRewriter &rewriter,
//     linalg::LinalgOp producer,
//     linalg::LinalgOp consumer,
//     ArrayRef<int64_t> tileSizes) {

// // We only need to call this on the **consumer**; the helper finds and fuses
// // its producers automatically.
// rewriter.setInsertionPoint(consumer);

// linalg::LinalgTilingOptions tiling;
// tiling.setTileSizes(tileSizes);

// linalg::LinalgFusionOptions fusion;
// fusion.controlFoldingFn =
// [](Operation *prod, Operation *cons) { return true; }; // fuse everything

// FailureOr<linalg::TileLoopNest> fused =
// linalg::tileAndFuseLinalgOps(rewriter, consumer, tiling, fusion);

// return succeeded(fused);
// }

// Convert int64 -> constant OpFoldResult.
static SmallVector<OpFoldResult>
asConstantMixedSizes(OpBuilder &b, ArrayRef<int64_t> sizes) {
  SmallVector<OpFoldResult> mixed;
  mixed.reserve(sizes.size());
  for (int64_t sz : sizes)
    mixed.push_back(b.getIndexAttr(sz));
  return mixed;
}
/// Tiles the `consumer` op (which must implement `TilingInterface`) and
/// greedily fuses all its in‑block producers.  Returns `true` on success.
/// Tiles `consumer` (implements TilingInterface) and greedily fuses all
/// in‑block producers using the up‑to‑date SCF helpers.
bool tileAndFuseWithSCF(PatternRewriter &rewriter,
  TilingInterface consumer,
  ArrayRef<int64_t> tileSizes) {
  // ---- 2.1 build OpFoldResult tile sizes ----
  // SmallVector<OpFoldResult> mixed;
  // mixed.reserve(tileSizes.size());
  // for (int64_t sz : tileSizes)
  // mixed.push_back(rewriter.getIndexAttr(sz));   // constant IndexAttr

  unsigned nLoops = consumer.getLoopIteratorTypes().size();
  SmallVector<int64_t> ts(tileSizes.begin(), tileSizes.end());
  ts.resize(nLoops, 0);          // trim or pad with 0 (“don’t tile”)

  // ---- 2.2 SCF tiling options ----
  scf::SCFTilingOptions tileOpts;
  tileOpts.setTileSizes(asConstantMixedSizes(rewriter, tileSizes));

  scf::SCFTileAndFuseOptions tfOpts;
  tfOpts.setTilingOptions(tileOpts);
  // (optional) you can also attach a fusion‑control lambda:
  // tfOpts.setFusionControlFn(/*ControlFnTy*/);

  // --- 3. Call the plural helper ---
  auto fused =
  scf::tileConsumerAndFuseProducersUsingSCF(rewriter, consumer, tfOpts);

  // fusing didn't succeeed
  if (failed(fused)){
    return false;
  }
// Success! Now replace/erase the original operations
  
  // 1. Replace the consumer with appropriate values from the replacements map
  // First get all results of the consumer
  // Operation *consumerOp = consumer.getOperation();
  // SmallVector<Value> replacementValues;
 // --- 1. Replace every mapped value ---
  for (auto &kv : fused->replacements)           // kv : <old, new>
    rewriter.replaceAllUsesWith(kv.first, kv.second);
  
  // for (Value result : consumerOp->getResults()) {
  //   // Look up the replacement in the map
  //   if (fused->replacements.count(result)) {
  //     replacementValues.push_back(fused->replacements[result]);
  //   } else {
  //     // No replacement found, use the original result
  //     replacementValues.push_back(result);
  //   }
  // }

  // --- 2. Erase the original consumer and all fused producers ---
  Operation *origConsumer = consumer.getOperation();
  rewriter.eraseOp(origConsumer);

  for (Operation *op : fused->fusedProducers)
    if (op->use_empty())                        // safety belt
      rewriter.eraseOp(op);
  // return succeeded(fused);
  return true;
}


  } // namespace gpu
  } // namespace mlir
  




#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"         // for PatternRewriter

// void mlir::gpu::applyTiling(PatternRewriter &rewriter,
//                             Operation *op,
//                             ArrayRef<int64_t> tileSizes) {
//   // Try to cast to a LinalgOp
//   if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
//     assert(linalgOp && "applyTiling expects a LinalgOp");
//     assert(tileSizes.size() == linalgOp.getNumLoops() &&
//        "tile size vector length must equal loop nest length");

//     // 1) Prepare tiling options.
//     Location loc = op->getLoc();
//     linalg::LinalgTilingOptions tilingOptions;
//     tilingOptions = tilingOptions.setTileSizes(tileSizes);
    
//     // 2) Ensure we insert *at* the original op.
//     rewriter.setInsertionPoint(op);
    
//     // 3) Invoke the tiler using the RewriterBase API.
//     FailureOr<linalg::TiledLinalgOp> result =
//       linalg::tileLinalgOp(rewriter, linalgOp, tilingOptions);
    
//     if (succeeded(result)) {
//       // 4) Erase the original op (rewriter knows how to handle this)
//       rewriter.eraseOp(linalgOp);
//     }
//   }

//   // (2) The inline “convenience” wrapper also needs a body:
// }

void mlir::gpu::applyTiling(PatternRewriter &rewriter,
  Operation *op,
  ArrayRef<int64_t> tileSizes) {
auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
if (!linalgOp)
return;

assert(tileSizes.size() == linalgOp.getNumLoops() &&
"tile size vector length must equal loop nest length");

linalg::LinalgTilingOptions opts;
opts.setTileSizes(tileSizes);

rewriter.setInsertionPoint(linalgOp);
FailureOr<linalg::TiledLinalgOp> tiled =
linalg::tileLinalgOp(rewriter, linalgOp, opts);
if (failed(tiled))
return;

// ► Wire new results to old users *before* erasing the op.
if (!linalgOp->getResults().empty())
rewriter.replaceOp(linalgOp, tiled->tensorResults);
else
rewriter.eraseOp(linalgOp);
}


// 2) Convenience wrapper
// void mlir::gpu::applyTiling(Operation *op,
//   llvm::ArrayRef<int64_t> tileSizes) {
// PatternRewriter rewriter(op->getContext());
// rewriter.setInsertionPoint(op);
// applyTiling(rewriter, op, tileSizes);
