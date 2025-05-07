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

using namespace mlir;
// using namespace mlir::gpu;

namespace {

// /// Calculate dimensional reuse (γ) for each dimension in a linalg operation
// static SmallVector<double> calculateDimensionalReuse(Operation *op) {
//   // Implement your dimensional reuse calculation algorithm here
//   // based on the paper you shared
  
//   // Example implementation:
//   SmallVector<double> reuse;
//   if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
//     unsigned numDims = linalgOp.getNumLoops();
//     reuse.resize(numDims, 0.0);

//     // ANalyse each opperands access pattterns 
//     for (OpOperand &opOperand : linalgOp->getOperands()) {
//       // skip opperands that don't have affine access patterns
//       if (!linalgOp.hasPureBufferSemantics() && !linalgOp.hasPureTensorSemantics())
//         continue;
      

//       AffineMap accessMap;
//       if (linalgOp.hasPureBufferSemantics()) {
      
//       }
//     }
    
//     // Calculate reuse factors for each dimension
//     // (simplified example)
//     if (numDims >= 3) {
//       reuse[0] = 0.5;  // Less reuse in outer dimension
//       reuse[1] = 1.0;  // Medium reuse in middle dimension
//       reuse[2] = 2.0;  // More reuse in inner dimension
//     }
//   }
  
//   return reuse;
// }

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
        
      // AffineMap accessMap;
      // if (linalgOp.hasPureBufferSemantics())
      //   // accessMap = linalgOp.getIndexingMapIndex(&opOperand);
      //   accessMap = linalgOp.getIndexingMapIndex(&opOperand);
      // else
      //   // accessMap = linalgOp.getTiedIndexingMap(&opOperand);
      //   accessMap = linalgOp.getMatchingIndexingMap(&opOperand);

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
// static SmallVector<int64_t> computeTileSizes(ArrayRef<double> reuse, int64_t cacheSize) {
//   // Implement algorithm from the paper for computing tile sizes
//   // based on cache size and reuse factors
  
//   unsigned numDims = reuse.size();
//   SmallVector<int64_t> tileSizes(numDims);
  
//   // Example implementation (simplified):
//   double totalReuse = 0.0;
//   for (double r : reuse)
//     totalReuse += r;
  
//   // Distribute cache space proportionally to reuse
//   for (unsigned i = 0; i < numDims; ++i) {
//     double ratio = reuse[i] / totalReuse;
//     int64_t tileSize = static_cast<int64_t>(std::sqrt(cacheSize * ratio));
//     tileSizes[i] = std::max(int64_t(16), std::min(int64_t(256), tileSize));
//   }
  
//   return tileSizes;
// }
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

// Function to fuse operations with producer-consumer relationship
// static void fuseProducerConsumer(Operation *producer, Operation *consumer, 
//                          ArrayRef<int64_t> tileSizes, 
//                          PatternRewriter &rewriter) {
//   // ── Step 1.  Find the *consumer operand* that points at the producer result.
//   OpOperand *use = nullptr;
//   for (OpOperand &operand : consumer->getOpOperands())
//     if (operand.get().getDefiningOp() == producer) {
//       use = &operand;
//       break;
//     }
//   if (!use)
//     return;                    // nothing to fuse  (shouldn’t happen in your pass)
  
//   // ── Step 2.  Ask Linalg to perform the fusion.  The 2‑arg overload expects
//   //            (OpBuilder&, OpOperand&).  It stitches the producer’s computation
//   //            right before the consumer *inside* the consumer’s loop nest.
//   FailureOr<linalg::FusionInfo> fused =
//       linalg::fuseProducerOfTensor(rewriter, *use);
//   if (failed(fused))
//     return;                    // fusion deemed illegal => silently skip
  
//   // After fusion we get (potentially) new producer / consumer handles:
//   Operation *fusedProducer = fused->fusedProducer ? fused->fusedProducer
//                                                   : producer;
//   Operation *consumerToTile = consumer; // unchanges
  
//     // ── Step 3.  Tile both operations with the same sizes.
//   //     NOTE:  applyTiling erases the original op, so always call it *last*.
//   mlir::gpu::applyTiling(rewriter, fusedProducer, tileSizes);
//   mlir::gpu::applyTiling(rewriter, consumerToTile, tileSizes);
  
//   // // Get lcoations for the operations
//   // Location producerLoc = producer->getLoc();
//   // Location consumerLoc = consumer->getLoc();

//   // // Cast to LinalgOps for proper handling
//   // auto producerOp = dyn_cast<linalg::LinalgOp>(producer);
//   // auto consumerOp = dyn_cast<linalg::LinalgOp>(consumer);

//   // if (!producerOp || !consumerOp) {
//   //   // If either operation is not a LinalgOp, we can't fuse them
//   //   llvm::errs() << "Cannot fuse operations that are not LinalgOps\n";
//   //   return;
//   // }

//   // // Set up fusion options
//   // linalg::LinalgTilingOptions tilingOptions;
//   // tilingOptions = tilingOptions.setTileSizes(tileSizes);

//   // Value fusionValue = nullptr;
//   // for (Value result : producer->getResults()) {
//   //   for (OpOperand &use : result.getUses()) {
//   //     if (use.getOwner() == consumer) {
//   //       fusionValue = result;
//   //       break;
//   //     }
//   //   }
//   //   if (fusionValue)
//   //     break;
//   // }

//   // if (!fusionValue) {
//   //   llvm::errs() << "No direct producer-consumer relationship found\n";
//   //   return;
//   // }
  
//   // // Use linalg's fusion utilities
//   // FailureOr<linalg::FusionInfo> fusionInfo = linalg::fuseProducerOfTensor(
//   //     rewriter, fusionValue, tilingOptions);
  
//   // if (failed(fusionInfo)) {
//   //   llvm::errs() << "Fusion failed\n";
//   //   return;
//   // }

//   // // Successfully fused - now need to cleanup
  
//   // // Depending on fusion approach, we might need to erase the original ops
//   // // or they might have been replaced already by the fusion utilities
  
//   // // Check if the operations are still valid and need to be erased
//   // // if (!producer->isOpBeingReplaced() && !producer->use_empty())
//   // //   rewriter.eraseOp(producer);
  
//   // // if (!consumer->isOpBeingReplaced() && !consumer->use_empty())
//   // //   rewriter.eraseOp(consumer);
  
//   llvm::errs() << "Successfully fused producer and consumer operations\n";
  
  
//   // // Create a fusion group
//   // OpBuilder builder(producer);
//   // Location loc = producer->getLoc();
  
//   // Apply tiling to both operations with the same tile sizes
//   // This ensures they can be fused
  
//   // For real implementation, use the transform dialect's tileAndFuse operation
//   // or implement custom fusion logic
// }

/// Tiles producer & consumer with the same sizes, then fuses the producer slice
/// into the consumer.  Works on current MLIR main.p

// ──────────────────────────────────────────────────────────────────────────────
// Tiles PRODUCER and CONSUMER with the SAME sizes, then fuses the producer
// slice into the consumer.  Works on current MLIR main.
// ──────────────────────────────────────────────────────────────────────────────
// static void fuseProducerConsumer(Operation        *producer,
//                                  Operation        *consumer,
//                                  ArrayRef<int64_t> tileSizes,
//                                  PatternRewriter  &rewriter) {
//   // Build tiling options once.
//   linalg::LinalgTilingOptions opt;
//   opt = opt.setTileSizes(tileSizes);

//   // ── Tile producer ──────────────────────────────────────────────────────────
//   rewriter.setInsertionPoint(producer);
//   auto tiledProd = linalg::tileLinalgOp(
//         rewriter, cast<linalg::LinalgOp>(producer), opt);
//   if (failed(tiledProd)) return;
//   Operation *newProducer = tiledProd->op;   // keep handle

//   // ── Tile consumer ──────────────────────────────────────────────────────────
//   rewriter.setInsertionPoint(consumer);
//   auto tiledCons = linalg::tileLinalgOp(
//         rewriter, cast<linalg::LinalgOp>(consumer), opt);
//   if (failed(tiledCons)) return;
//   Operation *newConsumer = tiledCons->op;

//   // ── Find the operand that links them ───────────────────────────────────────
//   OpOperand *use = nullptr;
//   for (OpOperand &o : newConsumer->getOpOperands())
//     if (o.get().getDefiningOp() == newProducer) { use = &o; break; }
//   if (!use) return; // shapes didn’t match → skip

//   // ── Fuse ───────────────────────────────────────────────────────────────────
//   if (succeeded(linalg::fuseProducerOfTensor(rewriter, *use)))
//     llvm::errs() << "fusion OK\n";
//   else
//     llvm::errs() << "fusion FAILED after tiling\n";
// }


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
  // Get reuse factors for each dimension
  SmallVector<double> reuse = calculateDimensionalReuse(op);
  
  // Compute tile sizes based on GPU L1 cache size (typically around 48KB)
  const int64_t gpuL1CacheSize = 48 * 1024;
  // return computeTileSizes(reuse, gpuL1CacheSize);
  SmallVector<int64_t> sizes = computeTileSizes(reuse, gpuL1CacheSize);
  for (auto &s : sizes) s = std::min<int64_t>(s, 64);   // problem‑size guard
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
  SmallVector<int64_t> fitTileSizesToOp(Operation *op, ArrayRef<int64_t> tileSizes) {
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
      unsigned numLoops = linalgOp.getNumLoops();
      SmallVector<int64_t> result;
      result.reserve(numLoops);
      
      // Copy tile sizes up to the number of loops
      for (unsigned i = 0; i < std::min(numLoops, static_cast<unsigned>(tileSizes.size())); ++i) {
        result.push_back(tileSizes[i]);
      }
      
      // Pad with default sizes if needed
      while (result.size() < numLoops) {
        result.push_back(64); // Default tile size
      }
      
      return result;
    }
    return SmallVector<int64_t>(tileSizes);
  }

  bool fuseProducerConsumer(PatternRewriter &rewriter,
    Operation *producer, 
    Operation *consumer,
    ArrayRef<int64_t> tileSizes) {
// First tile the producer
rewriter.setInsertionPoint(producer);
auto producerOp = dyn_cast<linalg::LinalgOp>(producer);
if (!producerOp) return false;

linalg::LinalgTilingOptions producerOptions;
producerOptions = producerOptions.setTileSizes(tileSizes);

auto tiledProducer = linalg::tileLinalgOp(rewriter, producerOp, producerOptions);
if (failed(tiledProducer)) return false;

// Get the new producer operation after tiling
Operation *newProducer = tiledProducer->op;

// Now tile the consumer with compatible tile sizes
rewriter.setInsertionPoint(consumer);
auto consumerOp = dyn_cast<linalg::LinalgOp>(consumer);
if (!consumerOp) return false;

// Adapt tile sizes for the consumer
SmallVector<int64_t> consumerTileSizes = fitTileSizesToOp(consumer, tileSizes);

linalg::LinalgTilingOptions consumerOptions;
consumerOptions = consumerOptions.setTileSizes(consumerTileSizes);

auto tiledConsumer = linalg::tileLinalgOp(rewriter, consumerOp, consumerOptions);
if (failed(tiledConsumer)) return false;

// Instead of directly trying to fuse, which may cause issues with the current setup,
// we'll simply ensure both operations are tiled with compatible tile sizes
// and let the compiler's other passes handle the fusion

return true;
}
  
  } // namespace gpu
  } // namespace mlir
  




#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"         // for PatternRewriter

void mlir::gpu::applyTiling(PatternRewriter &rewriter,
                            Operation *op,
                            ArrayRef<int64_t> tileSizes) {
  // Try to cast to a LinalgOp
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    assert(linalgOp && "applyTiling expects a LinalgOp");
    assert(tileSizes.size() == linalgOp.getNumLoops() &&
       "tile size vector length must equal loop nest length");

    // 1) Prepare tiling options.
    Location loc = op->getLoc();
    linalg::LinalgTilingOptions tilingOptions;
    tilingOptions = tilingOptions.setTileSizes(tileSizes);
    
    // 2) Ensure we insert *at* the original op.
    rewriter.setInsertionPoint(op);
    
    // 3) Invoke the tiler using the RewriterBase API.
    FailureOr<linalg::TiledLinalgOp> result =
      linalg::tileLinalgOp(rewriter, linalgOp, tilingOptions);
    
    if (succeeded(result)) {
      // 4) Erase the original op (rewriter knows how to handle this)
      rewriter.eraseOp(linalgOp);
    }
  }

  // (2) The inline “convenience” wrapper also needs a body:
}

// 2) Convenience wrapper
// void mlir::gpu::applyTiling(Operation *op,
//   llvm::ArrayRef<int64_t> tileSizes) {
// PatternRewriter rewriter(op->getContext());
// rewriter.setInsertionPoint(op);
// applyTiling(rewriter, op, tileSizes);
