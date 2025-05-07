#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/TileSelection.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/PatternMatch.h"      // for PatternRewriter
#include "llvm/ADT/ArrayRef.h"        // for ArrayRef
#include "mlir/Dialect/GPU/Transforms/TileSelection.h"

using namespace mlir;

namespace {
// Forward declarations of helper functions
static bool hasProducerConsumerRelation(Operation *producer, Operation *consumer);
static void addPrefetching(Operation *tiledOp);
/// Test pass for the GPU tile selection utility
struct TestGpuTileSelectionPass
    : public PassWrapper<TestGpuTileSelectionPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestGpuTileSelectionPass)
  
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, linalg::LinalgDialect, scf::SCFDialect>();
  }
  
  StringRef getArgument() const final { return "test-gpu-tile-selection"; }
  
  StringRef getDescription() const final {
    return "Test pass for GPU operation tile size selection";
  }
  
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    
    // Print confirmation message
    llvm::errs() << "===============\n";
    llvm::errs() << "TILE SELECTION PASS IS RUNNING!\n";
    llvm::errs() << "===============\n";
    
    // Find operations to analyze and transform
    funcOp.walk([&](linalg::LinalgOp op) {
      // Only process operations with the test marker attribute
      if (!op->hasAttr("gpu.test_tile_selection"))
        return;
      
      // Calculate optimal tile sizes
      SmallVector<int64_t> tileSizes = gpu::calculateOptimalTileSizes(op);
      
      // Print the calculated tile sizes for testing
      llvm::errs() << "Calculated tile sizes for operation: " 
                  << op->getName() << "\n";
      for (auto size : tileSizes)
        llvm::errs() << size << " ";
      llvm::errs() << "\n";
      
      // Create a pattern rewriter for tiling
      PatternRewriter rewriter(op->getContext());
      rewriter.setInsertionPoint(op);
      
      // Apply tiling transformation with the rewriter
      // gpu::applyTiling(rewriter, op, tileSizes);
      // In your test pass:
      gpu::applyTiling(rewriter, op, tileSizes);  
    });
  }
};

/// Test pass for the GPU tile selection and fusion utility
struct TestGpuTileAndFusePass
    : public PassWrapper<TestGpuTileAndFusePass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestGpuTileAndFusePass)
  
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect, linalg::LinalgDialect, scf::SCFDialect>();
  }
  
  StringRef getArgument() const final { return "test-gpu-tile-and-fuse"; }
  
  StringRef getDescription() const final {
    return "Test pass for GPU tile selection, fusion and prefetching";
  }
  
  // void runOnOperation() override {
  //   func::FuncOp funcOp = getOperation();
    
  //   // Print confirmation message
  //   llvm::errs() << "===============\n";
  //   llvm::errs() << "TILE AND FUSION PASS IS RUNNING!\n";
  //   llvm::errs() << "===============\n";
    
  //   // Find candidate operations for tiling and fusion
  //   SmallVector<linalg::LinalgOp> candidates;
    
  //   // First pass: collect linalg operations with the marker attribute
  //   funcOp.walk([&](linalg::LinalgOp op) {
  //     if (op->hasAttr("gpu.test_tile_selection"))
  //       candidates.push_back(op);
  //   });
    
  //   llvm::errs() << "Found " << candidates.size() << " marked linalg operations\n";
    
  //   // Second pass: identify fusion opportunities
  //   // Second pass: identify fusion opportunities
  // // We'll track which operations have been processed to avoid duplicate work
  // SmallPtrSet<Operation*, 8> processedOps;
  //   for (unsigned i = 0; i < candidates.size(); i++) {
  //     // Skip if this operation has already been processed
  //     if (processedOps.contains(candidates[i].getOperation()))
  //       continue;
  //     bool fused = false;
      
  //     for (unsigned j = i + 1; j < candidates.size() && !fused; j++) {
  //       if (hasProducerConsumerRelation(candidates[i].getOperation(), 
  //                                       candidates[j].getOperation())) {
  //         // Calculate tile sizes based on the producer operation
  //         SmallVector<int64_t> tileSizes = 
  //             gpu::calculateOptimalTileSizes(candidates[i].getOperation());
          
  //         llvm::errs() << "Found fusion opportunity between operations at indices " 
  //                      << i << " and " << j << "\n";
  //         llvm::errs() << "Operation types: " << candidates[i]->getName() << " and "
  //                      << candidates[j]->getName() << "\n";
          
  //         // Print the calculated tile sizes
  //         llvm::errs() << "Using tile sizes: ";
  //         for (auto size : tileSizes)
  //           llvm::errs() << size << " ";
  //         llvm::errs() << "\n";
          
  //         // Create a pattern rewriter for tiling and fusion
  //         PatternRewriter rewriter(candidates[i]->getContext());

  //       //   // Attempt safer fusion
  //       // if (gpu::fuseProducerConsumer(rewriter, 
  //       //     candidates[i].getOperation(),
  //       //     candidates[j].getOperation(),
  //       //     tileSizes)) {
  //       //   llvm::errs() << "Successfully tiled and prepared for fusion\n";
  //       //   fused = true;
  //       //   // Skip both operations in future iterations
  //       //   i = j;
  //       // } else {
  //       //   llvm::errs() << "Could not tile and fuse operations, falling back to individual tiling\n";
  //       // }
  //       rewriter.setInsertionPoint(candidates[j]);
        
  //       // // We'll apply tiling and fusion to the CONSUMER operation
  //       // // This will automatically find and fuse the producer
  //       // if (succeeded(gpu::tileAndFuseConsumerWithProducers(
  //       //     rewriter, candidates[j], tileSizes))) {
  //       //   llvm::errs() << "Successfully tiled and fused operations\n";
          
  //       //   // Mark both ops as processed
  //       //   processedOps.insert(candidates[i].getOperation());
  //       //   processedOps.insert(candidates[j].getOperation());
          
  //       //   fused = true;
  //       // } else {
  //       //   llvm::errs() << "Failed to tile and fuse operations\n";
  //       // }
          
  //         // // Apply tiling to producer
  //         // rewriter.setInsertionPoint(candidates[i]);
  //         // gpu::applyTiling(rewriter, candidates[i].getOperation(), tileSizes);
          
  //         // // Apply tiling to consumer
  //         // rewriter.setInsertionPoint(candidates[j]);
  //         // auto consumerSizes = mlir::gpu::fitTileSizesToOp(candidates[j], tileSizes);
  //         // // gpu::applyTiling(rewriter, candidates[j].getOperation(), tileSizes);
  //         // gpu::applyTiling(rewriter, candidates[j].getOperation(), consumerSizes);
          
  //         // // Add prefetching
  //         // addPrefetching(candidates[j].getOperation());
          
  //         // // Mark as fused to skip individual tiling of this operation
  //         // fused = true;
          
  //         // // Skip both operations in future iterations
  //         // i = j;
  //         // break;
  //       }
  //     }
      
  //     // If no fusion opportunity, just tile this operation
  //     if (!fused && i < candidates.size()) {
  //       Operation *op = candidates[i].getOperation();
  //       SmallVector<int64_t> tileSizes = gpu::calculateOptimalTileSizes(op);
        
  //       llvm::errs() << "Applying tiling to operation at index " << i << "\n";
  //       llvm::errs() << "Operation type: " << op->getName() << "\n";
        
  //       // Print the calculated tile sizes
  //       llvm::errs() << "Using tile sizes: ";
  //       for (auto size : tileSizes)
  //         llvm::errs() << size << " ";
  //       llvm::errs() << "\n";
        
  //       // Create a pattern rewriter for tiling
  //       PatternRewriter rewriter(op->getContext());
  //       rewriter.setInsertionPoint(op);
        
  //       // Apply tiling transformation
  //       gpu::applyTiling(rewriter, op, tileSizes);
  //     }
  //   }
  // }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    llvm::errs() << "=== tile‑and‑fuse (SCF) ===\n";
  
    SmallVector<linalg::LinalgOp> marked;
    funcOp.walk([&](linalg::LinalgOp op) {
      if (op->hasAttr("gpu.test_tile_selection"))
        marked.push_back(op);
    });

    llvm::sort(marked, [](linalg::LinalgOp a, linalg::LinalgOp b) {
      return a->isBeforeInBlock(b);
    });
  
    DenseSet<Operation *> done;
  
    // for (linalg::LinalgOp consumer : marked) {
      // Process operations from consumers to producers (reverse program order)
    for (auto it = marked.rbegin(); it != marked.rend(); ++it) {
      linalg::LinalgOp consumer = *it;
      
      if (done.contains(consumer.getOperation()))
        continue;
  
      // Look for first marked producer.
      linalg::LinalgOp producer = nullptr;
      for (Value v : consumer->getOperands())
        if (auto p = v.getDefiningOp<linalg::LinalgOp>())
          if (p->hasAttr("gpu.test_tile_selection")) { producer = p; break; }

      Operation *target = producer ? producer.getOperation()
          : consumer.getOperation();
  
      // SmallVector<int64_t> ts =
      //     gpu::calculateOptimalTileSizes(producer ? producer.getOperation()
      //                                             : consumer.getOperation());

      SmallVector<int64_t> ts = gpu::calculateOptimalTileSizes(target);
      ts = mlir::gpu::fitTileSizesToOp(consumer.getOperation(), ts);
      llvm::errs() << "The ts looks like: [ ";
      for (auto s : ts)
        llvm::errs() << s << " ";
      llvm::errs() << "]\n";
      
  
      PatternRewriter rewriter(funcOp.getContext());
      rewriter.setInsertionPoint(consumer);
  
      if (producer &&
          mlir::gpu::tileAndFuseWithSCF(rewriter,
                             cast<TilingInterface>(consumer.getOperation()),
                             ts)) {
        llvm::errs() << "✓ fused and tile and fuse with SCF"
                     << producer->getName() << " → "
                     << consumer->getName() << "\n";
        done.insert(producer.getOperation());
        done.insert(consumer.getOperation());

        continue;
      } else {
        // fallback: just tile the single op
        llvm::errs() << "✓ Not fused ";
        gpu::applyTiling(rewriter, consumer.getOperation(), ts);
        done.insert(consumer.getOperation());
      }
    }

    // Process any remaining producers that were not fused
  for (linalg::LinalgOp op : marked) {
    if (!done.contains(op.getOperation())) {
      llvm::errs() << "Tiling unfused producer: " << op->getName() << "\n";
      
      PatternRewriter rewriter(funcOp.getContext());
      rewriter.setInsertionPoint(op);
      
      SmallVector<int64_t> ts = gpu::calculateOptimalTileSizes(op.getOperation());
      gpu::applyTiling(rewriter, op.getOperation(), ts);
    }
  }
  }
  
};

// Helper function implementations
static bool hasProducerConsumerRelation(Operation *producer, Operation *consumer) {
  // Check if any result of producer is used by consumer
  for (Value result : producer->getResults()) {
    for (OpOperand &use : result.getUses()) {
      if (use.getOwner() == consumer)
        return true;
    }
  }
  return false;
}

static void addPrefetching(Operation *tiledOp) {
  // For now, just log that we would add prefetching
  llvm::errs() << "Would add prefetching to operation: " << tiledOp->getName() << "\n";
  
  // Future implementation:
  // 1. Find the tiled loops (usually SCF ops created by the tiling)
  // 2. Add shared memory allocations and copies
}




} // namespace

namespace mlir {
namespace test {
void registerTestGpuTileSelectionPass() {
  PassRegistration<TestGpuTileSelectionPass>();
}
void registerTestGpuTileAndFusePass() {
  PassRegistration<TestGpuTileAndFusePass>();
}
}
} // namespace mlir