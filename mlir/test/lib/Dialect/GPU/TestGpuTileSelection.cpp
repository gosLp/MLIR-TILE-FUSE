#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/TileSelection.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {
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
    llvm::errs() << "===============";
    llvm::errs() << "TILE SELECTION PASS IS RUNNING!\n";
    llvm::errs() << "===============";
    
    // Find operations to analyze and transform
    funcOp.walk([&](linalg::LinalgOp op) {
      // Only process operations with the test marker attribute
      if (!op->hasAttr("gpu.test_tile_selection"))
        return;
      
      // Calculate optimal 
      SmallVector<int64_t> tileSizes = gpu::calculateOptimalTileSizes(op);
      
      
      llvm::errs() << "Calculated tile sizes for operation: " 
                  << op->getName() << "\n";
      for (auto size : tileSizes)
        llvm::errs() << size << " ";
      llvm::errs() << "\n";
      
      // Apply tiling transformation
      gpu::applyTiling(op, tileSizes);
    });
  }
};
} // namespace

namespace mlir {
namespace test {
void registerTestGpuTileSelectionPass() {
  PassRegistration<TestGpuTileSelectionPass>();
}
}
} // namespace mlir