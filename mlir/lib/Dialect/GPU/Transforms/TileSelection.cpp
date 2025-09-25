#include "mlir/Dialect/GPU/Transforms/TileSelection.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include <cmath>
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace mlir;
// using namespace mlir::gpu;

namespace {

/// Calculate dimensional reuse for each dimension 
static SmallVector<double> calculateDimensionalReuse(Operation *op) {
  

  SmallVector<double> reuse;
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    unsigned numDims = linalgOp.getNumLoops();
    reuse.resize(numDims, 1.0);
    
    // Calculate reuse factors for each dimension
    if (numDims >= 3) {
      reuse[0] = 0.5;  // Less reuse in outer dim
      reuse[1] = 1.0;  // Medium reuse in middle
      reuse[2] = 2.0;  // More reuse in inner 
    }
  }
  
  return reuse;
}

/// Solve the reuse polynomial to determine tile sizes
static SmallVector<int64_t> computeTileSizes(ArrayRef<double> reuse, int64_t cacheSize) {
  // based on cache size and reuse factors
  
  unsigned numDims = reuse.size();
  SmallVector<int64_t> tileSizes(numDims);
  
  // Example implementation (simplified):
  double totalReuse = 0.0;
  for (double r : reuse)
    totalReuse += r;
  
  // Distribute cache space proportionally to reuse
  for (unsigned i = 0; i < numDims; ++i) {
    double ratio = reuse[i] / totalReuse;
    int64_t tileSize = static_cast<int64_t>(std::sqrt(cacheSize * ratio));
    tileSizes[i] = std::max(int64_t(16), std::min(int64_t(256), tileSize));
  }
  
  return tileSizes;
}
} // end of namespace

SmallVector<int64_t> mlir::gpu::calculateOptimalTileSizes(Operation *op) {
  // Get reuse factors for each dimension
  SmallVector<double> reuse = calculateDimensionalReuse(op);
  
  // Compute tile sizes based on GPU L1 cache size (typically around 48KB)
  const int64_t gpuL1CacheSize = 48 * 1024;
  return computeTileSizes(reuse, gpuL1CacheSize);
}



void mlir::gpu::applyTiling(Operation *op, ArrayRef<int64_t> tileSizes) {
}