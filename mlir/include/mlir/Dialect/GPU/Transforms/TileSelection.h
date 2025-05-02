#ifndef MLIR_DIALECT_GPU_TRANSFORMS_TILESELECTION_H
#define MLIR_DIALECT_GPU_TRANSFORMS_TILESELECTION_H


#include "mlir/Support/LLVM.h"
#include "mlir/IR/Operation.h"

namespace mlir {

class MemRefType;
class Value;
class Operation;

namespace gpu {
class GPUFuncOp;

/// Calculate optimal tile sizes for the given operation based on reuse analysis.
/// Returns a vector of tile sizes for each dimension.
SmallVector<int64_t> calculateOptimalTileSizes(Operation *op);

/// Apply tiling transformation using the calculated tile sizes.
void applyTiling(Operation *op, ArrayRef<int64_t> tileSizes);

} // namespace gpu
} // namespace mlir

#endif // MLIR_DIALECT_GPU_TRANSFORMS_TILESELECTION_H