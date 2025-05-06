#ifndef MLIR_DIALECT_GPU_TRANSFORMS_TILESELECTION_H
#define MLIR_DIALECT_GPU_TRANSFORMS_TILESELECTION_H


#include "mlir/Support/LLVM.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"

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
//void applyTiling(Operation *op, ArrayRef<int64_t> tileSizes);
/// Apply tiling transformation using the calculated tile sizes.
/// This overload drives the underlying PatternRewriter.
// void applyTiling(mlir::PatternRewriter &rewriter,
//     Operation *op,
//     llvm::ArrayRef<int64_t> tileSizes);
void applyTiling(::mlir::PatternRewriter &rewriter,   // fully qualified
    Operation *op,
    llvm::ArrayRef<int64_t> tileSizes);


/// Optional convenience wrapper if you need to call without an existing rewriter.
// inline void applyTiling(Operation *op, llvm::ArrayRef<int64_t> tileSizes) {
//     mlir::PatternRewriter rewriter(op->getContext());
//     rewriter.setInsertionPoint(op);
//     applyTiling(rewriter, op, tileSizes);
//   }
/// Convenience wrapper that builds its own rewriter.
void applyTiling(Operation *op, llvm::ArrayRef<int64_t> tileSizes);

} // namespace gpu
} // namespace mlir

#endif // MLIR_DIALECT_GPU_TRANSFORMS_TILESELECTION_H