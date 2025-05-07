// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(test-gpu-tile-and-fuse))' | FileCheck %s
module {
  func.func @main() -> tensor<64x64xf32> {
    %zero = arith.constant 0.0 : f32

    // ---- A : 64×128  -------------------------------------------------------
    %A0 = tensor.empty() : tensor<64x128xf32>
    %A  = linalg.fill ins(%zero : f32)
                    outs(%A0 : tensor<64x128xf32>) -> tensor<64x128xf32>

    // ---- B : 128×64 --------------------------------------------------------
    %B0 = tensor.empty() : tensor<128x64xf32>
    %B  = linalg.fill ins(%zero : f32)
                    outs(%B0 : tensor<128x64xf32>) -> tensor<128x64xf32>

    // ---- bias : 64×64 ------------------------------------------------------
    %bias0 = tensor.empty() : tensor<64x64xf32>
    %bias  = linalg.fill ins(%zero : f32)
                       outs(%bias0 : tensor<64x64xf32>) -> tensor<64x64xf32>

    // ---- C init ------------------------------------------------------------
    %C0 = tensor.empty() : tensor<64x64xf32>

    // ------------------------------------------------------------------
    // 1. Matmul   (producer)  – mark for our pass
    // ------------------------------------------------------------------
    %C1 = linalg.matmul {gpu.test_tile_selection}
            ins(%A, %B : tensor<64x128xf32>, tensor<128x64xf32>)
           outs(%C0   : tensor<64x64xf32>) -> tensor<64x64xf32>

    // ------------------------------------------------------------------
    // 2. Element‑wise add  (consumer) – also marked
    // ------------------------------------------------------------------
    %C2 = linalg.generic
          { indexing_maps = [
              affine_map<(i,j) -> (i,j)>,
              affine_map<(i,j) -> (i,j)>,
              affine_map<(i,j) -> (i,j)> ],
            iterator_types = ["parallel", "parallel"],
            gpu.test_tile_selection }
          ins(%C1, %bias : tensor<64x64xf32>, tensor<64x64xf32>)
          outs(%C0       : tensor<64x64xf32>) {
        ^bb0(%c : f32, %b : f32, %unused : f32):
          %sum = arith.addf %c, %b : f32
          linalg.yield %sum : f32
      } -> tensor<64x64xf32>

    return %C2 : tensor<64x64xf32>
  }
}


