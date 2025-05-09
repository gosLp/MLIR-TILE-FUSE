// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(test-gpu-tile-and-fuse))' | FileCheck %s

module attributes {gpu_smem_kb = 16} {
  func.func @gmm() -> tensor<128x128xf32> {
    %zero = arith.constant 0.0 : f32
    
    // ---- A : 128×256 -------------------------------------------------------
    %A0 = tensor.empty() : tensor<128x256xf32>
    %A  = linalg.fill ins(%zero : f32)
                      outs(%A0 : tensor<128x256xf32>) -> tensor<128x256xf32>
    
    // ---- B : 256×128 --------------------------------------------------------
    %B0 = tensor.empty() : tensor<256x128xf32>
    %B  = linalg.fill ins(%zero : f32)
                      outs(%B0 : tensor<256x128xf32>) -> tensor<256x128xf32>
    
    // ---- bias : 128×128 ------------------------------------------------------
    %bias0 = tensor.empty() : tensor<128x128xf32>
    %bias  = linalg.fill ins(%zero : f32)
                         outs(%bias0 : tensor<128x128xf32>) -> tensor<128x128xf32>
    
    // ---- C init ------------------------------------------------------------
    %C0 = tensor.empty() : tensor<128x128xf32>
    
    // ------------------------------------------------------------------
    // 1. Matmul (producer) – mark for our pass
    // ------------------------------------------------------------------
    %C1 = linalg.matmul {gpu.test_tile_selection}
              ins(%A, %B : tensor<128x256xf32>, tensor<256x128xf32>)
              outs(%C0 : tensor<128x128xf32>) -> tensor<128x128xf32>
    
    // ------------------------------------------------------------------
    // 2. Element-wise add (consumer) – also marked
    // ------------------------------------------------------------------
    %C2 = linalg.generic
            { indexing_maps = [
                affine_map<(i,j) -> (i,j)>,
                affine_map<(i,j) -> (i,j)>,
                affine_map<(i,j) -> (i,j)> ],
              iterator_types = ["parallel", "parallel"],
              gpu.test_tile_selection }
            ins(%C1, %bias : tensor<128x128xf32>, tensor<128x128xf32>)
            outs(%C0 : tensor<128x128xf32>) {
          ^bb0(%c : f32, %b : f32, %unused : f32):
            %sum = arith.addf %c, %b : f32
            linalg.yield %sum : f32
        } -> tensor<128x128xf32>
    
    // ------------------------------------------------------------------
    // 3. ReLU activation (consumer) – also marked
    // ------------------------------------------------------------------
    %C3 = linalg.generic
            { indexing_maps = [
                affine_map<(i,j) -> (i,j)>,
                affine_map<(i,j) -> (i,j)> ],
              iterator_types = ["parallel", "parallel"],
              gpu.test_tile_selection }
            ins(%C2 : tensor<128x128xf32>)
            outs(%C0 : tensor<128x128xf32>) {
          ^bb0(%in : f32, %unused : f32):
            %zero_cst = arith.constant 0.0 : f32
            %relu = arith.maxf %in, %zero_cst : f32
            linalg.yield %relu : f32
        } -> tensor<128x128xf32>
        
    return %C3 : tensor<128x128xf32>
  }
}

// CHECK: func.func @gmm
// CHECK: linalg.matmul
// CHECK: linalg.generic
// CHECK: linalg.generic
