// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(test-gpu-tile-and-fuse))' | FileCheck %s

module attributes {gpu_smem_kb = 32} {
  func.func @mlp(%input: tensor<128x64xf32>) -> tensor<128x32xf32> {
    %zero = arith.constant 0.0 : f32
    
    // ---- Layer 1 weights and bias -----------------------------------------
    %W1_0 = tensor.empty() : tensor<64x128xf32>
    %W1 = linalg.fill ins(%zero : f32)
                      outs(%W1_0 : tensor<64x128xf32>) -> tensor<64x128xf32>
    
    %b1_0 = tensor.empty() : tensor<128xf32>
    %b1 = linalg.fill ins(%zero : f32)
                      outs(%b1_0 : tensor<128xf32>) -> tensor<128xf32>
    
    // ---- Layer 2 weights and bias -----------------------------------------
    %W2_0 = tensor.empty() : tensor<128x32xf32>
    %W2 = linalg.fill ins(%zero : f32)
                      outs(%W2_0 : tensor<128x32xf32>) -> tensor<128x32xf32>
    
    %b2_0 = tensor.empty() : tensor<32xf32>
    %b2 = linalg.fill ins(%zero : f32)
                      outs(%b2_0 : tensor<32xf32>) -> tensor<32xf32>
    
    // ---- Intermediate tensors ---------------------------------------------
    %hidden_0 = tensor.empty() : tensor<128x128xf32>
    %hidden_1 = tensor.empty() : tensor<128x128xf32>
    %output_0 = tensor.empty() : tensor<128x32xf32>
    
    // ------------------------------------------------------------------
    // 1. First layer matmul
    // ------------------------------------------------------------------
    %hidden = linalg.matmul {gpu.test_tile_selection}
                  ins(%input, %W1 : tensor<128x64xf32>, tensor<64x128xf32>)
                  outs(%hidden_0 : tensor<128x128xf32>) -> tensor<128x128xf32>
    
    // ------------------------------------------------------------------
    // 2. First layer bias add
    // ------------------------------------------------------------------
    %biased_hidden = linalg.generic
                      { indexing_maps = [
                          affine_map<(i,j) -> (i,j)>,
                          affine_map<(i,j) -> (j)>,
                          affine_map<(i,j) -> (i,j)> ],
                        iterator_types = ["parallel", "parallel"],
                        gpu.test_tile_selection }
                      ins(%hidden, %b1 : tensor<128x128xf32>, tensor<128xf32>)
                      outs(%hidden_1 : tensor<128x128xf32>) {
                    ^bb0(%h : f32, %b : f32, %unused : f32):
                      %sum = arith.addf %h, %b : f32
                      linalg.yield %sum : f32
                  } -> tensor<128x128xf32>
    
    // ------------------------------------------------------------------
    // 3. First layer ReLU activation
    // ------------------------------------------------------------------
    %relu_hidden = linalg.generic
                    { indexing_maps = [
                        affine_map<(i,j) -> (i,j)>,
                        affine_map<(i,j) -> (i,j)> ],
                      iterator_types = ["parallel", "parallel"],
                      gpu.test_tile_selection }
                    ins(%biased_hidden : tensor<128x128xf32>)
                    outs(%hidden_1 : tensor<128x128xf32>) {
                  ^bb0(%in : f32, %unused : f32):
                    %zero_cst = arith.constant 0.0 : f32
                    %relu = arith.maxf %in, %zero_cst : f32
                    linalg.yield %relu : f32
                } -> tensor<128x128xf32>
    
    // ------------------------------------------------------------------
    // 4. Second layer matmul
    // ------------------------------------------------------------------
    %output = linalg.matmul {gpu.test_tile_selection}
                  ins(%relu_hidden, %W2 : tensor<128x128xf32>, tensor<128x32xf32>)
                  outs(%output_0 : tensor<128x32xf32>) -> tensor<128x32xf32>
    
    // ------------------------------------------------------------------
    // 5. Second layer bias add
    // ------------------------------------------------------------------
    %biased_output = linalg.generic
                      { indexing_maps = [
                          affine_map<(i,j) -> (i,j)>,
                          affine_map<(i,j) -> (j)>,
                          affine_map<(i,j) -> (i,j)> ],
                        iterator_types = ["parallel", "parallel"],
                        gpu.test_tile_selection }
                      ins(%output, %b2 : tensor<128x32xf32>, tensor<32xf32>)
                      outs(%output_0 : tensor<128x32xf32>) {
                    ^bb0(%o : f32, %b : f32, %unused : f32):
                      %sum = arith.addf %o, %b : f32
                      linalg.yield %sum : f32
                  } -> tensor<128x32xf32>
                  
    return %biased_output : tensor<128x32xf32>
  }
}

// CHECK: func.func @mlp
// CHECK: linalg.matmul
// CHECK: linalg.generic
// CHECK: linalg.generic
// CHECK: linalg.matmul
// CHECK: linalg.generic
