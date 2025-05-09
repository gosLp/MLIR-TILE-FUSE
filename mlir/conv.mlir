// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(test-gpu-tile-and-fuse))' | FileCheck %s

module attributes {gpu_smem_kb = 48} {
  func.func @conv2d() -> tensor<1x32x28x28xf32> {
    %zero = arith.constant 0.0 : f32
    
    // ---- Input: N=1, C=3, H=32, W=32 ------------------------------------
    %input0 = tensor.empty() : tensor<1x3x32x32xf32>
    %input = linalg.fill ins(%zero : f32)
                        outs(%input0 : tensor<1x3x32x32xf32>) -> tensor<1x3x32x32xf32>
    
    // ---- Filter: OC=32, IC=3, KH=5, KW=5 --------------------------------
    %filter0 = tensor.empty() : tensor<32x3x5x5xf32>
    %filter = linalg.fill ins(%zero : f32)
                         outs(%filter0 : tensor<32x3x5x5xf32>) -> tensor<32x3x5x5xf32>
    
    // ---- Bias: OC=32 ----------------------------------------------------
    %bias0 = tensor.empty() : tensor<32xf32>
    %bias = linalg.fill ins(%zero : f32)
                       outs(%bias0 : tensor<32xf32>) -> tensor<32xf32>
    
    // ---- Output tensors -------------------------------------------------
    %conv_out0 = tensor.empty() : tensor<1x32x28x28xf32>
    %final_out0 = tensor.empty() : tensor<1x32x28x28xf32>
    
    // ------------------------------------------------------------------
    // 1. 2D Convolution (producer)
    // ------------------------------------------------------------------
    %conv_out = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>,
                                         strides = dense<1> : tensor<2xi64>,
                                         gpu.test_tile_selection}
                    ins(%input, %filter : tensor<1x3x32x32xf32>, tensor<32x3x5x5xf32>)
                    outs(%conv_out0 : tensor<1x32x28x28xf32>) -> tensor<1x32x28x28xf32>
    
    // ------------------------------------------------------------------
    // 2. Add bias (consumer)
    // ------------------------------------------------------------------
    %bias_out = linalg.generic
                  { indexing_maps = [
                      affine_map<(n,c,h,w) -> (n,c,h,w)>,
                      affine_map<(n,c,h,w) -> (c)>,
                      affine_map<(n,c,h,w) -> (n,c,h,w)> ],
                    iterator_types = ["parallel", "parallel", "parallel", "parallel"],
                    gpu.test_tile_selection }
                  ins(%conv_out, %bias : tensor<1x32x28x28xf32>, tensor<32xf32>)
                  outs(%final_out0 : tensor<1x32x28x28xf32>) {
                ^bb0(%c : f32, %b : f32, %unused : f32):
                  %sum = arith.addf %c, %b : f32
                  linalg.yield %sum : f32
              } -> tensor<1x32x28x28xf32>
    
    // ------------------------------------------------------------------
    // 3. ReLU activation (consumer)
    // ------------------------------------------------------------------
    %relu_out = linalg.generic
                  { indexing_maps = [
                      affine_map<(n,c,h,w) -> (n,c,h,w)>,
                      affine_map<(n,c,h,w) -> (n,c,h,w)> ],
                    iterator_types = ["parallel", "parallel", "parallel", "parallel"],
                    gpu.test_tile_selection }
                  ins(%bias_out : tensor<1x32x28x28xf32>)
                  outs(%final_out0 : tensor<1x32x28x28xf32>) {
                ^bb0(%in : f32, %unused : f32):
                  %zero_cst = arith.constant 0.0 : f32
                  %relu = arith.maxf %in, %zero_cst : f32
                  linalg.yield %relu : f32
              } -> tensor<1x32x28x28xf32>
    
    return %relu_out : tensor<1x32x28x28xf32>
  }
}

// CHECK: func.func @conv2d
// CHECK: linalg.conv_2d_nchw_fchw
// CHECK: linalg.generic
// CHECK: linalg.generic
