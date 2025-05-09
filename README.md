# MLIR-TILE-FUSE: Fusion and Prefetch Pass

A collection of MLIR compiler passes to perform tiled loop fusion and data prefetching optimized for GPU workloads. This project builds on top of the MLIR GPU dialect and introduces a single-pass solution combining:

- **Loop Tiling** to expose data locality and subdivide work into blocks and threads.
- **Cross-Operation Fusion** collapsing producer–consumer chains into single kernels.
- **Shared-Memory Prefetching** to load halo regions and intermediate buffers ahead of compute.

## Features

- **`test-gpu-tile-and-fuse` Pass**  
  Annotate operations with `gpu.test_tile_selection` to target them for fusion and tiling.
- **SCF-Based Tiling & Fusion**  
  Uses `scf.for` loops and extract/insert slice ops to build fused, tiled kernels.
- **Integration with MLIR GPU Dialect**  
  Works seamlessly with `gpu.module`, `gpu.launch`, and lowering to NVVM and LLVM.


## Building

```bash

git clone https://github.com/llvm/llvm-project.git 
git clone https://github.com/gosLp/MLIR-TILE-FUSE.git
Add all the files in this repo to the llvm-project 
cd MLIR-TILE-FUSE
mkdir build && cd build
cmake -G Ninja ../llvm -DLLVM_ENABLE_PROJECTS="mlir" -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU"   -DLLVM_BUILD_EXAMPLES=ON   -DCMAKE_BUILD_TYPE=Release   -DLLVM_ENABLE_ASSERTIONS=ON   -DCMAKE_C_COMPILER=clang   -DCMAKE_CXX_COMPILER=clang++   -DLLVM_ENABLE_LLD=ON
ninja
```

## Benchmarks
- **GEMM + Bias + ReLU** (`test/gmm.mlir`)  
- **Simple MLP** (`test/mlp.mlir`)  
- **2D Convolution** (`test/conv.mlir`)

## Run Benchmarks
```bash

 ./bin/mlir-opt   --pass-pipeline="builtin.module(func.func(test-gpu-tile-and-fuse))"   path/to/test/file
```
