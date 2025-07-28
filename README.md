<img src="loxtite.png" alt="loxtite logo" width="400" />

# Introduction
Loxtite is a WIP programming language and compiler adapted from Robert Nystrom's Lox from his book, _Crafting Interpreters_. Instead of writing the entire compiler stack from scratch, the backend is composed of MLIR and LLVM. 

This is my attempt at learning how to build a compiler frontend and use MLIR/LLVM. My goal is to make this repo as readable and easy-to-follow as possible to aid other beginners who want to learn how to emit MLIR.

## How to run
Make sure you have LLVM and MLIR installed. 

To build, run:
```
cd build
cmake .. -DLLVM_DIR=/opt/homebrew/lib/cmake/llvm -DMLIR_DIR=/opt/homebrew/lib/cmake/mlir
```
Then, to make and run the executable, simply run:
```
make -j$(sysctl -n hw.ncpu)
./loxtite path_to_script
```
This will output unoptimized MLIR into the console at the present state. The goal
is to implement optimization passes, lower into LLVM, and create an executable from
you script or run it directly. 

## Directory Structure
```
examples - Example loxtite code
include - Header files
scripts - Scripts to generate header files
src - C++ source
```

## How to write Loxtite
Coming soon!