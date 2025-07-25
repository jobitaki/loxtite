<img src="loxtite.png" alt="loxtite logo" width="400" />

# Introduction
Loxtite is a WIP programming language and compiler adapted from Robert Nystrom's 
Lox from his book, _Crafting Interpreters_. Instead of writing the entire compiler stack from 
scratch, the backend is composed of MLIR and LLVM. 

This is my attempt at learning how to build a compiler frontend and use MLIR/LLVM.
My goal is to make this repo as readable and easy-to-follow as possible to aid other
beginners who want to learn how to emit MLIR.

## How to run
To build, run:
```
cd build
cmake .. -DLLVM_DIR=/opt/homebrew/lib/cmake/llvm -DMLIR_DIR=/opt/homebrew/lib/cmake/mlir
```
Then, to make and run the executable, simply run:
```
make
./loxtite path_to_script
```

## Directory Structure
```
examples - Example loxtite code
include - Header files
scripts - Scripts to generate header files
src - C++ source
```

## How to write Loxtite
Coming soon!