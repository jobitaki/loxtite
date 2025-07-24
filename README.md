<img src="media/loxtite.png" alt="loxtite logo" width="400" />

# Introduction
Loxtite is a WIP programming language and compiler adapted from Robert Nystrom's 
Lox from his book, _Crafting Interpreters_. Instead of writing the entire compiler stack from 
scratch, the backend is composed of MLIR and LLVM. This is my attempt at 
learning how to build a compiler frontend and use MLIR/LLVM.

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

## How to write Loxtite
Coming soon!