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
This will create two files, out.mlir and out.ll. Respectively, they contain the unoptimized MLIR code, and the MLIR code lowered to LLVM IR. To run your code with the LLVM interpreter, run:
```
lli out.ll
```

## Directory Structure
```
examples - Example loxtite code
include - Header files
scripts - Scripts to generate header files
src - C++ source
```

## How to write Loxtite
```
// Declare variables. Only floats supported for now.
var x = 0.0;

// Assign variables.
x = 10.0;

// Arithmetic
(1.0 + 2.0) * 3.0 / (4.0 - 6.0);

// Inequality
1.0 <= 2.0; /* Returns 1-bit integer, so use for conditionals for now. */

// Control flow
if (x) {
    ... 
    /* No returns inside of body */
    ...
} else { 
    ...
    /* No else ifs supported for now */
    ...
}

while (...) {
    ...
    /* No returns inside of body */
    ...
}

// Functions
fun foo(arg1, arg2) {
    return arg1 + arg2;
}

foo(1.0, 2.0);

// Built in functions
print(0.0); /* Print only supports float for now */
print(x);

```