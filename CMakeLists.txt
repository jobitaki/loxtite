cmake_minimum_required(VERSION 3.20)
project(loxtite)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find LLVM and MLIR
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Found MLIR ${MLIR_PACKAGE_VERSION}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

# Set LLVM and MLIR directories
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

# Include LLVM and MLIR CMake modules
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# Include directories
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Separate LLVM and MLIR definitions
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

# Get MLIR and LLVM libraries
llvm_map_components_to_libnames(llvm_libs support core irreader)

# Source files
set(SOURCES
    src/main.cpp
    src/Scanner.cpp
    src/Parser.cpp
    src/Token.cpp
    src/AstLowering.cpp
    src/Loxtite.cpp
    src/AstPrinter.cpp
    # Add other source files as needed
)

# Create executable
add_executable(loxtite ${SOURCES})

# Link MLIR and LLVM libraries
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

target_link_libraries(loxtite
    PRIVATE
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}
    MLIROptLib
    MLIRParser
    MLIRPass
    MLIRTransforms
    MLIRTransformUtils
    MLIRSupport
    MLIRAnalysis
    MLIRIR
    MLIRDialect
    MLIRFuncDialect
    MLIRArithDialect
    MLIRMemRefDialect
    MLIRSCFDialect
    ${llvm_libs}
)

# Compiler flags
target_compile_features(loxtite PRIVATE cxx_std_17)

# Optional: Add debug information
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(loxtite PRIVATE -g -O0)
endif()