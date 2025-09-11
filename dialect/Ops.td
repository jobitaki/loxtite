include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def M_Loop_Dialect : Dialect {
    let name = "mloop";

    let summary = "A toy dialect for basic loops";

    let cppNamespace = "::mlir::mloop";
}

// Base class for mloop dialect operations. It inherits from the base Op class
// OpBase.td and provides:
//   - The parent dialect of the operation.
//   - The mnemonic for the operation, or the name without the dialect prefix
//   - A list of traits for the operations.
class M_Loop_Op<string mnemonic, list<Trait> traits = []> :
    Op<M_Loop_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// MLoop Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

def ConstantOp : M_Loop_Op<"constant", [ConstantLike, Pure]> {
    // Summary
    let summary = "constant";
    let description = [{
        Constant operation turns a literal into an SSA value. The data is 
        attached to the operation as an attribute. For example:

        ```mlir
        %0 = mloop.constant 10 : f64
        ```
    }];

    // An attribute is a constant value. $value gives it a name.
    let arguments = (ins F64Attr:$value);

    // F64 is a type. 
    let results = (outs F64:$result);

    let builders = [
        // We need only a double constant value as an argument.
        OpBuilder<(ins "double":$value)>
    ];
}

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//

def WhileOp : M_Loop_Op<"while"> {}