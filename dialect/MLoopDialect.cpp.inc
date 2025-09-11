/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: Ops.td                                                               *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::mloop::MLoopDialect)
namespace mlir {
namespace mloop {

MLoopDialect::MLoopDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<MLoopDialect>())
    
     {
  
  initialize();
}

MLoopDialect::~MLoopDialect() = default;

} // namespace mloop
} // namespace mlir
