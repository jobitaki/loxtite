/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: Ops.td                                                               *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace mloop {

class MLoopDialect : public ::mlir::Dialect {
  explicit MLoopDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~MLoopDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("mloop");
  }
};
} // namespace mloop
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::mloop::MLoopDialect)
