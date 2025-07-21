#pragma once

#include "Expr.h"
#include "Stmt.h"

// MLIR Includes
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"


#include <unordered_map>

class AstLowering : public Visitor {
private:
    mlir::MLIRContext* context;
    mlir::OpBuilder builder;
    mlir::ModuleOp module;
    mlir::Location loc;

    std::unordered_map<std::string, mlir::Value> symbolTable;

public:
    AstLowering(mlir::MLIRContext* ctx);

    std::any visitBlockStmt(Block& stmt) override;
    std::any visitExpressionStmt(Expression& stmt) override;
    std::any visitIfStmt(If& stmt) override;
    std::any visitVarStmt(Var& stmt) override;
    std::any visitWhileStmt(While& stmt) override;
    
    std::any visitBinaryExpr(Binary& expr) override;
    std::any visitGroupingExpr(Grouping& expr) override;
    std::any visitLiteralExpr(Literal& expr) override;
    std::any visitUnaryExpr(Unary& expr) override;
    std::any visitVariableExpr(Variable& expr) override;
    std::any visitAssignExpr(Assign& expr) override;

    mlir::ModuleOp getModule() { return module; }
    void createMainFunction();
};