#include "AstLowering.h"

#include <iostream>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"

AstLowering::AstLowering(mlir::MLIRContext* ctx) 
    : context(ctx), builder(ctx), loc(builder.getUnknownLoc()) {
        module = mlir::ModuleOp::create(loc);
        builder.setInsertionPointToEnd(module.getBody());
}

void AstLowering::createMainFunction() {
    auto funcType = builder.getFunctionType({}, {});

    auto mainFunc = builder.create<mlir::func::FuncOp>(loc, "main", funcType);
    auto& entryBlock = *mainFunc.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);
}

std::any AstLowering::visitBlockStmt(Block& stmt) {
    return nullptr;
}

std::any AstLowering::visitExpressionStmt(Expression& stmt) {
    return stmt.expression->accept(*this);
}

std::any AstLowering::visitIfStmt(If& stmt) {
    return nullptr;
}

std::any AstLowering::visitVarStmt(Var& stmt) {
    return nullptr;
}

std::any AstLowering::visitWhileStmt(While& stmt) {
    return nullptr;
}

std::any AstLowering::visitBinaryExpr(Binary& expr) {
    auto left = expr.left->accept(*this);
    auto right = expr.right->accept(*this);

    mlir::Value leftVal = std::any_cast<mlir::Value>(left);
    mlir::Value rightVal = std::any_cast<mlir::Value>(right);

    mlir::Value result;

    switch (expr.oper.getType()) {
        case TokenType::PLUS:
            result = builder.create<mlir::arith::AddFOp>(loc, leftVal, rightVal);
            break;
        case TokenType::MINUS:
        case TokenType::STAR:
        case TokenType::SLASH:
        default:
            std::cerr << "Unimplemented binary expr." << std::endl;
            break;
    }

    return result;
}

std::any AstLowering::visitGroupingExpr(Grouping& expr) {
    return expr.expression->accept(*this);
}

std::any AstLowering::visitLiteralExpr(Literal& expr) {
    mlir::Value result; 

    auto literal = expr.value;

    if (literal.type() == typeid(double)) {
        double val = std::any_cast<double>(literal);
        auto floatType = builder.getF64Type();
        result = builder.create<mlir::arith::ConstantFloatOp>(
            loc, llvm::APFloat(val), floatType
        );
    } else {
        std::cerr << "Unimplemented literal." << std::endl;
    }

    std::cout << "Added literal into MLIR" << std::endl;
    return result;
}

std::any AstLowering::visitUnaryExpr(Unary& expr) {
    return nullptr;
}

std::any AstLowering::visitVariableExpr(Variable& expr) {
    return nullptr;
}

std::any AstLowering::visitAssignExpr(Assign& expr) {
    return nullptr;
}