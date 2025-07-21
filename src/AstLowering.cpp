#include "AstLowering.h"

#include <iostream>

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
    
    auto whileOp = builder.create<mlir::scf::WhileOp>(
        loc,
        mlir::TypeRange{},
        mlir::ValueRange{}
    );

    // Condition check region
    // (1) Gets the before region of the while op, reference because copy not
    //     allowed. We get this to add blocks to it or manipulate it.
    auto& beforeRegion = whileOp.getBefore();
    // (2) Creates a new basic block at the before region. It returns a pointer
    //     to it, hence the auto* (for style). &beforeRegion because we want to
    //     pass in a pointer to the reference.
    auto* beforeBlock = builder.createBlock(&beforeRegion);
    // (3) Set the builder to start inserting at the start of beforeBlock.
    builder.setInsertionPointToStart(beforeBlock);

    // Condition
    auto conditionResult = stmt.condition->accept(*this);
    mlir::Value condition = std::any_cast<mlir::Value>(conditionResult);

    // Convert condition to i1 (1-bit int)
    mlir::Value boolCondition; 
    if (condition.getType().isF64()) {
        // Creates a zero
        auto zero = builder.create<mlir::arith::ConstantFloatOp>(
            loc, llvm::APFloat(0.0), builder.getF64Type()
        );
        // If condition != 0 return true
        boolCondition = builder.create<mlir::arith::CmpFOp>(
            loc, mlir::arith::CmpFPredicate::ONE, condition, zero
        );
    } else {
        boolCondition = condition; // Assume it's i1 for now
    }

    // Create condition branch
    builder.create<mlir::scf::ConditionOp>(
        loc, 
        boolCondition, 
        mlir::ValueRange{} // No loop carried values yielded.
    );

    // Create after region (loop body)
    // (1) Get reference to the while loop's after region.
    auto& afterRegion = whileOp.getAfter();
    // (2) Create a new basic block for it at afterRegion.
    auto* afterBlock = builder.createBlock(&afterRegion);
    // (3) Set the builder insertion point to there.
    builder.setInsertionPointToStart(afterBlock);

    // Execute the loop body
    stmt.body->accept(*this);
    
    // Terminator. No loop carried values for now.
    builder.create<mlir::scf::YieldOp>(
        loc, mlir::ValueRange{}
    );

    builder.setInsertionPointAfter(whileOp);

    return mlir::Value{};
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
            result = builder.create<mlir::arith::SubFOp>(loc, leftVal, rightVal);
            break;
        case TokenType::STAR:
            result = builder.create<mlir::arith::MulFOp>(loc, leftVal, rightVal);
            break;
        case TokenType::SLASH:
            result = builder.create<mlir::arith::DivFOp>(loc, leftVal, rightVal);
            break;
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