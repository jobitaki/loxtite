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
    // Note StmtPtr& means we are not copying the unique_ptr
    for (const StmtPtr& statement : stmt.statements) {
        statement->accept(*this);
    }

    std::cout << "MLIR: Added block statement to MLIR" << std::endl;
    return nullptr;
}

std::any AstLowering::visitExpressionStmt(Expression& stmt) {
    auto result = stmt.expression->accept(*this);
    std::cout << "MLIR: Added expression statement to MLIR" << std::endl;
    return result;
}

std::any AstLowering::visitFunctionStmt(Function& stmt) {
    std::cout << "MLIR: Added function statement to MLIR" << std::endl;
    return nullptr;
}

std::any AstLowering::visitIfStmt(If& stmt) {
    // (1) Evaluate the condition
    auto conditionResult = stmt.condition->accept(*this);
    mlir::Value condition = std::any_cast<mlir::Value>(conditionResult);

    // (2) Convert condition to i1 (1-bit int)
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

    // (3) Create the if operation
    auto ifOp = builder.create<mlir::scf::IfOp>(
        loc, 
        mlir::ValueRange{}, 
        boolCondition, 
        stmt.elseBranch ? 1 : 0 // If else branch is not null, add it
    );

    // (4) Create the then branch
    auto& thenRegion = ifOp.getThenRegion();
    auto* thenBlock = builder.createBlock(&thenRegion);
    builder.setInsertionPointToStart(thenBlock);

    // (5) Execute the then branch
    stmt.thenBranch->accept(*this);

    // (6) Add yield terminator to the then block
    builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{});

    // (7) If there is an else branch, create it
    if (stmt.elseBranch) {
        auto& elseRegion = ifOp.getElseRegion();
        auto* elseBlock = builder.createBlock(&elseRegion);
        builder.setInsertionPointToStart(elseBlock);

        // (7a) Execute the else branch
        stmt.elseBranch->accept(*this);

        // (7b) Add yield terminator to the else block
        builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{});
    }

    // (8) Set the insertion point after the if operation
    builder.setInsertionPointAfter(ifOp);

    std::cout << "MLIR: Added if statement to MLIR" << std::endl;
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

    std::cout << "MLIR: Added while loop to MLIR" << std::endl;
    return mlir::Value{};
}

std::any AstLowering::visitPrintStmt(Print& stmt) {
    std::cout << "MLIR: Added print statement to MLIR" << std::endl;
    return nullptr;
}

std::any AstLowering::visitReturnStmt(Return& stmt) {
    std::cout << "MLIR: Added return statement to MLIR" << std::endl;
    return nullptr;
}

std::any AstLowering::visitVarStmt(Var& stmt) {
    // (1) Allocate mutable memory on stack
    auto allocaOp = builder.create<mlir::memref::AllocaOp>(
        loc, mlir::MemRefType::get({}, builder.getF64Type())
    );

    // (2) Initialize the memory9
    mlir::Value initValue;
    if (stmt.initializer) {
        // (2.1) If initializer exists, set that as initial value.
        auto initResult = stmt.initializer->accept(*this);
        initValue = std::any_cast<mlir::Value>(initResult);
    } else {
        // (2.2) If initializer does not exist, set to zero.
        initValue = builder.create<mlir::arith::ConstantFloatOp>(
            loc, llvm::APFloat(0.0), builder.getF64Type()
        );
    }

    // (3) Store initial value
    builder.create<mlir::memref::StoreOp>(loc, initValue, allocaOp);

    // (4) Store the memory reference in symbol table
    symbolTable[stmt.name.getLexeme()] = allocaOp;

    std::cout << "MLIR: Added variable declaration '" << stmt.name.getLexeme() << "' to MLIR" << std::endl;
    return allocaOp;
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
            std::cout << "MLIR: Added addition operation to MLIR" << std::endl;
            break;
        case TokenType::MINUS:
            result = builder.create<mlir::arith::SubFOp>(loc, leftVal, rightVal);
            std::cout << "MLIR: Added subtraction operation to MLIR" << std::endl;
            break;
        case TokenType::STAR:
            result = builder.create<mlir::arith::MulFOp>(loc, leftVal, rightVal);
            std::cout << "MLIR: Added multiplication operation to MLIR" << std::endl;
            break;
        case TokenType::SLASH:
            result = builder.create<mlir::arith::DivFOp>(loc, leftVal, rightVal);
            std::cout << "MLIR: Added division operation to MLIR" << std::endl;
            break;
        default:
            std::cerr << "MLIR: Unimplemented binary expr." << std::endl;
            break;
    }

    return result;
}

std::any AstLowering::visitGroupingExpr(Grouping& expr) {
    auto result = expr.expression->accept(*this);
    std::cout << "MLIR: Added grouping expression to MLIR" << std::endl;
    return result;
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
        std::cout << "MLIR: Added literal " << val << " to MLIR" << std::endl;
    } else {
        std::cerr << "MLIR: Unimplemented literal." << std::endl;
    }

    return result;
}

std::any AstLowering::visitUnaryExpr(Unary& expr) {
    std::cout << "MLIR: Added unary expression to MLIR" << std::endl;
    return nullptr;
}

std::any AstLowering::visitCallExpr(Call& expr) {
    std::cout << "MLIR: Added function call to MLIR" << std::endl;
    return nullptr;
}

std::any AstLowering::visitVariableExpr(Variable& expr) {
    // (1) Find the variable in the symbol table.
    auto it = symbolTable.find(expr.name.getLexeme());
    if (it != symbolTable.end()) {
        // (2a) Create a load op to get the variable.
        mlir::Value result = builder.create<mlir::memref::LoadOp>(loc, it->second);
        std::cout << "MLIR: Added variable access '" << expr.name.getLexeme() << "' to MLIR" << std::endl;
        return result;
    }

    // (2b) Symbol does not exist, error handling.
    std::cerr << "MLIR: Oops, referenced variable does not exist." << std::endl;
    return nullptr;
}

std::any AstLowering::visitAssignExpr(Assign& expr) {
    // (1) Get the RHS.
    auto valueResult = expr.value->accept(*this);
    mlir::Value value = std::any_cast<mlir::Value>(valueResult);


    // (2) Find the variable in symbol table.
    auto it = symbolTable.find(expr.name.getLexeme());
    if (it != symbolTable.end()) {
        // (3a) Create a store op to assign a new value to it.
        builder.create<mlir::memref::StoreOp>(loc, value, it->second);
        std::cout << "MLIR: Added assignment to variable '" << expr.name.getLexeme() << "' to MLIR" << std::endl;
        return value;
    }

    // (3b) Symbol does not exist, error handling.

    std::cerr << "MLIR: Oops, referenced variable does not exist." << std::endl;
    return nullptr;
}