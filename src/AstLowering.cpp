#include "AstLowering.h"

#include <iostream>

AstLowering::AstLowering(mlir::MLIRContext* ctx) 
    : context(ctx), builder(ctx), loc(builder.getUnknownLoc()), llvmContext() {
    module = mlir::ModuleOp::create(loc);
    builder.setInsertionPointToEnd(module.getBody());
    // For global scope
    pushScope();
}

void AstLowering::createMainFunction() {
    auto funcType = builder.getFunctionType({}, {});
    auto mainFunc = builder.create<mlir::func::FuncOp>(loc, "main", funcType);
    auto& entryBlock = *mainFunc.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);
    // For main function scope
    pushScope();
}

void AstLowering::finishMainFunction() {
    builder.create<mlir::func::ReturnOp>(loc);
    popScope();
}

void AstLowering::lowerToLLVM() {
    // (1) Create a pass manager
    mlir::PassManager pm(context);

    // (2) Add conversion passes
    pm.addPass(mlir::createConvertSCFToCFPass());
    pm.addPass(mlir::createArithToLLVMConversionPass());
    pm.addPass(mlir::createConvertControlFlowToLLVMPass());
    pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    
    // (3) Run the passes
    if (mlir::failed(pm.run(module))) {
        std::cerr << "Failed to lower to LLVM" << std::endl;
        return;
    }

    std::cout << "Successfully lowered to LLVM Dialect" << std::endl;
}

std::unique_ptr<llvm::Module> AstLowering::convertToLLVMIR() {
    auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
    if (!llvmModule) {
        std::cerr << "Failed to translate MLIR module to LLVM" << std::endl;
        return nullptr;
    }

    return llvmModule;
}

std::any AstLowering::visitBlockStmt(Block& stmt) {
    pushScope();

    // Note StmtPtr& means we are not copying the unique_ptr
    for (const StmtPtr& statement : stmt.statements) {
        statement->accept(*this);
    }

    popScope();

    // std::cout << "MLIR: Added block statement to MLIR" << std::endl;
    return nullptr;
}

std::any AstLowering::visitExpressionStmt(Expression& stmt) {
    auto result = stmt.expression->accept(*this);
    // std::cout << "MLIR: Added expression statement to MLIR" << std::endl;
    return result;
}

std::any AstLowering::visitFunctionStmt(Function& stmt) {
    // (1) Build parameter types (All are f64 for now)
    std::vector<mlir::Type> paramTypes;
    for (const Token& param : stmt.params) {
        paramTypes.push_back(builder.getF64Type());
    }

    // (2) Build the function type (functions must return f64 for now)
    auto funcType = builder.getFunctionType(paramTypes, {builder.getF64Type()});

    // (3) Save current insertion point -- We do this because everything else
    //     is inserted within the main function body. We want to exit temporarily.
    auto savedIP = builder.saveInsertionPoint();

    // (4) Set the insertion point to end of module body (start is also fine)
    builder.setInsertionPointToEnd(module.getBody());

    // (5) Create the function operation
    auto funcOp = builder.create<mlir::func::FuncOp>(
        loc, 
        stmt.name.getLexeme(),
        funcType
    );

    // (6) Add the function body block and set the insertion point to it.
    //     * deferences the pointer, & makes a reference not a copy.
    auto& entryBlock = *funcOp.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);

    // (7) Push new scope for function parameters
    pushScope();

    // (8) Add parameters to the symbol table and create store ops.
    for (size_t i = 0; i < stmt.params.size(); ++i) {
        auto paramAlloca = builder.create<mlir::memref::AllocaOp>(
            loc, mlir::MemRefType::get({}, builder.getF64Type())
        );

        builder.create<mlir::memref::StoreOp>(
            loc, entryBlock.getArgument(i), paramAlloca
        );

        addVariable(stmt.params[i].getLexeme(), paramAlloca);
    }

    // (9) Generate function body
    for (const auto& statement : stmt.body) {
        statement->accept(*this);
    }

    // (10) If there was no return add one.
    auto* currentBlock = builder.getInsertionBlock();
    if (currentBlock->empty() || !currentBlock->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
        mlir::Value defaultReturn = builder.create<mlir::arith::ConstantFloatOp>(
            loc, llvm::APFloat(0.0), builder.getF64Type()
        );
        builder.create<mlir::func::ReturnOp>(loc, defaultReturn);
    }

    // (11) Pop scope
    popScope();

    // (12) Restore insertion point
    builder.restoreInsertionPoint(savedIP);

    // std::cout << "MLIR: Added function statement to MLIR" << std::endl;
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
        mlir::TypeRange{}, 
        boolCondition, 
        stmt.elseBranch != nullptr // If else branch is not null, add it
    );

    // (4) Create the then branch
    auto& thenRegion = ifOp.getThenRegion();
    auto& thenBlock = thenRegion.front();
    builder.setInsertionPointToStart(&thenBlock);

    // (5) Execute the then branch
    stmt.thenBranch->accept(*this);

    // (6) Add yield terminator to the then block
    // builder.create<mlir::scf::YieldOp>(loc);

    // (7) If there is an else branch, create it
    if (stmt.elseBranch) {
        auto& elseRegion = ifOp.getElseRegion();
        auto& elseBlock = elseRegion.front();
        builder.setInsertionPointToStart(&elseBlock);

        // (7a) Execute the else branch
        stmt.elseBranch->accept(*this);

        // (7b) Add yield terminator to the else block
        // builder.create<mlir::scf::YieldOp>(loc);
    }

    // (8) Set the insertion point after the if operation
    builder.setInsertionPointAfter(ifOp);

    // std::cout << "MLIR: Added if statement to MLIR" << std::endl;
    return nullptr;
}

std::any AstLowering::visitWhileStmt(While& stmt) {
    // (1) Create a while operation
    auto whileOp = builder.create<mlir::scf::WhileOp>(
        loc,
        mlir::TypeRange{},
        mlir::ValueRange{}
    );

    // (2) Create the before, condition check region.
    // (a) Gets the before region of the while op, reference because copy not
    //     allowed. We get this to add blocks to it or manipulate it.
    auto& beforeRegion = whileOp.getBefore();
    // (b) Creates a new basic block at the before region. It returns a pointer
    //     to it, hence the auto* (for style). &beforeRegion because we want to
    //     pass in a pointer to the reference.
    auto* beforeBlock = builder.createBlock(&beforeRegion);
    // (c) Set the builder to start inserting at the start of beforeBlock.
    builder.setInsertionPointToStart(beforeBlock);

    // (3) Execute the condition expression and get it as a mlir::Value
    auto conditionResult = stmt.condition->accept(*this);
    mlir::Value condition = std::any_cast<mlir::Value>(conditionResult);

    // (4) Convert condition to i1 (1-bit int)
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

    // (5) Create condition branch
    builder.create<mlir::scf::ConditionOp>(
        loc, 
        boolCondition, 
        mlir::ValueRange{} // No loop carried values yielded.
    );

    // (6) Create after region (loop body)
    // (a) Get reference to the while loop's after region.
    auto& afterRegion = whileOp.getAfter();
    // (b) Create a new basic block for it at afterRegion.
    auto* afterBlock = builder.createBlock(&afterRegion);
    // (c) Set the builder insertion point to there.
    builder.setInsertionPointToStart(afterBlock);

    // (7) Execute the loop body
    stmt.body->accept(*this);
    
    // (8) Terminator. No loop carried values for now.
    builder.create<mlir::scf::YieldOp>(
        loc, mlir::ValueRange{}
    );

    // (9) Set the insertion point after the while operation
    builder.setInsertionPointAfter(whileOp);

    // std::cout << "MLIR: Added while loop to MLIR" << std::endl;
    return mlir::Value{};
}

std::any AstLowering::visitPrintStmt(Print& stmt) {
    // (1) Evaluate the expression to print
    auto exprResult = stmt.expression->accept(*this);
    mlir::Value value = std::any_cast<mlir::Value>(exprResult);

    // (2) Create a global string constant for the format
    static int formatStrCounter = 0;
    std::string formatName = "format_str_" + std::to_string(formatStrCounter++);
    
    auto savedIP = builder.saveInsertionPoint();
    builder.setInsertionPointToEnd(module.getBody());
    
    auto formatStr = builder.create<mlir::LLVM::GlobalOp>(
        loc,
        mlir::LLVM::LLVMArrayType::get(builder.getI8Type(), 5), // "%.6g\n\0"
        /*isConstant=*/true,
        mlir::LLVM::Linkage::Private,
        formatName,
        builder.getStringAttr("%.6g\n")
    );
    
    builder.restoreInsertionPoint(savedIP);

    // (3) Get address of format string
    auto formatAddr = builder.create<mlir::LLVM::AddressOfOp>(loc, formatStr);
    
    // (4) Cast to i8*
    auto i8PtrType = mlir::LLVM::LLVMPointerType::get(context);
    auto formatPtr = builder.create<mlir::LLVM::BitcastOp>(
        loc, i8PtrType, formatAddr
    );

    // (5) Declare printf if not already declared
    auto printfFunc = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>("printf");
    if (!printfFunc) {
        auto savedIP2 = builder.saveInsertionPoint();
        builder.setInsertionPointToEnd(module.getBody());
        
        auto printfType = mlir::LLVM::LLVMFunctionType::get(
            builder.getI32Type(), // return type
            {i8PtrType}, // parameters
            /*isVarArg=*/true
        );
        
        printfFunc = builder.create<mlir::LLVM::LLVMFuncOp>(
            loc, "printf", printfType
        );
        
        builder.restoreInsertionPoint(savedIP2);
    }

    // (6) Call printf
    builder.create<mlir::LLVM::CallOp>(
        loc, printfFunc, mlir::ValueRange{formatPtr, value}
    );

    // std::cout << "MLIR: Added print statement to MLIR" << std::endl;
    return nullptr;
}

std::any AstLowering::visitReturnStmt(Return& stmt) {
    mlir::Value returnValue;

    if (stmt.expression) {
        auto result = stmt.expression->accept(*this);
        returnValue = std::any_cast<mlir::Value>(result);
    } else {
        returnValue = builder.create<mlir::arith::ConstantFloatOp>(
            loc, llvm::APFloat(0.0), builder.getF64Type()
        );
    }

    builder.create<mlir::func::ReturnOp>(loc, returnValue);

    // std::cout << "MLIR: Added return statement to MLIR" << std::endl;
    return nullptr;
}

std::any AstLowering::visitVarStmt(Var& stmt) {
    // (1) Allocate mutable memory on stack as opposed to on heap
    auto allocaOp = builder.create<mlir::memref::AllocaOp>(
        loc, mlir::MemRefType::get({}, builder.getF64Type())
    );

    // (2) Initialize the memory
    mlir::Value initValue;
    if (stmt.initializer) {
        // (2a) If initializer exists, set that as initial value.
        auto initResult = stmt.initializer->accept(*this);
        initValue = std::any_cast<mlir::Value>(initResult);
    } else {
        // (2b) If initializer does not exist, set to zero.
        initValue = builder.create<mlir::arith::ConstantFloatOp>(
            loc, llvm::APFloat(0.0), builder.getF64Type()
        );
    }

    // (3) Store initial value
    builder.create<mlir::memref::StoreOp>(loc, initValue, allocaOp);

    // (4) Store the memory reference in symbol table
    addVariable(stmt.name.getLexeme(), allocaOp);

    // std::cout << "MLIR: Added variable declaration '" << stmt.name.getLexeme() << "' to MLIR" << std::endl;
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
            // std::cout << "MLIR: Added addition operation to MLIR" << std::endl;
            break;
        case TokenType::MINUS:
            result = builder.create<mlir::arith::SubFOp>(loc, leftVal, rightVal);
            // std::cout << "MLIR: Added subtraction operation to MLIR" << std::endl;
            break;
        case TokenType::STAR:
            result = builder.create<mlir::arith::MulFOp>(loc, leftVal, rightVal);
            // std::cout << "MLIR: Added multiplication operation to MLIR" << std::endl;
            break;
        case TokenType::SLASH:
            result = builder.create<mlir::arith::DivFOp>(loc, leftVal, rightVal);
            // std::cout << "MLIR: Added division operation to MLIR" << std::endl;
            break;
        case TokenType::LESS:
            result = builder.create<mlir::arith::CmpFOp>(
                loc, mlir::arith::CmpFPredicate::OLT, leftVal, rightVal
            );
            break;
        case TokenType::LESS_EQUAL:
            result = builder.create<mlir::arith::CmpFOp>(
                loc, mlir::arith::CmpFPredicate::OLE, leftVal, rightVal
            );
            break;
        case TokenType::GREATER:
            result = builder.create<mlir::arith::CmpFOp>(
                loc, mlir::arith::CmpFPredicate::OGT, leftVal, rightVal
            );
            break;
        case TokenType::GREATER_EQUAL:
            result = builder.create<mlir::arith::CmpFOp>(
                loc, mlir::arith::CmpFPredicate::OGE, leftVal, rightVal
            );
            break;
        case TokenType::EQUAL_EQUAL:
            result = builder.create<mlir::arith::CmpFOp>(
                loc, mlir::arith::CmpFPredicate::OEQ, leftVal, rightVal
            );
            break;
        case TokenType::BANG_EQUAL:
            result = builder.create<mlir::arith::CmpFOp>(
                loc, mlir::arith::CmpFPredicate::ONE, leftVal, rightVal
            );
            break;
        default:
            std::cerr << "MLIR: Unimplemented binary expr." << std::endl;
            break;
    }

    return result;
}

std::any AstLowering::visitGroupingExpr(Grouping& expr) {
    auto result = expr.expression->accept(*this);
    // std::cout << "MLIR: Added grouping expression to MLIR" << std::endl;
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
        // std::cout << "MLIR: Added literal " << val << " to MLIR" << std::endl;
    } else {
        std::cerr << "MLIR: Unimplemented literal." << std::endl;
    }

    return result;
}

std::any AstLowering::visitUnaryExpr(Unary& expr) {
    // std::cout << "MLIR: Added unary expression to MLIR" << std::endl;
    return nullptr;
}

std::any AstLowering::visitCallExpr(Call& expr) {
    // (1) Get the function name from callee.
    //     Dynamic cast tries to safely cast Expr* into Variable*.
    //     get() is a unique_ptr member function that extracts the raw pointer.
    auto calleeVar = dynamic_cast<Variable*>(expr.callee.get());
    if (!calleeVar) {
        std::cerr << "MLIR: Only direct function calls supported" << std::endl;
        return nullptr;
    }

    std::string functionName = calleeVar->name.getLexeme();

    // (2) Lookup function in module.
    auto funcOp = module.lookupSymbol<mlir::func::FuncOp>(functionName);
    if (!funcOp) {
        std::cerr << "MLIR: Function '" << functionName << "' not found" << std::endl;
        return nullptr;
    }

    // (3) Evaluate all arguments.
    std::vector<mlir::Value> arguments;
    for (const auto& arg : expr.arguments) {
        auto argResult = arg->accept(*this);
        mlir::Value argValue = std::any_cast<mlir::Value>(argResult);
        arguments.push_back(argValue);
    }

    // (4) Create the function call op.
    auto callOp = builder.create<mlir::func::CallOp>(
        loc,
        funcOp, // Function to call
        arguments
    );

    // (5) Get the result (TODO handle empty return)
    mlir::Value result = callOp.getResult(0);

    // std::cout << "MLIR: Added function call '" << functionName << "' to MLIR" << std::endl;
    return result;
}

std::any AstLowering::visitVariableExpr(Variable& expr) {
    // (1) Find the variable in the symbol table. Throws error if not found.
    auto it = lookupVariable(expr.name.getLexeme());
    
    // (2) Create a load op to get the variable.
    mlir::Value result = builder.create<mlir::memref::LoadOp>(loc, it);
    // std::cout << "MLIR: Added variable access '" << expr.name.getLexeme() << "' to MLIR" << std::endl;
    return result;
}

std::any AstLowering::visitAssignExpr(Assign& expr) {
    // (1) Get the RHS.
    auto valueResult = expr.value->accept(*this);
    mlir::Value value = std::any_cast<mlir::Value>(valueResult);


    // (2) Find the variable in symbol table.
    auto it = lookupVariable(expr.name.getLexeme());

    // (3) Create a store op to assign a new value to it.
    builder.create<mlir::memref::StoreOp>(loc, value, it);
    // std::cout << "MLIR: Added assignment to variable '" << expr.name.getLexeme() << "' to MLIR" << std::endl;
    return value;
}