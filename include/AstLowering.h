#pragma once

#include "Expr.h"
#include "Stmt.h"

// Custom Dialect Includes
#include "MLoopDialect.h"

// MLIR Includes
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"

// MLIR Pass Includes
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/ControlFlowToSCF/ControlFlowToSCF.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir//Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/GenericValue.h"

#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"

#include <unordered_map>

class AstLowering : public Visitor {
private:
    mlir::MLIRContext* context;
    mlir::OpBuilder builder;
    mlir::ModuleOp module;
    mlir::Location loc;
    llvm::LLVMContext llvmContext;

    // std::vector<std::unordered_map<std::string, mlir::Value>> symbolTableStack;
    std::vector<std::unordered_map<std::string, mlir::Value>> ssaVarStack;

    void pushScope() {
        ssaVarStack.emplace_back();
    }

    void popScope() {
        if (!ssaVarStack.empty()) {
            ssaVarStack.pop_back();
        }
    }

    mlir::Value lookupVariable(const std::string& name) {
        for (auto it = ssaVarStack.rbegin(); it != ssaVarStack.rend(); ++it) {
            auto found = it->find(name);
            if (found != it->end()) {
                return found->second;
            }
        }
        throw std::runtime_error("Variable not found: " + name);
    }

    void addVariable(const std::string& name, mlir::Value value) {
        if (!ssaVarStack.empty()) {
            ssaVarStack.back()[name] = value;
        } else {
            throw std::runtime_error("No active scope to add variable: " + name);
        }
    }

public:
    AstLowering(mlir::MLIRContext* ctx);

    std::any visitBlockStmt(Block& stmt) override;
    std::any visitExpressionStmt(Expression& stmt) override;
    std::any visitFunctionStmt(Function& stmt) override;
    std::any visitIfStmt(If& stmt) override;
    std::any visitWhileStmt(While& stmt) override;
    std::any visitPrintStmt(Print& stmt) override;
    std::any visitReturnStmt(Return& stmt) override;
    std::any visitVarStmt(Var& stmt) override;
    
    std::any visitBinaryExpr(Binary& expr) override;
    std::any visitGroupingExpr(Grouping& expr) override;
    std::any visitLiteralExpr(Literal& expr) override;
    std::any visitUnaryExpr(Unary& expr) override;
    std::any visitCallExpr(Call& expr) override;
    std::any visitVariableExpr(Variable& expr) override;
    std::any visitAssignExpr(Assign& expr) override;

    mlir::ModuleOp getModule() { return module; }
    void createMainFunction();
    void finishMainFunction();
    void cleanUpDeadBlocks();
    void lowerToLLVM();
    void raiseToSCF();
    std::unique_ptr<llvm::Module> convertToLLVMIR();
};