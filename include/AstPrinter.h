#pragma once

#include <string>
#include <sstream>
#include "Expr.h"
#include "Stmt.h"

/// @brief Useful for printing out the AST of expressions.
class AstPrinter : public Visitor {
private:
    /// @brief Recursive function to print out the AST with a series of ().
    ///        For example, it may look like ( * (1 + 2) (456))
    /// @tparam ...Expr
    /// @param name The name of the operation.
    /// @param ...expr A series of pointers to expression types.
    /// @return The string of the AST.
    template<typename... ExprPtr>
    std::string parenthesizeExpr(std::string_view name, const ExprPtr&... expr) {
        std::ostringstream builder;

        builder << "(" << name;

        ((builder << " " << std::any_cast<std::string>(expr->accept(*this))), ...);

        builder << ")";

        return builder.str();
    }

    template<typename... StmtPtr>
    std::string parenthesizeStmt(std::string_view name, StmtPtr&... stmt) {
        std::ostringstream builder;

        builder << "(" << name;

        ((builder << " " << std::any_cast<std::string>(stmt.accept(*this))), ...);

        builder << ")";

        return builder.str();
    }

public:
    /// @brief Calls the accept function of the expression given to get the AST
    ///        in string format.
    /// @param expr The statement reference.
    /// @return The string of AST.
    std::string print(Stmt& stmt);

    std::any visitBlockStmt(Block& stmt) override;
    std::any visitExpressionStmt(Expression& stmt) override;
    std::any visitIfStmt(If& stmt) override;
    // std::any visitPrintStmt(Print& stmt) override;
    std::any visitVarStmt(Var& stmt) override;
    std::any visitWhileStmt(While& stmt) override;

    /// @brief Overridden visitor function for Binary expressions that simply
    ///        returns a string of its expression.
    /// @param expr The binary expression to represent.
    /// @return The string representation of the expression.
    std::any visitBinaryExpr(Binary& expr) override;

    /// @brief Overridden visitor function for Grouping expressions that simply
    ///        returns a string of its expression.
    /// @param expr The grouping expression to represent.
    /// @return The string representation of the expression.
    std::any visitGroupingExpr(Grouping& expr) override;

    /// @brief Overridden visitor function for Literal expressions that simply
    ///        returns a string of its expression.
    /// @param expr The literal expression to represent.
    /// @return The string representation of the expression.
    std::any visitLiteralExpr(Literal& expr) override;

    /// @brief Overridden visitor function for Unary expressions that simply
    ///        returns a string of its expression.
    /// @param expr The Unary expression to represent.
    /// @return The string representation of the expression.
    std::any visitUnaryExpr(Unary& expr) override;

    std::any visitVariableExpr(Variable& expr) override;
    std::any visitAssignExpr(Assign& expr) override;
};