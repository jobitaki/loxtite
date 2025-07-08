#pragma once

#include <string>
#include <sstream>
#include <Expr.h>

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
    std::string parenthesize(std::string_view name, const ExprPtr&... expr) {
        std::ostringstream builder;

        builder << "(" << name;

        ((builder << " " << std::any_cast<std::string>(expr->accept(*this))), ...);

        builder << ")";

        return builder.str();
    }

public:
    /// @brief Calls the accept function of the expression given to get the AST
    ///        in string format.
    /// @param expr The expression reference.
    /// @return The string of AST.
    std::string print(Expr& expr);

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
};