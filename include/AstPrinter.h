#pragma once

#include <string>
#include <sstream>
#include <Expr.h>

class AstPrinter : public Visitor<std::string> {
private:
    template<typename... Expr>
    std::string parenthesize(std::string_view name, const Expr&... expr) {
        std::ostringstream builder;

        builder << "(" << name;

        ((builder << " " << expr->accept(*this)), ...);

        builder << ")";

        return builder.str();
    }

public:
    std::string print(Expr<std::string>& expr);
    std::string visitBinaryExpr(Binary<std::string>& expr) override;
    std::string visitGroupingExpr(Grouping<std::string>& expr) override;
    std::string visitLiteralExpr(Literal<std::string>& expr) override;
    std::string visitUnaryExpr(Unary<std::string>& expr) override;
};