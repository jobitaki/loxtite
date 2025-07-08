#include <AstPrinter.h>

#include <string>
#include <sstream>

std::string AstPrinter::print(Expr& expr) {
    return std::any_cast<std::string>(expr.accept(*this));
}

std::any AstPrinter::visitBinaryExpr(Binary& expr) {
    return parenthesize(expr.oper.getLexeme(), expr.left, expr.right);
}

std::any AstPrinter::visitGroupingExpr(Grouping& expr) {
    return parenthesize("group", expr.expression);
}

std::any AstPrinter::visitLiteralExpr(Literal& expr) {
    if (!expr.value.has_value()) return "nil";
    return expr.valueToString();
}

std::any AstPrinter::visitUnaryExpr(Unary& expr) {
    return parenthesize(expr.oper.getLexeme(), expr.right);
}