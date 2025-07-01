#include <AstPrinter.h>

#include <string>
#include <sstream>

std::string AstPrinter::print(Expr<std::string>& expr) {
    return expr.accept(*this);
}

std::string AstPrinter::visitBinaryExpr(Binary<std::string>& expr) {
    return parenthesize(expr.oper.getLexeme(), expr.left, expr.right);
}

std::string AstPrinter::visitGroupingExpr(Grouping<std::string>& expr) {
    return parenthesize("group", expr.expression);
}

std::string AstPrinter::visitLiteralExpr(Literal<std::string>& expr) {
    if (!expr.value.has_value()) return "nil";
    return expr.valueToString();
}

std::string AstPrinter::visitUnaryExpr(Unary<std::string>& expr) {
    return parenthesize(expr.oper.getLexeme(), expr.right);
}