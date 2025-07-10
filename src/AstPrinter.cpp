#include <AstPrinter.h>

#include <string>
#include <sstream>

#include "Parser.h"

std::string AstPrinter::print(Stmt& stmt) {
    return std::any_cast<std::string>(stmt.accept(*this));
}

std::any AstPrinter::visitBlockStmt(Block& stmt) {
}

std::any AstPrinter::visitExpressionStmt(Expression& stmt) {
    return parenthesizeStmt("expr", stmt);
}

std::any AstPrinter::visitIfStmt(If& stmt) {
    std::ostringstream builder;

    builder << "(IF (" << parenthesizeExpr("", stmt.condition) << "\n";
    return builder.str();
}

std::any AstPrinter::visitVarStmt(Var& stmt) {
    std::ostringstream builder;

    builder << "(VAR " << stmt.name.getLexeme() << "(" 
        << parenthesizeExpr("init", stmt.initializer) << ")\n";

    return builder.str();
}

std::any AstPrinter::visitWhileStmt(While& stmt) {
    std::ostringstream builder;

    builder << "(WHILE " << parenthesizeExpr("cond", stmt.condition) << ")\n";

    return builder.str();
}

std::any AstPrinter::visitBinaryExpr(Binary& expr) {
    return parenthesizeExpr(expr.oper.getLexeme(), expr.left, expr.right);
}

std::any AstPrinter::visitGroupingExpr(Grouping& expr) {
    return parenthesizeExpr("group", expr.expression);
}

std::any AstPrinter::visitLiteralExpr(Literal& expr) {
    if (!expr.value.has_value()) return "nil";
    return expr.valueToString();
}

std::any AstPrinter::visitUnaryExpr(Unary& expr) {
    return parenthesizeExpr(expr.oper.getLexeme(), expr.right);
}

std::any AstPrinter::visitVariableExpr(Variable& expr) {
    return expr.name.getLexeme();
}

std::any AstPrinter::visitAssignExpr(Assign& expr) {
    return parenthesizeExpr(expr.name.getLexeme(), expr.value);
}
