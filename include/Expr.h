/// The purpose of Visitors is to be able to scale the number of expression-
/// specific processing functions there are. It simply would not be scalable
/// to keep adding new member functions to the derived classes. Instead, we
/// have a visitor class that can be used as a base for other classes to 
/// implement functions.

#pragma once

#include <memory>
#include <any>
#include <string>
#include "Token.h"

// Forward declarations
// Expressions
class Binary;
class Grouping;
class Literal;
class Unary;
class Assign;
class Variable;
class Call;

// Statements
class Block;
class Expression;
class Function;
class If;
class While;
class Print;
class Return;
class Var;

class Visitor {
public:
    virtual ~Visitor() = default;

    virtual std::any visitBlockStmt(Block& stmt) = 0;
    virtual std::any visitExpressionStmt(Expression& stmt) = 0;
    virtual std::any visitFunctionStmt(Function& stmt) = 0;
    virtual std::any visitIfStmt(If& stmt) = 0;
    virtual std::any visitWhileStmt(While& stmt) = 0;
    virtual std::any visitPrintStmt(Print& stmt) = 0;
    virtual std::any visitReturnStmt(Return& stmt) = 0;
    virtual std::any visitVarStmt(Var& stmt) = 0;
    
    virtual std::any visitBinaryExpr(Binary& expr) = 0;
    virtual std::any visitGroupingExpr(Grouping& expr) = 0;
    virtual std::any visitLiteralExpr(Literal& expr) = 0;
    virtual std::any visitUnaryExpr(Unary& expr) = 0;
    virtual std::any visitCallExpr(Call& expr) = 0;
    virtual std::any visitVariableExpr(Variable& expr) = 0;
    virtual std::any visitAssignExpr(Assign& expr) = 0;
};

class Expr {
public:
    virtual ~Expr() = default;
    virtual std::any accept(Visitor& visitor) = 0;
};

using ExprPtr = std::unique_ptr<Expr>;

class Binary : public Expr {
public:
    Binary(ExprPtr left, Token oper, ExprPtr right)
        : left(std::move(left)), oper(oper), right(std::move(right)) {}

    std::any accept(Visitor& visitor) override {
        return visitor.visitBinaryExpr(*this);
    }

    const ExprPtr left;
    const Token oper;
    const ExprPtr right;
};

class Grouping : public Expr {
public:
    Grouping(ExprPtr expression)
        : expression(std::move(expression)) {}

    std::any accept(Visitor& visitor) override {
        return visitor.visitGroupingExpr(*this);
    }

    const ExprPtr expression;
};

class Literal : public Expr {
public:
    Literal(std::any value)
        : value(value) {}

    std::any accept(Visitor& visitor) override {
        return visitor.visitLiteralExpr(*this);
    }

    std::string valueToString() {
        std::string literalStr;

        if (value.type() == typeid(std::string)) {
            literalStr = std::any_cast<std::string>(value);
        } else if (value.type() == typeid(double)) {
            literalStr = std::to_string(std::any_cast<double>(value));
        } else if (value.type() == typeid(bool)) {
            literalStr = std::any_cast<bool>(value) ? "true" : "false";
        } else {
            literalStr = "null";
        }
    
        return literalStr;
    }

    const std::any value;
};

class Unary : public Expr {
public:
    Unary(Token oper, ExprPtr right)
        : oper(oper), right(std::move(right)) {}

    std::any accept(Visitor& visitor) override {
        return visitor.visitUnaryExpr(*this);
    }

    const Token oper;
    const ExprPtr right;
};

class Variable : public Expr {
public:
    Variable(Token name)
        : name(name) {}

    std::any accept(Visitor& visitor) override {
        return visitor.visitVariableExpr(*this);
    }

    const Token name;
};

class Assign : public Expr {
public:
    Assign(Token name, ExprPtr value)
        : name(name), value(std::move(value)) {}

    std::any accept(Visitor& visitor) override {
        return visitor.visitAssignExpr(*this);
    }

    const Token name;
    const ExprPtr value;
};

class Call : public Expr {
public:
    Call(ExprPtr callee, Token paren, std::vector<ExprPtr> arguments)
        : callee(std::move(callee)), paren(paren), arguments(std::move(arguments)) {}
    
    std::any accept(Visitor& visitor) override {
        return visitor.visitCallExpr(*this);
    }

    const ExprPtr callee;
    const Token paren;
    const std::vector<ExprPtr> arguments;
};