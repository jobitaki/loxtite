#pragma once

#include <memory>
#include <any>
#include "Token.h"

template<typename R> class Binary;
template<typename R> class Grouping;
template<typename R> class Literal;
template<typename R> class Unary;
template<typename R> class Visitor;

template<typename R>
class Visitor {
public:
    virtual ~Visitor() = default;
    virtual R visitBinaryExpr(Binary<R>& expr) = 0;
    virtual R visitGroupingExpr(Grouping<R>& expr) = 0;
    virtual R visitLiteralExpr(Literal<R>& expr) = 0;
    virtual R visitUnaryExpr(Unary<R>& expr) = 0;
};

template<typename R>
class Expr {
public:
    virtual ~Expr() = default;

    virtual R accept(Visitor<R>& visitor) = 0;
};

template<typename R>
using ExprPtr = std::unique_ptr<Expr<R>>;

template<typename R>
class Binary : public Expr<R> {
public:
    Binary(ExprPtr<R> left, Token oper, ExprPtr<R> right)
        : left(std::move(left)), oper(oper), right(std::move(right)) {}

    R accept(Visitor<R>& visitor) override {
        return visitor.visitBinaryExpr(*this);
    }

    const ExprPtr<R> left;
    const Token oper;
    const ExprPtr<R> right;
};

template<typename R>
class Grouping : public Expr<R> {
public:
    Grouping(ExprPtr<R> expression)
        : expression(std::move(expression)) {}

    R accept(Visitor<R>& visitor) override {
        return visitor.visitGroupingExpr(*this);
    }

    const ExprPtr<R> expression;
};

template<typename R>
class Literal : public Expr<R> {
public:
    Literal(std::any value)
        : value(value) {}

    R accept(Visitor<R>& visitor) override {
        return visitor.visitLiteralExpr(*this);
    }

    // TODO codify into script
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

template<typename R>
class Unary : public Expr<R> {
public:
    Unary(Token oper, ExprPtr<R> right)
        : oper(oper), right(std::move(right)) {}

    R accept(Visitor<R>& visitor) override {
        return visitor.visitUnaryExpr(*this);
    }

    const Token oper;
    const ExprPtr<R> right;
};

