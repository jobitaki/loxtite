#pragma once

#include <memory>
#include "Token.h"

class Expr {
public:
    virtual ~Expr() = default;
};

using ExprPtr = std::unique_ptr<Expr>;

class BinaryExpr : public Expr {
public: 
    const ExprPtr left;
    const Token oper;
    const ExprPtr right;

    BinaryExpr(ExprPtr left, Token oper, ExprPtr right);
    virtual ~BinaryExpr() noexcept override = default;
};