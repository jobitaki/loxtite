#pragma once

#include <memory>
#include <any>
#include "Token.h"
#include "Expr.h"

class Stmt {
public:
    virtual ~Stmt() = default;

    virtual std::any accept(Visitor& visitor) = 0;
};

using StmtPtr = std::unique_ptr<Stmt>;

class Block : public Stmt {
public:
    Block(std::vector<StmtPtr> statements)
        : statements(std::move(statements)) {}

    std::any accept(Visitor& visitor) override {
        return visitor.visitBlockStmt(*this);
    }

    const std::vector<StmtPtr> statements;
};

class Expression : public Stmt {
public:
    Expression(ExprPtr expression)
        : expression(std::move(expression)) {}

    std::any accept(Visitor& visitor) override {
        return visitor.visitExpressionStmt(*this);
    }

    const ExprPtr expression;
};

class If : public Stmt {
public:
    If(ExprPtr condition, StmtPtr thenBranch, StmtPtr elseBranch)
        : condition(std::move(condition)), thenBranch(std::move(thenBranch)), 
          elseBranch(std::move(elseBranch)) {}

    std::any accept(Visitor& visitor) override {
        return visitor.visitIfStmt(*this);
    }

    const ExprPtr condition;
    const StmtPtr thenBranch;
    const StmtPtr elseBranch;
};

// class Print : public Stmt {
// public:
//     Print(ExprPtr expression)
//         : expression(std::move(expression)) {}

//     std::any accept(Visitor& visitor) override {
//         return visitor.visitPrintStmt(*this);
//     }

//     const ExprPtr expression;
// };

class Var : public Stmt {
public:
    Var(Token name, ExprPtr initializer)
        : name(name), initializer(std::move(initializer)) {}

    std::any accept(Visitor& visitor) override {
        return visitor.visitVarStmt(*this);
    }

    const Token name;
    const ExprPtr initializer;
};

class While : public Stmt {
public:
    While(ExprPtr condition, StmtPtr body)
        : condition(std::move(condition)), body(std::move(body)) {}

    std::any accept(Visitor& visitor) override {
        return visitor.visitWhileStmt(*this);
    }

    const ExprPtr condition;
    const StmtPtr body;
};

