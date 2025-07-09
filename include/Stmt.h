#pragma once

#include <memory>
#include <any>
#include "Token.h"
#include "Expr.h"

class Block;
class Expression;
class If;
class Print;
class Var;
class While;

class Visitor {
public:
    virtual ~Visitor() = default;
    virtual std::any visitBlockStmt(Block& stmt) = 0;
    virtual std::any visitExpressionStmt(Expression& stmt) = 0;
    virtual std::any visitIfStmt(If& stmt) = 0;
    virtual std::any visitPrintStmt(Print& stmt) = 0;
    virtual std::any visitVarStmt(Var& stmt) = 0;
    virtual std::any visitWhileStmt(While& stmt) = 0;
};

class Stmt {
public:
    virtual ~Stmt() = default;

    virtual std::any accept(Visitor& visitor) = 0;
};

using StmtPtr = std::unique_ptr<Stmt>;

class Block : public Stmt {
public:
    Block(std::vector<StmtPtr> statements)
        : statements(statements) {}

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
    If(StmtPtr condition, StmtPtr thenBranch, StmtPtr elseBranch)
        : condition(std::move(condition)), thenBranch(std::move(thenBranch)), 
          elseBranch(std::move(elseBranch)) {}

    std::any accept(Visitor& visitor) override {
        return visitor.visitIfStmt(*this);
    }

    const StmtPtr condition;
    const StmtPtr thenBranch;
    const StmtPtr elseBranch;
};

class Print : public Stmt {
public:
    Print(StmtPtr expression)
        : expression(std::move(expression)) {}

    std::any accept(Visitor& visitor) override {
        return visitor.visitPrintStmt(*this);
    }

    const StmtPtr expression;
};

class Var : public Stmt {
public:
    Var(Token name, StmtPtr initializer)
        : name(name), initializer(std::move(initializer)) {}

    std::any accept(Visitor& visitor) override {
        return visitor.visitVarStmt(*this);
    }

    const Token name;
    const StmtPtr initializer;
};

class While : public Stmt {
public:
    While(StmtPtr condition, StmtPtr body)
        : condition(std::move(condition)), body(std::move(body)) {}

    std::any accept(Visitor& visitor) override {
        return visitor.visitWhileStmt(*this);
    }

    const StmtPtr condition;
    const StmtPtr body;
};

