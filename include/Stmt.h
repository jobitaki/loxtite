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

class Function : public Stmt {
public:
    Function(Token name, std::vector<Token> params, std::vector<StmtPtr> body) 
        : name(name), params(std::move(params)), body(std::move(body)) {}

    std::any accept(Visitor& visitor) override {
        return visitor.visitFunctionStmt(*this);
    }

    const Token name;
    const std::vector<Token> params;
    const std::vector<StmtPtr> body;
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

class Print : public Stmt {
public:
    Print(Token keyword, ExprPtr expression)
        : keyword(keyword), expression(std::move(expression)) {}

    std::any accept(Visitor& visitor) override {
        return visitor.visitPrintStmt(*this);
    }

    const Token keyword;
    const ExprPtr expression;
};

class Return : public Stmt {
public:
    Return(Token keyword, ExprPtr expression)
        : keyword(keyword), expression(std::move(expression)) {}

    std::any accept(Visitor& visitor) override {
        return visitor.visitReturnStmt(*this);
    }

    const Token keyword;
    const ExprPtr expression;
};

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


