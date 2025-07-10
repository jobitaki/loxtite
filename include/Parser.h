#pragma once

#include <string>

#include "Token.h"
#include "Expr.h"
#include "Stmt.h"

// assignment     → equality          ← Your existing function
// equality       → comparison        ← Your existing function  
// comparison     → term              ← Your existing function
// term           → factor            ← Your existing function
// factor         → unary             ← Your existing function
// unary          → primary           ← Your existing function (with variable support)
// primary        → literals, variables, grouping

class Parser {
private: 
    std::vector<Token> tokens;
    int current = 0;
    
    bool isAtEnd();
    Token peek();
    Token previous();
    Token advance();
    bool check(TokenType type);
    
    template<typename... Types>
    bool match(Types... types);

    Token consume(TokenType type, std::string message);

    void synchronize();

    // Expressions
    std::unique_ptr<Expr> primary();
    std::unique_ptr<Expr> unary();
    std::unique_ptr<Expr> factor();
    std::unique_ptr<Expr> term();
    std::unique_ptr<Expr> comparison();
    std::unique_ptr<Expr> equality();
    std::unique_ptr<Expr> assignment();
    std::unique_ptr<Expr> expression();

    // Statements
    std::unique_ptr<Stmt> declaration();
    std::unique_ptr<Stmt> statement();
    std::unique_ptr<Stmt> exprStatement();
    std::unique_ptr<Stmt> ifStatement();
    std::unique_ptr<Stmt> printStatement();
    std::unique_ptr<Stmt> varStatement();
    std::unique_ptr<Stmt> whileStatement();
    std::vector<std::unique_ptr<Stmt>> block();


public:
    Parser(std::vector<Token> tokens);
    std::vector<std::unique_ptr<Stmt>> parse();
};
