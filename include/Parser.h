#pragma once

#include "Token.h"
#include "Expr.h"

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

    std::unique_ptr<Expr> primary();
    std::unique_ptr<Expr> unary();
    std::unique_ptr<Expr> factor();
    std::unique_ptr<Expr> term();
    std::unique_ptr<Expr> comparison();
    std::unique_ptr<Expr> equality();
    std::unique_ptr<Expr> expression();

public:
    Parser(std::vector<Token> tokens);
};
