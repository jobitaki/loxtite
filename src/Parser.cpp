#include "Parser.h"

#include <iostream>

bool Parser::isAtEnd() {
    return peek().getType() == TokenType::MY_EOF;
}

Token Parser::peek() {
    return tokens.at(current);
}

Token Parser::previous() {
    return tokens.at(current - 1);
}

Token Parser::advance() {
    if (!isAtEnd()) current++;
    return previous();
}

bool Parser::check(TokenType type) {
    if (isAtEnd()) return false;
    return peek().getType() == type;
}

template<typename... Types>
bool Parser::match(Types... types) {
    for (TokenType type : {types...}) {
        if (check(type)) {
            advance();
            return true;
        }
    }

    return false;
}

Token Parser::consume(TokenType type, std::string message) {
    if (check(type)) return advance();

    // throw error(peek(), message);

    // TODO implement the stuff
    std::cerr << "Parse error, " << message << std::endl;
    exit(1);
}

void Parser::synchronize() {
    advance();
    
    while (!isAtEnd()) {
        if (previous().getType() == SEMICOLON) return;

        switch (peek().getType()) {
            case CLASS:
            case FUN:
            case VAR:
            case FOR:
            case IF:
            case WHILE:
            case PRINT:
            case RETURN:
                return;
        }

        advance();
    }
}

std::unique_ptr<Stmt> Parser::declaration() {
    try {
        if (match(TokenType::VAR)) return varStatement();
        return statement();
    } catch (...) {
        synchronize();
        return nullptr;
    }
}

std::unique_ptr<Stmt> Parser::varStatement() {
    Token name = consume(TokenType::IDENTIFIER, "Expect variable name.");

    std::unique_ptr<Expr> initializer = nullptr;

    if (match(TokenType::EQUAL)) {
        initializer = expression();
    }

    consume(TokenType::SEMICOLON, "Expect semicolon.");
    std::cout << "ADDED Var statement to tree" << std::endl;
    return std::make_unique<Var>(name, std::move(initializer));
}

std::unique_ptr<Stmt> Parser::statement() {
    if (match(TokenType::IF)) return ifStatement();
    if (match(TokenType::WHILE)) return whileStatement();
    if (match(TokenType::LEFT_BRACE)) return std::make_unique<Block>(block());

    return exprStatement();
}

std::unique_ptr<Stmt> Parser::ifStatement() {
    consume(TokenType::LEFT_PAREN, "Expect '('.");
    auto condition = expression();
    consume(TokenType::RIGHT_PAREN, "Expect ')'.");

    auto thenBranch = statement();
    std::unique_ptr<Stmt> elseBranch = nullptr;
    if (match(TokenType::ELSE)) {
        elseBranch = statement();
    }

    std::cout << "ADDED If statement to tree" << std::endl;
    return std::make_unique<If>(std::move(condition), std::move(thenBranch), 
                                std::move(elseBranch));
}

std::unique_ptr<Stmt> Parser::whileStatement() {
    consume(TokenType::LEFT_PAREN, "Expect '('.");
    auto condition = expression();
    consume(TokenType::RIGHT_PAREN, "Expect ')'.");

    auto body = statement();

    std::cout << "ADDED While statement to tree" << std::endl;
    return std::make_unique<While>(std::move(condition), std::move(body));
}

std::unique_ptr<Stmt> Parser::exprStatement() {
    auto expr = expression();
    consume(TokenType::SEMICOLON, "Expect ;.");
    std::cout << "ADDED Expr statement to tree" << std::endl;
    return std::make_unique<Expression>(std::move(expr));
}

std::vector<std::unique_ptr<Stmt>> Parser::block() {
    std::vector<std::unique_ptr<Stmt>> statements;

    while (!check(TokenType::RIGHT_BRACE) && !isAtEnd()) {
        statements.push_back(declaration());
    }

    consume(TokenType::RIGHT_BRACE, "Expect a '}'.");
    return statements;
}

std::unique_ptr<Expr> Parser::expression() {
    return assignment();
}

std::unique_ptr<Expr> Parser::assignment() {
    std::unique_ptr<Expr> expr = equality();

    if (match(TokenType::EQUAL)) {
        Token equals = previous(); // For error purposes
        std::unique_ptr<Expr> value = assignment();

        if (auto check = dynamic_cast<Variable*>(expr.get())) {
            Token name = check->name;
            return std::make_unique<Assign>(name, std::move(value));
        }

        // Error TODO
        std::cerr << "Invalid assignment target" << std::endl;
    }

    return expr;
}

std::unique_ptr<Expr> Parser::equality() {
    std::unique_ptr<Expr> expr = comparison();

    while (match(TokenType::BANG_EQUAL, TokenType::EQUAL_EQUAL)) {
        Token oper = previous();
        std::unique_ptr<Expr> right = comparison();
        expr = std::make_unique<Binary>(std::move(expr), oper, std::move(right));
    }

    return expr;
}

std::unique_ptr<Expr> Parser::comparison() {
    std::unique_ptr<Expr> expr = term();

    while (match(TokenType::GREATER, TokenType::GREATER_EQUAL,
                 TokenType::LESS, TokenType::LESS_EQUAL)) {
        Token oper = previous();
        std::unique_ptr<Expr> right = term();
        expr = std::make_unique<Binary>(std::move(expr), oper, std::move(right));
    }

    return expr;
}

std::unique_ptr<Expr> Parser::term() {
    std::unique_ptr<Expr> expr = factor();

    while (match(TokenType::MINUS, TokenType::PLUS)) {
        Token oper = previous();
        std::unique_ptr<Expr> right = factor();
        expr = std::make_unique<Binary>(std::move(expr), oper, std::move(right));
    }

    return expr;
}

std::unique_ptr<Expr> Parser::factor() {
    std::unique_ptr<Expr> expr = unary();
    
    while (match(TokenType::SLASH, TokenType::STAR)) {
        Token oper = previous();
        std::unique_ptr<Expr> right = unary();
        expr = std::make_unique<Binary>(std::move(expr), oper, std::move(right));
    }

    return expr;
}

std::unique_ptr<Expr> Parser::unary() {
    if (match(TokenType::BANG, TokenType::MINUS)) {
        Token oper = previous();
        std::unique_ptr<Expr> right = unary();
        return std::make_unique<Unary>(oper, std::move(right));
    }

    return primary();
}

std::unique_ptr<Expr> Parser::primary() {
    if (match(TokenType::FALSE)) return std::make_unique<Literal>(false);
    if (match(TokenType::TRUE)) return std::make_unique<Literal>(true);
    if (match(TokenType::NIL)) return std::make_unique<Literal>(nullptr);

    if (match(TokenType::NUMBER, TokenType::STRING)) {
        return std::make_unique<Literal>(previous().getLiteral());
    }

    if (match(TokenType::LEFT_PAREN)) {
        std::unique_ptr<Expr> expr = expression();
        consume(TokenType::RIGHT_PAREN, "Expect ')' after expression.");
        return std::make_unique<Grouping>(std::move(expr));
    }

    if (match(TokenType::IDENTIFIER)) {
        return std::make_unique<Variable>(previous());
    }

    // TODO throw error
    std::cerr << "Expect expression." << std::endl;
    exit(1);
}

Parser::Parser(std::vector<Token> tokens) : tokens(tokens) {}

std::vector<std::unique_ptr<Stmt>> Parser::parse() {
    std::vector<std::unique_ptr<Stmt>> statements;

    while (!isAtEnd()) {
        statements.push_back(declaration());
    }

    return statements;
}