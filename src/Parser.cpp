#include "Parser.h"

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
    for (TokenType type : types) {
        if (check(type)) {
            advance();
            return true;
        }
    }

    return false;
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
}

std::unique_ptr<Expr> Parser::unary() {
    if (match(TokenType::BANG, TokenType::MINUS)) {
        Token oper = previous();
        std::unique_ptr<Expr> right = unary();
        return std::make_unique<Unary>(oper, std::move(right));
    }

    return primary();
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

std::unique_ptr<Expr> Parser::term() {
    std::unique_ptr<Expr> expr = factor();

    while (match(TokenType::MINUS, TokenType::PLUS)) {
        Token oper = previous();
        std::unique_ptr<Expr> right = factor();
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

std::unique_ptr<Expr> Parser::equality() {
    std::unique_ptr<Expr> expr = comparison();

    while (match(TokenType::BANG_EQUAL, TokenType::EQUAL_EQUAL)) {
        Token oper = previous();
        std::unique_ptr<Expr> right = comparison();
        expr = std::make_unique<Binary>(std::move(expr), oper, std::move(right));
    }

    return expr;
}

std::unique_ptr<Expr> Parser::expression() {
    return equality();
}

Parser::Parser(std::vector<Token> tokens) : tokens(tokens) {}
