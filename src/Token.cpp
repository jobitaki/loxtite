#include <string>
#include <any>

#include "Token.h"

constexpr std::string_view TokenTypeString[] = {
    // Single character tokens
    "LEFT_PAREN", "RIGHT_PAREN", "LEFT_BRACE", "RIGHT_BRACE",
    "COMMA", "DOT", "MINUS", "PLUS", "SEMICOLON", "SLASH", "STAR",

    // One or two character tokens
    "BANG", "BANG_EQUAL", "EQUAL", "EQUAL_EQUAL",
    "GREATER", "GREATER_EQUAL", "LESS", "LESS_EQUAL",

    // Literals
    "IDENTIFIER", "STRING", "NUMBER",

    // Keywords
    "AND", "CLASS", "ELSE", "FALSE", "FUN", "FOR", "IF", "NIL", "OR",
    "PRINT", "RETURN", "SUPER", "THIS", "TRUE", "VAR", "WHILE", "MY_EOF"
};

Token::Token(TokenType type, const std::string& lexeme, const std::any& literal, int line)
    : type(type), lexeme(lexeme), literal(literal), line(line) {}

std::string Token::toString() {
    std::string literalStr;
    if (literal.type() == typeid(std::string)) {
        literalStr = std::any_cast<std::string>(literal);
    } else if (literal.type() == typeid(double)) {
        literalStr = std::to_string(std::any_cast<double>(literal));
    } else if (literal.type() == typeid(bool)) {
        literalStr = std::any_cast<bool>(literal) ? "true" : "false";
    } else {
        literalStr = "null";
    }
    
    return "<" + std::string(TokenTypeString[type]) + " " + lexeme + " " + literalStr + ">";
}

std::string Token::getLexeme() const {
    return lexeme;
}

TokenType Token::getType() const {
    return type;
}

std::any Token::getLiteral() const {
    return literal;
}