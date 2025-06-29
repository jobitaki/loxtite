#include <string>
#include <any>

#include "Token.h"
#include "TokenType.h"

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
    
    return "<" + std::to_string(type) + " " + lexeme + " " + literalStr + ">";
}