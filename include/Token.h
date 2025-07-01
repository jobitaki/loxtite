#pragma once

#include <string>
#include <any>

#include "TokenType.h"

class Token {
private:
    const TokenType type;
    const std::string lexeme;
    const std::any literal;
    const int line;

public:
    Token(TokenType type, const std::string& lexeme, const std::any& literal, int line);
    
    /// @brief Creates string representation of token.
    /// @return Returns string representation of token.
    std::string toString();

    /// @brief Getter for lexeme
    /// @return lexeme
    std::string getLexeme() const;
};