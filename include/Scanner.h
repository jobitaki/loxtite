#pragma once

#include <string>
#include <vector>
#include <map>

#include "Token.h"

class Scanner {
private:
    const std::string source;
    std::vector<Token> tokens = {};
    int start = 0;
    int current = 0;
    int line = 1;

    inline static const std::map<std::string, TokenType> keywords = {
        {"and", AND},
        {"class", CLASS},
        {"else", ELSE},
        {"false", FALSE},
        {"for", FOR},
        {"fun", FUN},
        {"if", IF},
        {"nil", NIL},
        {"or", OR},
        {"print", PRINT},
        {"return", RETURN},
        {"super", SUPER},
        {"this", THIS},
        {"true", TRUE},
        {"var", VAR},
        {"while", WHILE}
    };

    bool isAtEnd() const;
    char advance();
    
    /// @brief A conditional advance. Advances only if expected matches.
    /// @param expected Expected character.
    /// @return True if matches the expected.
    bool match(char expected);

    /// @brief Returns current char without advancing.
    /// @return The current char.
    char peek();
    
    char peekNext();

    void string();

    bool isDigit(char c);

    void number();

    bool isAlpha(char c);

    bool isAlphaNumeric(char c);

    void identifier();

    void addToken(TokenType type);
    void addToken(TokenType type, std::any literal);
    void scanToken();

public:
    Scanner(const std::string& source);
    std::vector<Token> scanTokens();
};