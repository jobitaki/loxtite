#pragma once

#include <string>
#include <vector>
#include <map>

#include "Token.h"

class Scanner {
private:
    const std::string source;
    std::vector<Token> tokens = {};
    size_t start = 0;
    size_t current = 0;
    size_t line = 1;

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

    /// @brief Returns true if current has reached the end of file.
    /// @return True if current == end of file.
    bool isAtEnd() const;

    /// @brief Returns the character at current and increments current.
    /// @return The char at current in source.
    char advance();
    
    /// @brief A conditional advance. Advances only if expected matches.
    /// @param expected Expected character.
    /// @return True if matches the expected.
    bool match(char expected);

    /// @brief Returns current char without advancing.
    /// @return The current char.
    char peek();
    
    /// @brief Returns current+1 char without advancing.
    /// @return The current+1 char.
    char peekNext();

    /// @brief Reads a string value enclosed by "" and creates a token.
    void string();

    /// @brief Returns true if c is '0'-'9'.
    /// @param c The char to be checked.
    /// @return True if char is numeric.
    bool isDigit(char c);

    /// @brief Reads a number value and creates a token.
    void number();

    /// @brief Returns true if c is alphabetical including '_'.
    /// @param c The char to be checked.
    /// @return True if char is alphabetical.
    bool isAlpha(char c);

    /// @brief Returns true if c is either numeric or alphabetical.
    /// @param c The char to be checked.
    /// @return True if char is numeric or alphabetical.
    bool isAlphaNumeric(char c);

    /// @brief Reads an identifier and creates a new token.
    void identifier();

    /// @brief Adds token to tokens vector.
    /// @param type The TokenType of the token.
    void addToken(TokenType type);

    /// @brief Adds token to tokens vector including literal object.
    /// @param type The TokenType of the token.
    /// @param literal The literal object.
    void addToken(TokenType type, std::any literal);

    /// @brief Detects individual tokens and creates a token for them.
    void scanToken();

public:
    Scanner(const std::string& source);
    std::vector<Token> scanTokens();
};