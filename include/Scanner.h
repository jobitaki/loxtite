#include <string>
#include <vector>
#include "Token.h"

class Scanner {
private:
    const std::string source;
    std::vector<Token> tokens = {};
    int start = 0;
    int current = 0;
    int line = 1;

    bool isAtEnd() const;
    char advance();
    
    /// @brief A conditional advance. Advances only if expected matches.
    /// @param expected Expected character.
    /// @return True if matches the expected.
    bool match(char expected);

    /// @brief Returns current char without advancing.
    /// @return The current char.
    char peek();

    void addToken(TokenType type);
    void addToken(TokenType type, std::any literal);
    void scanToken();

public:
    Scanner(const std::string& source);
    std::vector<Token> scanTokens();
};