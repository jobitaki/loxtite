#pragma once

#include <string>

#include "Token.h"
#include "Expr.h"
#include "Stmt.h"

/// @brief Parser implements organizing expressions and statements into a tree.
class Parser {
private: 
    /// @brief The source represented as tokens (lexemes).
    std::vector<Token> tokens;

    /// @brief The current token being parsed in our tokens vector.
    int current = 0;
    
    /// @brief Returns whether we are at the end of our tokens vector.
    /// @return True if we are at EOF.
    bool isAtEnd();

    /// @brief Returns the token at the current location.
    /// @return Token at tokens[current].
    Token peek();

    /// @brief Returns the token at the previous location.
    /// @return Token at tokens[current-1].
    Token previous();

    /// @brief Returns the current token and advances the current counter.
    /// @return Token at tokens[current] before incrementing current.
    Token advance();

    /// @brief Checks for equality of type and tokens[current].
    /// @param type The type to be compared to.
    /// @return True if tokens[current] == type.
    bool check(TokenType type);
    
    /// @brief Checks if the current token matches a list of token types.
    ///        Advances if it does.
    /// @tparam ...Types The list of types to match.
    /// @param ...types The list of types to match.
    /// @return True if there was a match, advances. Does not advance if false.
    template<typename... Types>
    bool match(Types... types);

    /// @brief A presumptive advance. If tokens[current] != type, the advance
    ///        will fail.
    /// @param type The type to be expected.
    /// @param message The message to print when consume fails.
    /// @return tokens[current] if successful. 
    Token consume(TokenType type, std::string message);

    void synchronize();

    // Statements
    std::unique_ptr<Stmt> declaration();
    std::unique_ptr<Stmt> varStatement();
    std::unique_ptr<Stmt> statement();
    std::unique_ptr<Stmt> ifStatement();
    std::unique_ptr<Stmt> printStatement();
    std::unique_ptr<Stmt> whileStatement();
    std::unique_ptr<Stmt> exprStatement();
    std::vector<std::unique_ptr<Stmt>> block();

    // Expressions
    std::unique_ptr<Expr> expression();
    std::unique_ptr<Expr> assignment();
    std::unique_ptr<Expr> equality();
    std::unique_ptr<Expr> comparison();
    std::unique_ptr<Expr> term();
    std::unique_ptr<Expr> factor();
    std::unique_ptr<Expr> unary();
    std::unique_ptr<Expr> primary();

public:
    Parser(std::vector<Token> tokens);
    std::vector<std::unique_ptr<Stmt>> parse();
};
