#include <string>
#include <vector>

#include "Loxtite.h"
#include "Scanner.h"

Scanner::Scanner(const std::string& source) : source(source) {}

bool Scanner::isAtEnd() const {
    return current >= source.length();
}

char Scanner::advance() {
    return source.at(current++);
}

void Scanner::addToken(TokenType type) {
    addToken(type, nullptr);
}

void Scanner::addToken(TokenType type, std::any literal) {
    std::string text = source.substr(start, (current - start));
    tokens.push_back(Token(type, text, literal, line));
}

bool Scanner::match(char expected) {
    if (isAtEnd()) return false;
    if (source.at(current) != expected) return false;

    // Only if the next character is what we expect should we advance.
    current++;
    return true;
}

char Scanner::peek() {
    if (isAtEnd()) return '\0';
    return source.at(current);
}

char Scanner::peekNext() {
    if (current + 1 >= source.length()) return '\0';
    return source.at(current + 1);
}

void Scanner::string() {
    while (peek() != '"' && !isAtEnd()) {
        // If there is a newline
        if (peek() == '\n') line++;
        advance();
    }

    if (isAtEnd()) {
        // TODO test this out
        Loxtite::error(line, "Unterminated string.");
        return;
    }

    // For the closing "
    advance();

    // Trim the surrounding quotes
    std::string value = source.substr(start + 1, (current - start - 2));
    addToken(TokenType::STRING, value);
}

bool Scanner::isDigit(char c) {
    return c >= '0' && c <= '9';
}

void Scanner::number() {
    while (isDigit(peek())) advance();

    // Look for fractional part.
    if (peek() == '.' && isDigit(peekNext())) {
        // Consume the '.'
        advance();

        while (isDigit(peek())) advance();
    }

    addToken(TokenType::NUMBER, 
             std::stod(source.substr(start, (current - start))));
}

bool Scanner::isAlpha(char c) {
    return (c >= 'a' && c <= 'z') ||
           (c >= 'A' && c <= 'Z') ||
            c == '_';
}

bool Scanner::isAlphaNumeric(char c) {
    return isAlpha(c) || isDigit(c);
}

void Scanner::identifier() {
    while (isAlphaNumeric(peek())) advance();

    std::string text = source.substr(start, (current - start));
    auto keyword = keywords.find(text);
    TokenType type;
    if (keyword == keywords.end()) {
        // If not found in identifier map.
        type = TokenType::IDENTIFIER;
    } else {
        type = keyword->second;
    }

    addToken(type);
}

void Scanner::scanToken() {
    char c = advance();

    switch (c) {
        case '(': addToken(TokenType::LEFT_PAREN); break;
        case ')': addToken(TokenType::RIGHT_PAREN); break;
        case '{': addToken(TokenType::LEFT_BRACE); break;
        case '}': addToken(TokenType::RIGHT_BRACE); break;
        case ',': addToken(TokenType::COMMA); break;
        case '.': addToken(TokenType::DOT); break;
        case '-': addToken(TokenType::MINUS); break;
        case '+': addToken(TokenType::PLUS); break;
        case ';': addToken(TokenType::SEMICOLON); break;
        case '*': addToken(TokenType::STAR); break;
        case '!':
            addToken(match('=') ? TokenType::BANG_EQUAL : TokenType::BANG);
            break;
        case '=':
            addToken(match('=') ? TokenType::EQUAL_EQUAL: TokenType::EQUAL);
            break;
        case '<':
            addToken(match('=') ? TokenType::LESS_EQUAL : TokenType::LESS);
            break;
        case '>':
            addToken(match('=') ? TokenType::GREATER_EQUAL : TokenType::GREATER);
            break;
        case '/':
            if (match('/')) {
                // If we find a second /, keep reading until EOL.
                while (peek() != '\n' && !isAtEnd()) advance();
            } else if (match('*')) {
                // If we find a * keep reading until we see */
                while ((peek() != '*' && peekNext() != '/') && !isAtEnd()) advance(); 
                if (isAtEnd()) {
                    Loxtite::error(line, "Unterminated comment");
                } else {
                    // Account for '*/'
                    advance();
                    advance();
                }
            } else {
                addToken(TokenType::SLASH);
            }
            break;
        case ' ':
        case '\r':
        case '\t':
            // Ignore whitespace
            break;
        case '\n':
            line++;
            break;
        case '"': string(); break;
        default: 
            if (isDigit(c)) {
                number();
            } else if (isAlpha(c)) {
                identifier();
            } else {
                Loxtite::error(line, "Unexpected character");
            }
            break;
    }
}

std::vector<Token> Scanner::scanTokens() {
    while (!isAtEnd()) {
        start = current;
        scanToken();
    }

    // Add an EOF token to signify the end of the input
    tokens.push_back(Token(TokenType::MY_EOF, "", std::any(), line));
    return tokens;
}

