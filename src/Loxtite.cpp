#include "Loxtite.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include "Scanner.h"
#include "Token.h"
#include "Parser.h"
#include "AstPrinter.h"

bool Loxtite::hadError = false;

void Loxtite::run(const std::string& source) {
    Scanner scanner(source);
    std::vector<Token> tokens = scanner.scanTokens();
    Parser parser(tokens);
    std::vector<std::unique_ptr<Stmt>> statements = parser.parse();

    if (hadError) return;

    // std::cout << "SCANNER DUMP" << std::endl;
    // for (Token& token : tokens) {
    //     std::cout << token.toString() << std::endl;
    // }

    std::cout << "AST DUMP" << std::endl;
    AstPrinter printer;
    for (StmtPtr& stmt : statements) {
        std::cout << printer.print(*stmt) << std::endl;
    }
}

void Loxtite::runFile(const std::string& path) {
    std::ifstream file(path);

    if (!file) {
        throw std::runtime_error("Could not open file: " + path);
    }

    std::ostringstream buffer;
    buffer << file.rdbuf();
    std::string source = buffer.str();

    run(source);

    if (hadError) {
        exit(65);
    }
}

void Loxtite::runPrompt() {
    std::string line;
    std::cout << "Welcome to Loxtite! Type your code below (type 'exit' to quit):" << std::endl;

    while (true) {
        std::cout << "> ";

        if (!std::getline(std::cin, line)) {
            break;
        }

        run(line);
        hadError = false;
    }
}

void Loxtite::error(int line, const std::string& message) {
    report(line, "", message);
}

void Loxtite::report(int line, const std::string& where, 
                     const std::string& message) {
    std::cerr << "[line " << line << "] Error: " << message << std::endl;
}

