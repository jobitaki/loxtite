#include "Loxtite.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include "Scanner.h"
#include "Token.h"
#include "Parser.h"
#include "AstPrinter.h"
#include "AstLowering.h"

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

    // std::cout << "AST DUMP" << std::endl;
    // AstPrinter printer;
    // for (StmtPtr& stmt : statements) {
    //     std::cout << printer.print(*stmt) << std::endl;
    // }

    mlir::MLIRContext context;
    
    context.getOrLoadDialect<mlir::arith::ArithDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();

    AstLowering lowerer(&context);

    lowerer.createMainFunction();

    for (StmtPtr& stmt : statements) {
        stmt->accept(lowerer);
    }

    auto module = lowerer.getModule();

    module.print(llvm::outs());
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

void Loxtite::error(int line, std::string_view message) {
    report(line, "", message);
}

void Loxtite::report(int line, std::string_view where, 
                     std::string_view message) {
    std::cerr << "[line " << line << "] Error: " << message << std::endl;
}

