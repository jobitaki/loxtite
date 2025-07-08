#include <iostream>
#include <memory>

#include "Loxtite.h"
#include "Expr.h"
#include "Token.h"
#include "AstPrinter.h"

// int main(int argc, char* argv[]) {
//     if (argc > 2) {
//         std::cout << "Usage: jlox [script]" << std::endl;
//         return 1;
//     } else if (argc == 2) {
//         Loxtite::runFile(argv[1]);
//     } else {
//         Loxtite::runPrompt();
//     }
//     return 0;
// }

int main(int argc, char* argv[]) {
    auto literal1 = std::make_unique<Literal>(123.00);
    auto literal2 = std::make_unique<Literal>(45.67);

    Token minus(TokenType::MINUS, "-", nullptr, 1);
    Token star(TokenType::STAR, "*", nullptr, 1);

    auto unary1 = std::make_unique<Unary>(minus, std::move(literal1));
    auto grouping1 = std::make_unique<Grouping>(std::move(literal2));

    auto expression = std::make_unique<Binary>(std::move(unary1), star, std::move(grouping1));

    AstPrinter printer;
    std::cout << printer.print(*expression) << std::endl;
}