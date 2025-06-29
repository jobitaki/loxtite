#include <iostream>

#include "Loxtite.h"

int main(int argc, char* argv[]) {
    if (argc > 2) {
        std::cout << "Usage: jlox [script]" << std::endl;
        return 1;
    } else if (argc == 2) {
        Loxtite::runFile(argv[1]);
    } else {
        Loxtite::runPrompt();
    }
    return 0;
}