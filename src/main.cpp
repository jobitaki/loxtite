#include "Loxtite.h"

int main(int argc, char* argv[]) {
    if (argc > 1) {
        std::cout << "Usage: jlox [script]" << std::endl;
        return 1;
    } else if (argc == 1) {
        Loxtite::runFile(argv[0]);
    } else {
        Loxtite::runPrompt();
    }
    return 0;
}