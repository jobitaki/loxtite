#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include "Scanner.h"

class Loxtite {
private: 
    static bool hadError;

    /// @brief Prints where and what the error is and sets hadError.
    /// @param line The line of error.
    /// @param where A description of where the error is.
    /// @param message A description of the error.
    static void report(int line, const std::string& where, 
                       const std::string& message);

    /// @brief Runs a Loxtite script from a string. It tokenizes the source and
    ///        prints the tokens to the console for now.
    static void run(const std::string& source);

public:
    /// @brief Runs a Loxtite script from a file.
    /// @param path The path to the file to run.
    static void runFile(const std::string& path);

    /// @brief Runs a Loxtite script from the command line prompt.
    ///        It allows the user to enter code line by line.
    static void runPrompt();

    /// @brief Prints error message.
    /// @param line The line at which error occurred.
    /// @param message The message to be printed.
    static void error(int line, const std::string& message);
};