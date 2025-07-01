import sys

def defineType(f, baseName, className, fieldList):
    f.write("class " + className + " : public " + baseName + "{\n")
    f.write("public:\n")
    f.write("    " + className + "(" + fieldList + ")\n")

    fields = fieldList.split(", ")
    for i, field in enumerate(fieldList):
        if field == baseName:
            fieldList[i] = baseName + "Ptr"

    f.write("        : ")
    for i, field in enumerate(fields):
        type, name = field.split(" ")
        if type == "ExprPtr":
            f.write(name + "(std::move(" + name + "))")
        else:
            f.write(name + "(" + name + ")")
        if i < len(fields) - 1:
            f.write(", ")

    f.write(" {}\n\n")

    for field in fields:
        f.write("    const " + field + ";\n")
    
    f.write("};\n\n")



def defineAst(outputDir, baseName, types):
    path = outputDir + "/" + baseName + ".h"

    with open(path, "w") as f:
        f.write("#pragma once\n\n")
        f.write("#include <memory>\n")
        f.write('#include "Token.h"\n\n')
        f.write("class " + baseName + " {\n")
        f.write("public:\n")
        f.write("    virtual ~Expr() = default;\n")
        f.write("};\n\n")
        f.write("using " + baseName + "Ptr = std::unique_ptr<" + baseName + ">;\n\n")
        for type in types:
            className = type.split(";")[0].strip()
            fields = type.split(";")[1].strip()
            defineType(f, baseName, className, fields)

def main():
    args = sys.argv[1:]

    if len(args) != 1:
        print("Provide output directory")
        sys.exit(1)

    outputDir = args[0]

    defineAst(outputDir, "Expr", [
        "Binary   ; ExprPtr left, Token oper, ExprPtr right",
        "Grouping ; ExprPtr expression",
        "Literal  ; std::any value",
        "Unary    ; Token oper, ExprPtr right"
    ]);

if __name__ == "__main__":
    main()