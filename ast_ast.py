import sys

def defineType(f, baseName, className, fieldList):
    f.write("template<typename R>\n")
    f.write("class " + className + " : public " + baseName + "<R> {\n")
    f.write("public:\n")
    f.write("    " + className + "(" + fieldList + ")\n")

    fields = fieldList.split(", ")

    f.write("        : ")
    for i, field in enumerate(fields):
        type, name = field.split(" ")
        if type == "ExprPtr<R>":
            f.write(name + "(std::move(" + name + "))")
        else:
            f.write(name + "(" + name + ")")
        if i < len(fields) - 1:
            f.write(", ")

    f.write(" {}\n\n")

    f.write("    R accept(Visitor<R>& visitor) override {\n")
    f.write("        return visitor.visit" + className + baseName + "(*this);\n")
    f.write("    }\n\n")

    for field in fields:
        f.write("    const " + field + ";\n")
    
    f.write("};\n\n")


def defineVisitors(f, baseName, types):
    f.write("template<typename R>\n")
    f.write("class Visitor {\n")
    f.write("public:\n")
    f.write("    virtual ~Visitor() = default;\n")
    for type in types:
        typeName = type.split(";")[0].strip()
        f.write("    virtual R visit" + typeName + baseName + "(" +
                    typeName + "<R>& " + baseName.lower() + ") = 0;\n")
    f.write("};\n\n")


def defineAst(outputDir, baseName, types):
    path = outputDir + "/" + baseName + ".h"

    with open(path, "w") as f:
        f.write("#pragma once\n\n")
        f.write("#include <memory>\n")
        f.write("#include <any>\n")
        f.write('#include "Token.h"\n\n')
        
        f.write("template<typename R> class Binary;\n")
        f.write("template<typename R> class Grouping;\n")
        f.write("template<typename R> class Literal;\n")
        f.write("template<typename R> class Unary;\n")
        f.write("template<typename R> class Visitor;\n\n")
        defineVisitors(f, baseName, types)

        f.write("template<typename R>\n")
        f.write("class " + baseName + " {\n")
        f.write("public:\n")
        f.write("    virtual ~Expr() = default;\n\n")
        f.write("    virtual R accept(Visitor<R>& visitor) = 0;\n")
        f.write("};\n\n")

        f.write("template<typename R>\n")
        f.write("using " + baseName + "Ptr = std::unique_ptr<" + baseName + "<R>>;\n\n")

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
        "Binary   ; ExprPtr<R> left, Token oper, ExprPtr<R> right",
        "Grouping ; ExprPtr<R> expression",
        "Literal  ; std::any value",
        "Unary    ; Token oper, ExprPtr<R> right"
    ]);

if __name__ == "__main__":
    main()