# A Tiny C Compiler
## Implemetation Progress
- [ ] Arithmetic operations
- [ ] Function
- [ ] Variable
- [ ] Separate Compile and Linking
- [ ] Pointers
- [ ] Code Generation

## BNF

### Tokens
```
number := 0-9
alph := a-zA-Z
keyword := "char" | "else" | "if" | "int" | "return" | "while"
identifier := alph (alph | number)*
int := number (number)*
char := number (number)*
operator := "+" | "-" | "/" | "%" | "&" | "=" | "==" | "!=" | ">" | ">=" | "<" | "<="
```

### Grammer
```
expr = mul ("+" mul | "-" mul)*
mul = primary ("*" primary | "/" primary)*
primary = num | "(" expr ")"

expression = 
    | assignment-expression
    | expression ',' assignment-expression

assignment-expression =
    | conditional-expression
    | unary-expression assignment-operator assignment-expression

statement = 
    | labeled-statement
    | expression-statement
    | compound-statement
    | selection-statement
    | iteration-statement
    | jump-statement

expression-statement = (expr)?;
compound-statement = '{' (declaration)* (statement)* '}'
selection-statement = 
                    | 'if' '(' expression ')' statement
                    | 'if' '(' expression ')' statement 'else' statement
                    | 'switch' '(' expression ')' statement
iteration-statement =
                    | 'while' '(' expression ')' statement
statement = expression-statement | compound-statement
```

# References
- [tcc](https://bellard.org/tcc/)
- [lcc](https://github.com/drh/lcc)
- [chibicc](https://github.com/rui314/chibicc)
- [compiler book](https://www.sigbus.info/compilerbook)
