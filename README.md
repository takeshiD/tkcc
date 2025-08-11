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
expr = 
```

# References
- [tcc](https://bellard.org/tcc/)
- [lcc](https://github.com/drh/lcc)
- [chibicc](https://github.com/rui314/chibicc)
- [compiler book](https://www.sigbus.info/compilerbook)
