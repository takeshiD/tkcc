mod parser;
use crate::parser::{Span, statements};

fn main() {
    let program = Span::new(
        "
        -1+2;
        2;
    ",
    );
    let ret = statements(program);
    println!("{:#?}", ret);
}
