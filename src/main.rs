mod parser;
use nom::Finish;

use crate::parser::{Span, statements};

fn main() {
    let program = Span::new(
        "
        -1+( 2 + 3);
    ",
    );
    match statements(program).finish() {
        Ok((res, stmts)) => {
            println!("Parse Success");
            for stmt in stmts.iter() {
                println!("{:#?}", stmt);
            }
        }
        Err(e) => {
            println!("{}", e);
        }
    }
}
