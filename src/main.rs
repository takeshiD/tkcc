mod parser;
mod types;

use parser::{statements, expr_statement};
use types::Span;

fn main() {
    let program = Span::new("-1+2;");
    let ret = statements(program);
    println!("{:#?}", ret);
}
