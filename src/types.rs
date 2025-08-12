use nom::{IResult, Parser};
use nom_locate::LocatedSpan;

pub type Span<'a> = LocatedSpan<&'a str>;

#[derive(Debug, PartialEq, Clone)]
pub enum ExprEnum<'src> {
    Ident(Span<'src>),
    NumLiteral(f64),
    StrLiteral(String),
    FnInvoke(Span<'src>, Vec<Expression<'src>>),
    Add(Box<Expression<'src>>, Box<Expression<'src>>),
    Sub(Box<Expression<'src>>, Box<Expression<'src>>),
    Mul(Box<Expression<'src>>, Box<Expression<'src>>),
    Div(Box<Expression<'src>>, Box<Expression<'src>>),
    Gt(Box<Expression<'src>>, Box<Expression<'src>>),
    Lt(Box<Expression<'src>>, Box<Expression<'src>>),
    If(
        Box<Expression<'src>>,
        Box<Statements<'src>>,
        Option<Box<Statements<'src>>>,
    ),
}

#[derive(Debug, PartialEq, Clone)]
pub struct Expression<'src> {
    expr: ExprEnum<'src>,
    span: Span<'src>,
}

impl<'src> Expression<'src> {
    pub fn new(expr: ExprEnum<'src>, span: Span<'src>) -> Expression<'src> {
        Expression { expr, span }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Statement<'src> {
    Expression(Expression<'src>),
    VarDef {
        span: Span<'src>,
        name: Span<'src>,
        ex: Expression<'src>,
    },
    VarAssign {
        span: Span<'src>,
        name: Span<'src>,
        ex: Expression<'src>,
    },
    For {
        span: Span<'src>,
        loop_var: Span<'src>,
        start: Expression<'src>,
        end: Expression<'src>,
        stmts: Statements<'src>,
    },
    Break,
    Continue,
    FnDef {
        name: Span<'src>,
        args: Vec<Span<'src>>,
        stmts: Statements<'src>,
        cofn: bool,
    },
    Return(Expression<'src>),
    Yield(Expression<'src>),
}

pub type Statements<'src> = Vec<Statement<'src>>;
