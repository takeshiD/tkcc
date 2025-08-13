use core::panic;

use nom::branch::alt;
use nom::bytes::complete::tag;
use nom::character::complete::{alpha1, alphanumeric1, char, multispace0};
use nom::combinator::recognize;
use nom::error::{Error, ParseError};
use nom::multi::{fold_many0, many0};
use nom::number::complete::recognize_float;
use nom::sequence::{delimited, pair, terminated};
use nom::{IResult, Input, Offset, Parser};
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

/// Expression struct for parsing
#[derive(Debug, PartialEq, Clone)]
pub struct Expression<'src> {
    /// ExpressionKind see ExprEnum
    expr: ExprEnum<'src>,
    /// Original Sourcecode Span, such line, offset
    span: Span<'src>,
}

impl<'src> Expression<'src> {
    /// Return Expression
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

/// Helper parser to remove invisible characters such as space, newline, tab.
///
/// Remove invisible characters before and after the parser target.
/// In almost cases, this is combinated `tag`.
pub fn space_delimited<
    'a,
    O,
    E: ParseError<Span<'a>>,
    F: Parser<Span<'a>, Output = O, Error = E>,
>(
    f: F,
) -> impl Parser<Span<'a>, Output = O, Error = E> {
    delimited(multispace0, f, multispace0)
}

fn calc_offset<'a>(i: Span<'a>, r: Span<'a>) -> Span<'a> {
    i.take(i.offset(&r))
}

fn paren_delimited<'a, O, E: ParseError<Span<'a>>, F: Parser<Span<'a>, Output = O, Error = E>>(
    f: F,
) -> impl Parser<Span<'a>, Output = O, Error = E> {
    delimited(char('('), f, char(')'))
}

fn bracket_delimited<'a, O, E: ParseError<Span<'a>>, F: Parser<Span<'a>, Output = O, Error = E>>(
    f: F,
) -> impl Parser<Span<'a>, Output = O, Error = E> {
    delimited(char('{'), f, char('}'))
}

fn number(input: Span) -> IResult<Span, Expression> {
    let (input, num) = space_delimited(recognize_float).parse(input)?;
    Ok((
        input,
        Expression::new(
            ExprEnum::NumLiteral(num.parse().map_err(|_| {
                nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::Digit))
            })?),
            input,
        ),
    ))
}

fn primary(input: Span) -> IResult<Span, Expression> {
    alt((number, bracket_delimited(expr))).parse(input)
}

fn mul(input: Span) -> IResult<Span, Expression> {
    let (input, init) = primary(input)?;
    fold_many0(
        pair(space_delimited(alt((char('*'), char('/')))), primary),
        move || init.clone(),
        |acc, (op, val): (char, Expression)| match op {
            '*' => Expression::new(ExprEnum::Mul(Box::new(acc), Box::new(val)), input),
            '/' => Expression::new(ExprEnum::Div(Box::new(acc), Box::new(val)), input),
            _ => panic!("Binary operator must be '*' or '/'. Actualy got '{op}'"),
        },
    )
    .parse(input)
}

pub fn expr(input: Span) -> IResult<Span, Expression> {
    let (r, init) = mul(input)?;
    fold_many0(
        pair(space_delimited(alt((char('+'), char('-')))), mul),
        move || init.clone(),
        |acc, (op, val): (char, Expression)| {
            let span = calc_offset(input, acc.span);
            println!("{:#?}", acc);
            match op {
                '+' => Expression::new(ExprEnum::Add(Box::new(acc), Box::new(val)), span),
                '-' => Expression::new(ExprEnum::Sub(Box::new(acc), Box::new(val)), span),
                _ => panic!("Binary operator must be '+' or '-'. Actualy got '{op}'"),
            }
        },
    )
    .parse(r)
}

fn identifier(input: Span) -> IResult<Span, Span> {
    recognize(pair(
        alt((alpha1, tag("_"))),
        many0(alt((alphanumeric1, tag("_")))),
    ))
    .parse(input)
}

fn var_assign(input: Span) -> IResult<Span, Statement> {
    let span = input;
    let (input, name) = space_delimited(identifier).parse(input)?;
    let (input, _) = space_delimited(char('=')).parse(input)?;
    let (input, ex) = space_delimited(expr).parse(input)?;
    let (input, _) = space_delimited(char(';')).parse(input)?;
    Ok((
        input,
        Statement::VarAssign {
            span: calc_offset(span, input),
            name,
            ex,
        },
    ))
}

pub fn expr_statement(input: Span) -> IResult<Span, Statement> {
    let (input, e) = expr(input)?;
    Ok((input, Statement::Expression(e)))
}

pub fn statement(input: Span) -> IResult<Span, Statement> {
    let semicolon = space_delimited(char(';'));
    let (input, stmt) = alt((var_assign, terminated(expr_statement, semicolon))).parse(input)?;
    Ok((input, stmt))
}

pub fn statements(input: Span) -> IResult<Span, Statements> {
    let (input, stmts) = many0(statement).parse(input)?;
    Ok((input, stmts))
}

#[cfg(test)]
#[allow(clippy::approx_constant)]
mod tests {
    use super::*;
    use nom::error::ErrorKind;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_number() {
        let program = Span::new("1.0");
        assert_eq!(
            number(program).unwrap().1,
            Expression::new(ExprEnum::NumLiteral(1.0), unsafe {
                Span::new_from_raw_offset(3, 1, "", ())
            })
        );
        let program = Span::new("-1");
        assert_eq!(
            number(program).unwrap().1,
            Expression::new(ExprEnum::NumLiteral(-1.0), unsafe {
                Span::new_from_raw_offset(2, 1, "", ())
            })
        );
        let program = Span::new("+1");
        assert_eq!(
            number(program).unwrap().1,
            Expression::new(ExprEnum::NumLiteral(1.0), unsafe {
                Span::new_from_raw_offset(2, 1, "", ())
            })
        );
    }

    #[test]
    fn test_identifier() {
        // Normal Tests
        let program = Span::new("xyz");
        assert_eq!(
            identifier(program),
            Ok(
                (unsafe { Span::new_from_raw_offset(3, 1, "", ()) }, unsafe {
                    Span::new_from_raw_offset(0, 1, "xyz", ())
                },)
            )
        );
        let program = Span::new("___");
        assert_eq!(
            identifier(program),
            Ok(
                (unsafe { Span::new_from_raw_offset(3, 1, "", ()) }, unsafe {
                    Span::new_from_raw_offset(0, 1, "___", ())
                },)
            )
        );

        // Abnormal Tests
        let program = Span::new("-abc");
        assert_eq!(
            identifier(program),
            Err(nom::Err::Error(Error::new(
                unsafe { Span::new_from_raw_offset(0, 1, "-abc", ()) },
                ErrorKind::Tag,
            )))
        );
    }

    #[test]
    fn test_expr() {
        let program = Span::new("1 + 2 + 3");
        assert_eq!(
            expr(program),
            Ok((
                unsafe { Span::new_from_raw_offset(5, 1, "", (),) },
                Expression::new(
                    ExprEnum::Add(
                        Box::new(Expression::new(ExprEnum::NumLiteral(1.0), unsafe {
                            Span::new_from_raw_offset(2, 1, "+ 2", ())
                        })),
                        Box::new(Expression::new(ExprEnum::NumLiteral(2.0), unsafe {
                            Span::new_from_raw_offset(5, 1, "", ())
                        }))
                    ),
                    unsafe { Span::new_from_raw_offset(0, 1, "1 + 2", ()) }
                )
            ))
        );
    }
}
