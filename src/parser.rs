use core::panic;

use nom::branch::alt;
use nom::bytes::complete::{is_not, tag};
use nom::character::complete::{alpha1, alphanumeric1, char, multispace0, multispace1};
use nom::combinator::{map, recognize};
use nom::error::ParseError;
use nom::multi::{fold_many0, many0};
use nom::number::complete::recognize_float;
use nom::sequence::{delimited, pair, preceded, terminated};
use nom::{IResult, Input, Offset, Parser};
use nom_language::error::VerboseError;
use nom_locate::LocatedSpan;

pub type Span<'a> = LocatedSpan<&'a str>;

#[derive(Debug, PartialEq, Clone)]
pub enum ExprEnum<'src> {
    Ident(String),
    NumLiteral(f64),
    StrLiteral(String),
    Add(Box<Expression<'src>>, Box<Expression<'src>>),
    Sub(Box<Expression<'src>>, Box<Expression<'src>>),
    Mul(Box<Expression<'src>>, Box<Expression<'src>>),
    Div(Box<Expression<'src>>, Box<Expression<'src>>),
    If(
        Box<Expression<'src>>,
        Box<Statements<'src>>,
        Option<Box<Statements<'src>>>,
    ),
    Ternary(
        Box<Expression<'src>>,
        Box<Expression<'src>>,
        Box<Expression<'src>>,
    ),
    Or(Box<Expression<'src>>, Box<Expression<'src>>),
    And(Box<Expression<'src>>, Box<Expression<'src>>),
    Increment,
    Decrement,
    IndexAccess(Box<Expression<'src>>, Box<Expression<'src>>),
    MemberAccess(Box<Expression<'src>>, Box<Expression<'src>>),
    Assign(Box<Expression<'src>>, Box<Expression<'src>>),
    FuncCall(Box<Expression<'src>>, Vec<Expression<'src>>),
}

#[derive(Debug, PartialEq, Clone)]
enum PostFix<'src> {
    IndexAccess(Box<Expression<'src>>),
    FuncCall(Vec<Expression<'src>>),
    MemberAccess(String),
    PointerAccess(String),
    Increment,
    Decrement,
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
type ExprResult<'src> = IResult<Span<'src>, Expression<'src>, VerboseError<Span<'src>>>;

fn calc_offset<'a>(i: Span<'a>, r: Span<'a>) -> Span<'a> {
    i.take(i.offset(&r))
}

/// Helper parser to remove invisible characters such as space, newline, tab.
///
/// Remove invisible characters before and after the parser target.
/// In almost cases, this is combinated `tag`.
pub fn spaces_delimited<
    'a,
    O,
    E: ParseError<Span<'a>>,
    F: Parser<Span<'a>, Output = O, Error = E>,
>(
    f: F,
) -> impl Parser<Span<'a>, Output = O, Error = E> {
    delimited(multispace0, f, multispace0)
}

fn parens_delimited<'a, O, E: ParseError<Span<'a>>, F: Parser<Span<'a>, Output = O, Error = E>>(
    f: F,
) -> impl Parser<Span<'a>, Output = O, Error = E> {
    delimited(char('('), f, char(')'))
}

fn number(input: Span) -> ExprResult {
    let (input, num) = spaces_delimited(recognize_float).parse(input)?;
    Ok((
        input,
        Expression::new(ExprEnum::NumLiteral(num.parse().unwrap()), input),
    ))
}

fn literal_char(input: Span) -> IResult<Span, Span, VerboseError<Span>> {
    is_not("\"\\").parse(input)
}

fn string(input: Span) -> ExprResult {
    let (input, build_string) = delimited(char('"'), literal_char, char('"')).parse(input)?;
    Ok((
        input,
        Expression::new(ExprEnum::StrLiteral(build_string.to_string()), input),
    ))
}

fn constant_expression(input: Span) -> ExprResult {
    number.parse(input)
}

fn primary_expression(input: Span) -> ExprResult {
    let (input, e) = alt((
        map(identifier, |s: Span| {
            Expression::new(ExprEnum::Ident(s.to_string()), s)
        }),
        constant_expression,
        string,
        parens_delimited(expression),
    ))
    .parse(input)?;
    Ok((input, e))
}

fn logical_or_expression(input: Span) -> ExprResult {
    let (input, e) = logical_and_expression(input)?;
    fold_many0(
        preceded(tag("||"), logical_and_expression),
        || e.clone(),
        |acc, e_logic| Expression::new(ExprEnum::Or(Box::new(acc), Box::new(e_logic)), input),
    )
    .parse(input)
}

fn logical_and_expression(input: Span) -> ExprResult {
    let (input, e) = primary_expression(input)?;
    fold_many0(
        preceded(tag("&&"), primary_expression),
        || e.clone(),
        |acc, e_logic| Expression::new(ExprEnum::And(Box::new(acc), Box::new(e_logic)), input),
    )
    .parse(input)
}

fn ternary_expression(input: Span) -> ExprResult {
    let (input, cond) = logical_or_expression(input)?;
    let (input, _) = spaces_delimited(char('?')).parse(input)?;
    let (input, true_branch) = expression(input)?;
    let (input, _) = spaces_delimited(char(':')).parse(input)?;
    let (input, false_branch) = conditional_expression(input)?;
    Ok((
        input,
        Expression::new(
            ExprEnum::Ternary(
                Box::new(cond),
                Box::new(true_branch),
                Box::new(false_branch),
            ),
            input,
        ),
    ))
}

fn conditional_expression(input: Span) -> ExprResult {
    alt((ternary_expression, logical_or_expression)).parse(input)
}

fn postfix_tail(input: Span) -> IResult<Span, PostFix, VerboseError<Span>> {
    alt((
        map(
            delimited(
                spaces_delimited(char('[')),
                expression,
                spaces_delimited(char(']')),
            ),
            |idx| PostFix::IndexAccess(Box::new(idx)),
        ),
        map(
            delimited(
                spaces_delimited(char('(')),
                many0(assignment_expression),
                spaces_delimited(char(')')),
            ),
            |params| PostFix::FuncCall(params),
        ),
        map(
            preceded(spaces_delimited(char('.')), identifier),
            |symbol| PostFix::MemberAccess(symbol.to_string()),
        ),
        map(
            preceded(spaces_delimited(tag("->")), identifier),
            |symbol| PostFix::PointerAccess(symbol.to_string()),
        ),
        map(spaces_delimited(tag("++")), |_| PostFix::Increment),
        map(spaces_delimited(tag("--")), |_| PostFix::Decrement),
    ))
    .parse(input)
}

fn postfix_expression(input: Span) -> ExprResult {
    let (input, e) = primary_expression.parse(input)?;
    fold_many0(
        postfix_tail,
        move || e.clone(),
        |acc, pf| match pf {
            PostFix::IndexAccess(idx) => {
                Expression::new(ExprEnum::IndexAccess(Box::new(acc), idx), input)
            }
            PostFix::FuncCall(params) => {
                Expression::new(ExprEnum::FuncCall(Box::new(acc), params), input)
            }
            PostFix::MemberAccess(s) => Expression::new(
                ExprEnum::MemberAccess(
                    Box::new(acc),
                    Box::new(Expression::new(ExprEnum::Ident(s), input)),
                ),
                input,
            ),
            _ => panic!(""),
        },
    )
    .parse(input)
}

fn unary_expression(input: Span) -> ExprResult {
    postfix_expression.parse(input)
}

fn assign(input: Span) -> ExprResult {
    let (input, left_value) = unary_expression(input)?;
    let (input, assign_op) = delimited(
        multispace1,
        recognize(alt((
            tag("="),
            tag("+="),
            tag("-="),
            tag("*="),
            tag("/="),
            // tag("%="),
            // tag("<<="),
            // tag(">>="),
            // tag("&="),
            // tag("^="),
            // tag("|="),
        ))),
        multispace0,
    )
    .parse(input)?;
    let (input, right_value) = assignment_expression(input)?;
    let e = match *assign_op.fragment() {
        "=" => ExprEnum::Assign(Box::new(left_value), Box::new(right_value)),
        "+=" => ExprEnum::Assign(
            Box::new(left_value.clone()),
            Box::new(Expression::new(
                ExprEnum::Add(Box::new(left_value), Box::new(right_value)),
                input,
            )),
        ),
        "-=" => ExprEnum::Assign(
            Box::new(left_value.clone()),
            Box::new(Expression::new(
                ExprEnum::Sub(Box::new(left_value), Box::new(right_value)),
                input,
            )),
        ),
        "*=" => ExprEnum::Assign(
            Box::new(left_value.clone()),
            Box::new(Expression::new(
                ExprEnum::Mul(Box::new(left_value), Box::new(right_value)),
                input,
            )),
        ),
        "/=" => ExprEnum::Assign(
            Box::new(left_value.clone()),
            Box::new(Expression::new(
                ExprEnum::Div(Box::new(left_value), Box::new(right_value)),
                input,
            )),
        ),
        _ => panic!("Invalid assign operator"),
    };
    Ok((input, Expression::new(e, input)))
}
/// x ? y : z
/// x[y] = a ? b : c
/// ++x[y] = a ? b : c
/// x[y]++ = a ? b : c
/// x[y] *= e -> x[y] = x[y] * e
fn assignment_expression(input: Span) -> ExprResult {
    alt((assign, conditional_expression)).parse(input)
}

fn expression(input: Span) -> ExprResult {
    alt((assignment_expression,)).parse(input)
}

fn identifier(input: Span) -> IResult<Span, Span, VerboseError<Span>> {
    recognize(pair(
        alt((alpha1, tag("_"))),
        many0(alt((alphanumeric1, tag("_")))),
    ))
    .parse(input)
}

pub fn expression_statement(input: Span) -> IResult<Span, Statement, VerboseError<Span>> {
    let semicolon = spaces_delimited(char(';'));
    let (input, e) = terminated(expression, semicolon).parse(input)?;
    Ok((input, Statement::Expression(e)))
}

/// # Todos
/// - add compound-statement
/// - add selection-statement
/// - add iteration-statement
/// - add jump-statement
pub fn statement(input: Span) -> IResult<Span, Statement, VerboseError<Span>> {
    let (input, stmt) = alt((expression_statement,)).parse(input)?;
    Ok((input, stmt))
}

pub fn statements(input: Span) -> IResult<Span, Statements, VerboseError<Span>> {
    let (input, stmts) = many0(statement).parse(input)?;
    Ok((input, stmts))
}

#[cfg(test)]
#[allow(clippy::approx_constant)]
mod tests {
    use super::*;
    use nom::error::ErrorKind;
    use nom_language::error::VerboseErrorKind;
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
            Err(nom::Err::Error(VerboseError {
                errors: vec![
                    (
                        unsafe { Span::new_from_raw_offset(0, 1, "-abc", (),) },
                        VerboseErrorKind::Nom(ErrorKind::Tag)
                    ),
                    (
                        unsafe { Span::new_from_raw_offset(0, 1, "-abc", (),) },
                        VerboseErrorKind::Nom(ErrorKind::Alt)
                    ),
                ]
            }))
        );
    }

    #[test]
    fn test_expression_ternary() {
        let program = Span::new("1 ? 2 : 3");
        assert_eq!(
            expression(program),
            Ok((
                unsafe { Span::new_from_raw_offset(9, 1, "", ()) },
                Expression::new(
                    ExprEnum::Ternary(
                        Box::new(Expression::new(ExprEnum::NumLiteral(1.0), unsafe {
                            Span::new_from_raw_offset(2, 1, "? 2 : 3", ())
                        })),
                        Box::new(Expression::new(ExprEnum::NumLiteral(2.0), unsafe {
                            Span::new_from_raw_offset(6, 1, ": 3", ())
                        })),
                        Box::new(Expression::new(ExprEnum::NumLiteral(3.0), unsafe {
                            Span::new_from_raw_offset(9, 1, "", ())
                        }))
                    ),
                    unsafe { Span::new_from_raw_offset(9, 1, "", (),) }
                )
            ))
        );
    }

    #[test]
    fn test_expression_logical_or() {
        let program = Span::new("1 || 3");
        assert_eq!(
            expression(program),
            Ok((
                unsafe { Span::new_from_raw_offset(6, 1, "", ()) },
                Expression::new(
                    ExprEnum::Or(
                        Box::new(Expression::new(ExprEnum::NumLiteral(1.0), unsafe {
                            Span::new_from_raw_offset(2, 1, "|| 3", ())
                        })),
                        Box::new(Expression::new(ExprEnum::NumLiteral(3.0), unsafe {
                            Span::new_from_raw_offset(6, 1, "", ())
                        })),
                    ),
                    unsafe { Span::new_from_raw_offset(2, 1, "|| 3", (),) }
                )
            ))
        );
    }

    #[test]
    fn test_expression_logical_and() {
        let program = Span::new("1 && 3");
        assert_eq!(
            expression(program),
            Ok((
                unsafe { Span::new_from_raw_offset(6, 1, "", ()) },
                Expression::new(
                    ExprEnum::And(
                        Box::new(Expression::new(ExprEnum::NumLiteral(1.0), unsafe {
                            Span::new_from_raw_offset(2, 1, "&& 3", ())
                        })),
                        Box::new(Expression::new(ExprEnum::NumLiteral(3.0), unsafe {
                            Span::new_from_raw_offset(6, 1, "", ())
                        })),
                    ),
                    unsafe { Span::new_from_raw_offset(2, 1, "&& 3", (),) }
                )
            ))
        );
    }

    #[test]
    fn test_expression_assign() {
        let program = Span::new("x = 1");
        assert_eq!(
            expression(program),
            Ok((
                unsafe { Span::new_from_raw_offset(5, 1, "", ()) },
                Expression::new(
                    ExprEnum::Assign(
                        Box::new(Expression::new(ExprEnum::Ident("x".to_string()), unsafe {
                            Span::new_from_raw_offset(0, 1, "x", ())
                        })),
                        Box::new(Expression::new(ExprEnum::NumLiteral(1.0), unsafe {
                            Span::new_from_raw_offset(5, 1, "", ())
                        })),
                    ),
                    unsafe { Span::new_from_raw_offset(5, 1, "", (),) }
                )
            ))
        );
        let program = Span::new("x += z");
        assert_eq!(
            expression(program),
            Ok((
                unsafe { Span::new_from_raw_offset(6, 1, "", ()) },
                Expression::new(
                    ExprEnum::Assign(
                        Box::new(Expression::new(ExprEnum::Ident("x".to_string()), unsafe {
                            Span::new_from_raw_offset(0, 1, "x", ())
                        })),
                        Box::new(Expression::new(
                            ExprEnum::Add(
                                Box::new(Expression::new(
                                    ExprEnum::Ident("x".to_string()),
                                    unsafe { Span::new_from_raw_offset(0, 1, "x", ()) }
                                )),
                                Box::new(Expression::new(
                                    ExprEnum::Ident("z".to_string()),
                                    unsafe { Span::new_from_raw_offset(5, 1, "z", ()) }
                                )),
                            ),
                            unsafe { Span::new_from_raw_offset(6, 1, "", ()) }
                        )),
                    ),
                    unsafe { Span::new_from_raw_offset(6, 1, "", (),) }
                )
            ))
        );
    }
}
