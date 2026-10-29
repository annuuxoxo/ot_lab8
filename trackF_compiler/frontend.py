"""Lexer and recursive-descent parser for the toy language.

Grammar:
  program := stmt*
  stmt    := ID '=' expr ';' | 'print' '(' expr ')' ';'
           | 'if' '(' expr ')' block ('else' block)? | 'while' '(' expr ')' block
  block   := '{' stmt* '}'
  expr    := arith (relop arith)?            relop in < <= > >= == !=
  arith   := term (('+' | '-') term)*
  term    := unary (('*' | '/' | '%') unary)*
  unary   := '-' unary | primary
  primary := INT | ID | '(' expr ')'

AST nodes are tuples:
  ('num', v) ('var', name) ('bin', op, left, right) ('neg', e)
  ('assign', name, e) ('print', e) ('if', cond, then_stmts, else_stmts) ('while', cond, body_stmts)
"""
import re

TOKEN = re.compile(r"\s*(?:(\d+)|([A-Za-z_]\w*)|(<=|>=|==|!=|[-+*/%<>=(){};]))")
KEYWORDS = {"if", "else", "while", "print"}
RELOPS = {"<": "lt", "<=": "le", ">": "gt", ">=": "ge", "==": "eq", "!=": "ne"}
ARITH = {"+": "add", "-": "sub", "*": "mul", "/": "div", "%": "mod"}


def tokenize(src):
    src = re.sub(r"#.*", "", src)  # comments
    pos, out = 0, []
    while pos < len(src):
        if src[pos:].strip() == "":
            break
        m = TOKEN.match(src, pos)
        if not m:
            raise SyntaxError(f"bad character at {pos}: {src[pos:pos + 10]!r}")
        num, name, sym = m.groups()
        if num:
            out.append(("num", int(num)))
        elif name:
            out.append(("kw", name) if name in KEYWORDS else ("id", name))
        else:
            out.append(("sym", sym))
        pos = m.end()
    out.append(("eof", None))
    return out


class Parser:
    def __init__(self, tokens):
        self.toks, self.i = tokens, 0

    def peek(self):
        return self.toks[self.i]

    def take(self, kind=None, val=None):
        t = self.toks[self.i]
        if (kind and t[0] != kind) or (val and t[1] != val):
            raise SyntaxError(f"expected {val or kind}, got {t}")
        self.i += 1
        return t

    def program(self):
        stmts = []
        while self.peek()[0] != "eof":
            stmts.append(self.stmt())
        return stmts

    def block(self):
        self.take("sym", "{")
        stmts = []
        while self.peek() != ("sym", "}"):
            stmts.append(self.stmt())
        self.take("sym", "}")
        return stmts

    def stmt(self):
        t = self.peek()
        if t == ("kw", "print"):
            self.take()
            self.take("sym", "(")
            e = self.expr()
            self.take("sym", ")")
            self.take("sym", ";")
            return ("print", e)
        if t == ("kw", "if"):
            self.take()
            self.take("sym", "(")
            c = self.expr()
            self.take("sym", ")")
            then = self.block()
            other = []
            if self.peek() == ("kw", "else"):
                self.take()
                other = self.block()
            return ("if", c, then, other)
        if t == ("kw", "while"):
            self.take()
            self.take("sym", "(")
            c = self.expr()
            self.take("sym", ")")
            return ("while", c, self.block())
        name = self.take("id")[1]
        self.take("sym", "=")
        e = self.expr()
        self.take("sym", ";")
        return ("assign", name, e)

    def expr(self):
        left = self.arith()
        t = self.peek()
        if t[0] == "sym" and t[1] in RELOPS:
            self.take()
            return ("bin", RELOPS[t[1]], left, self.arith())
        return left

    def arith(self):
        e = self.term()
        while self.peek()[0] == "sym" and self.peek()[1] in "+-":
            op = ARITH[self.take()[1]]
            e = ("bin", op, e, self.term())
        return e

    def term(self):
        e = self.unary()
        while self.peek()[0] == "sym" and self.peek()[1] in "*/%":
            op = ARITH[self.take()[1]]
            e = ("bin", op, e, self.unary())
        return e

    def unary(self):
        if self.peek() == ("sym", "-"):
            self.take()
            return ("neg", self.unary())
        return self.primary()

    def primary(self):
        t = self.take()
        if t[0] == "num":
            return ("num", t[1])
        if t[0] == "id":
            return ("var", t[1])
        if t == ("sym", "("):
            e = self.expr()
            self.take("sym", ")")
            return e
        raise SyntaxError(f"unexpected {t}")


def parse(src):
    return Parser(tokenize(src)).program()
