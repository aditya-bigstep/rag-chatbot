"""Safe arithmetic evaluator exposed as an LLM tool (no eval/exec)."""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from typing import Any

from langchain_core.tools import tool

MAX_EXPRESSION_LENGTH = 200
MAX_EXPONENT = 1000

_BINARY: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _evaluate(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ValueError("exponent too large")
        return _BINARY[type(node.op)](left, right)  # type: ignore[no-any-return]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_evaluate(node.operand))  # type: ignore[no-any-return]
    raise ValueError("only plain arithmetic is supported")


def _format(value: int | float) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression such as "2 + 3 * 4" and return the result.

    Supports + - * / // % ** and parentheses on numbers. Use this for any calculation
    instead of doing the math yourself. Returns a string starting with "Error" if the
    input is not valid arithmetic.
    """
    if len(expression) > MAX_EXPRESSION_LENGTH:
        return "Error: expression too long"
    try:
        return _format(_evaluate(ast.parse(expression.strip(), mode="eval")))
    except ZeroDivisionError:
        return "Error: division by zero"
    except (SyntaxError, ValueError, OverflowError, RecursionError) as exc:
        return f"Error: {exc}"
