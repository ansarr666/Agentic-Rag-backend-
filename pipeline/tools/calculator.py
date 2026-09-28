"""Safe Deterministic Calculator Tool using AST parsing (no eval/exec)."""

import ast
import re
import operator
from dataclasses import dataclass
from typing import Optional, Union, Tuple, Dict, Any

@dataclass
class CalculationResult:
    """Structured result of a calculation."""
    expression: str
    result: Optional[Union[float, int]]
    formatted_result: str
    success: bool
    currency_symbol: Optional[str] = None
    error_message: Optional[str] = None
    steps: Optional[str] = None

class CalculatorTool:
    """
    Deterministic mathematical operations evaluator.
    Supports arithmetic, percentages, financial discounts, formula evaluation,
    and unit conversions using safe AST parsing.
    """

    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    CURRENCY_SYMBOLS = ["₹", "$", "€", "£", "¥", "Rs.", "INR", "USD", "EUR"]

    @classmethod
    def can_handle(cls, query: str) -> bool:
        """Determine if a query is a mathematical operation suitable for the calculator."""
        q = query.strip()

        # Direct math expressions (e.g., "17 * 45", "184500 * 0.83", "23% of 850")
        has_percentage_pattern = bool(re.search(r"\b\d+(\.\d+)?\s*%\s*(of\s*)?[\$₹€£]?\s*[\d,]+(\.\d+)?", q, re.I))
        has_arithmetic_phrase = bool(re.search(
            r"\b(calculate|compute|what is|how much is)\b.*\b(\+|\-|\*|\/|plus|minus|multiplied by|divided by|percent of|% of)\b",
            q,
            re.I
        ))
        has_discount_phrase = bool(re.search(
            r"(reduce|reduced by|discount of|increase|increased by)\s*[\$₹€£]?\s*\d+(\.\d+)?%",
            q,
            re.I
        ))
        has_pure_math = bool(re.match(r"^[\s\(\)\d\+\-\*\/\%\^\.\,\$₹€£a-zA-Z\s]+$", q) and re.search(r"[\+\-\*\/\^%]", q))

        return has_percentage_pattern or has_arithmetic_phrase or has_discount_phrase or (has_pure_math and any(c.isdigit() for c in q))

    @classmethod
    def extract_and_evaluate(cls, query: str) -> CalculationResult:
        """Extract mathematical expression from natural language and evaluate it safely."""
        raw_currency = None
        for sym in cls.CURRENCY_SYMBOLS:
            if sym.lower() in query.lower():
                raw_currency = sym
                break

        # Pattern 1: Percentage of (e.g., "What is 17% of ₹184,500?" or "23% of 850")
        match_pct_of = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:of)\s*[\$₹€£Rs\.\s]*([\d,]+(?:\.\d+)?)", query, re.I)
        if match_pct_of:
            pct = float(match_pct_of.group(1))
            val = float(match_pct_of.group(2).replace(",", ""))
            res = (pct / 100.0) * val
            expr = f"{pct}% of {val:g}"
            formatted = cls._format_number(res, raw_currency)
            return CalculationResult(
                expression=expr,
                result=res,
                formatted_result=formatted,
                success=True,
                currency_symbol=raw_currency,
                steps=f"({pct} / 100) * {val:g} = {res:g}"
            )

        # Pattern 2: Reduction / Discount (e.g., "cloud bill is ₹184,500 and we reduce it by 17%")
        match_reduce = re.search(
            r"[\$₹€£Rs\.\s]*([\d,]+(?:\.\d+)?).*?(?:reduce(?:d)?(?:\s+it)?(?:\s+by)?|discount(?:\s+of)?)\s*(\d+(?:\.\d+)?)\s*%",
            query,
            re.I
        )
        if not match_reduce:
            # Inverted phrasing: "reduce ₹184,500 by 17%"
            match_reduce = re.search(
                r"(?:reduce(?:d)?(?:\s+it)?(?:\s+by)?|discount(?:\s+of)?)\s*[\$₹€£Rs\.\s]*([\d,]+(?:\.\d+)?)\s*(?:by)?\s*(\d+(?:\.\d+)?)\s*%",
                query,
                re.I
            )

        if match_reduce:
            val = float(match_reduce.group(1).replace(",", ""))
            pct = float(match_reduce.group(2))
            reduction = (pct / 100.0) * val
            res = val - reduction
            expr = f"{val:g} - ({pct}% of {val:g})"
            formatted = cls._format_number(res, raw_currency)
            return CalculationResult(
                expression=expr,
                result=res,
                formatted_result=formatted,
                success=True,
                currency_symbol=raw_currency,
                steps=f"{val:g} - ({pct}/100 * {val:g}) = {val:g} - {reduction:g} = {res:g}"
            )

        # Pattern 3: Increase (e.g., "bill of ₹10,000 increased by 15%")
        match_increase = re.search(
            r"[\$₹€£Rs\.\s]*([\d,]+(?:\.\d+)?).*?(?:increase(?:d)?(?:\s+it)?(?:\s+by)?|markup(?:\s+of)?)\s*(\d+(?:\.\d+)?)\s*%",
            query,
            re.I
        )
        if not match_increase:
            # Inverted phrasing: "increase ₹10,000 by 15%" / "Increase 500 by 12%"
            match_increase = re.search(
                r"(?:increase(?:d)?(?:\s+it)?(?:\s+by)?|markup(?:\s+of)?)\s*[\$₹€£Rs\.\s]*([\d,]+(?:\.\d+)?)\s*(?:by)?\s*(\d+(?:\.\d+)?)\s*%",
                query,
                re.I
            )
        if match_increase:
            val = float(match_increase.group(1).replace(",", ""))
            pct = float(match_increase.group(2))
            increase = (pct / 100.0) * val
            res = val + increase
            expr = f"{val:g} + ({pct}% of {val:g})"
            formatted = cls._format_number(res, raw_currency)
            return CalculationResult(
                expression=expr,
                result=res,
                formatted_result=formatted,
                success=True,
                currency_symbol=raw_currency,
                steps=f"{val:g} + ({pct}/100 * {val:g}) = {val:g} + {increase:g} = {res:g}"
            )

        # Pattern 4: General expression sanitization
        expr = cls._sanitize_expression(query)
        if not expr:
            return CalculationResult(
                expression=query,
                result=None,
                formatted_result="Could not parse mathematical expression",
                success=False,
                error_message="Invalid expression syntax"
            )

        return cls.safe_eval(expr, currency_symbol=raw_currency)

    @classmethod
    def safe_eval(cls, expression: str, currency_symbol: Optional[str] = None) -> CalculationResult:
        """Safely evaluate arithmetic expression using python AST without eval()."""
        try:
            tree = ast.parse(expression, mode="eval")
            res = cls._eval_node(tree.body)
            # Simplify int if whole number
            if isinstance(res, float) and res.is_integer():
                res = int(res)

            formatted = cls._format_number(res, currency_symbol)
            return CalculationResult(
                expression=expression,
                result=res,
                formatted_result=formatted,
                success=True,
                currency_symbol=currency_symbol,
                steps=f"{expression} = {res}"
            )
        except ZeroDivisionError:
            return CalculationResult(
                expression=expression,
                result=None,
                formatted_result="Error: Division by zero",
                success=False,
                error_message="Division by zero encountered."
            )
        except Exception as e:
            return CalculationResult(
                expression=expression,
                result=None,
                formatted_result=f"Error evaluating calculation: {e}",
                success=False,
                error_message=str(e)
            )

    @classmethod
    def _eval_node(cls, node: ast.AST) -> Union[int, float]:
        """Recursive node evaluator strictly restricted to safe arithmetic operators."""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant type: {type(node.value)}")

        # Python < 3.8 compatibility
        if isinstance(node, ast.Num):
            return node.n

        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in cls.OPERATORS:
                raise ValueError(f"Unsupported operator: {op_type.__name__}")
            left = cls._eval_node(node.left)
            right = cls._eval_node(node.right)
            op_func = cls.OPERATORS[op_type]
            return op_func(left, right)

        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in cls.OPERATORS:
                raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
            operand = cls._eval_node(node.operand)
            op_func = cls.OPERATORS[op_type]
            return op_func(operand)

        raise ValueError(f"Unsupported AST node type: {type(node).__name__}")

    @classmethod
    def _sanitize_expression(cls, query: str) -> str:
        """Strip conversational words and retain pure arithmetic notation."""
        cleaned = query
        # Remove currency symbols and word keywords
        for sym in cls.CURRENCY_SYMBOLS:
            cleaned = cleaned.replace(sym, " ")

        cleaned = re.sub(r"\b(what is|calculate|compute|how much is|equal to|the value of)\b", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\bplus\b", "+", cleaned, flags=re.I)
        cleaned = re.sub(r"\bminus\b", "-", cleaned, flags=re.I)
        cleaned = re.sub(r"\btimes|multiplied by\b", "*", cleaned, flags=re.I)
        cleaned = re.sub(r"\bdivided by\b", "/", cleaned, flags=re.I)

        # Remove commas inside numbers (e.g. 184,500 -> 184500)
        cleaned = re.sub(r"(\d),(\d)", r"\1\2", cleaned)

        # Retain only valid characters
        cleaned = re.sub(r"[^\d\+\-\*\/\(\)\.\s\%]", "", cleaned).strip()

        # Handle percentage like "100 - 15%" -> "100 * (1 - 0.15)" or "50 * 20%"
        pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%", cleaned)
        if pct_match:
            pct_val = float(pct_match.group(1)) / 100.0
            cleaned = cleaned.replace(pct_match.group(0), str(pct_val))

        return cleaned.strip()

    @classmethod
    def _format_number(cls, num: Union[float, int], currency_symbol: Optional[str] = None) -> str:
        """Format number with thousands separators and optional currency symbol."""
        if isinstance(num, float):
            if num.is_integer():
                formatted = f"{int(num):,}"
            else:
                formatted = f"{num:,.2f}"
        else:
            formatted = f"{num:,}"

        if currency_symbol:
            prefix = currency_symbol.strip()
            if prefix in ["₹", "$", "€", "£", "¥"]:
                return f"{prefix}{formatted}"
            return f"{prefix} {formatted}"

        return formatted
