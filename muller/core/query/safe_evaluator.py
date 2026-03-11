# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

"""
Safe query evaluator using AST-based whitelist approach.

This module provides a secure alternative to eval() for evaluating user-provided
query expressions. It uses Python's ast module to parse and validate queries,
ensuring only safe operations are allowed.
"""

import ast
import operator
from typing import Any, Dict, Set


class SafeQueryEvaluator:
    """
    Safe expression evaluator for dataset queries.

    Uses AST parsing with a whitelist approach to prevent arbitrary code execution
    while supporting all necessary query operations.

    Allowed operations:
    - Comparison: ==, !=, <, <=, >, >=
    - Logical: and, or, not
    - Membership: in, not in
    - Arithmetic: +, -, *, /, //, %, **
    - Attribute access: .shape, .min, .max, .mean, .size, .sample_info
    - Indexing: [index]

    Blocked operations:
    - Function calls
    - Imports
    - Lambda expressions
    - List/dict comprehensions
    - Attribute access to private/dunder attributes
    """

    # Allowed binary operations
    ALLOWED_BINARY_OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }

    # Allowed comparison operations
    ALLOWED_COMPARE_OPS = {
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.In: lambda x, y: x in y,
        ast.NotIn: lambda x, y: x not in y,
    }

    # Allowed boolean operations
    ALLOWED_BOOL_OPS = {
        ast.And: lambda x, y: x and y,
        ast.Or: lambda x, y: x or y,
    }

    # Allowed unary operations
    ALLOWED_UNARY_OPS = {
        ast.Not: operator.not_,
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    # Allowed attributes (from EvalObject)
    ALLOWED_ATTRS: Set[str] = {
        'shape', 'min', 'max', 'mean', 'size', 'sample_info', 'numpy_value'
    }

    def __init__(self, query: str):
        """
        Initialize the safe evaluator with a query string.

        Args:
            query: Query expression string to evaluate

        Raises:
            ValueError: If query contains disallowed operations
            SyntaxError: If query has invalid syntax
        """
        self.query = query
        try:
            self.tree = ast.parse(query, mode='eval')
        except SyntaxError as e:
            raise SyntaxError(f"Invalid query syntax: {e}")

        # Validate the AST before allowing evaluation
        self._validate(self.tree.body)

    def _validate(self, node: ast.AST) -> None:
        """
        Recursively validate AST nodes to ensure only safe operations.

        Args:
            node: AST node to validate

        Raises:
            ValueError: If node contains disallowed operations
        """
        if isinstance(node, ast.BinOp):
            # Binary operations: +, -, *, /, //, %, **
            if type(node.op) not in self.ALLOWED_BINARY_OPS:
                raise ValueError(
                    f"Binary operation {type(node.op).__name__} not allowed in query"
                )
            self._validate(node.left)
            self._validate(node.right)

        elif isinstance(node, ast.Compare):
            # Comparison operations: ==, !=, <, <=, >, >=, in, not in
            for op in node.ops:
                if type(op) not in self.ALLOWED_COMPARE_OPS:
                    raise ValueError(
                        f"Comparison operation {type(op).__name__} not allowed in query"
                    )
            self._validate(node.left)
            for comparator in node.comparators:
                self._validate(comparator)

        elif isinstance(node, ast.BoolOp):
            # Boolean operations: and, or
            if type(node.op) not in self.ALLOWED_BOOL_OPS:
                raise ValueError(
                    f"Boolean operation {type(node.op).__name__} not allowed in query"
                )
            for value in node.values:
                self._validate(value)

        elif isinstance(node, ast.UnaryOp):
            # Unary operations: not, +, -
            if type(node.op) not in self.ALLOWED_UNARY_OPS:
                raise ValueError(
                    f"Unary operation {type(node.op).__name__} not allowed in query"
                )
            self._validate(node.operand)

        elif isinstance(node, ast.Attribute):
            # Attribute access: obj.attr
            # Block access to private/dunder attributes
            if node.attr.startswith('_'):
                raise ValueError(
                    f"Access to private attribute '{node.attr}' not allowed in query"
                )
            # Note: We don't enforce ALLOWED_ATTRS here because attributes might be
            # on nested objects (e.g., metadata.timestamp). Runtime will handle invalid attrs.
            self._validate(node.value)

        elif isinstance(node, ast.Subscript):
            # Indexing: obj[index]
            self._validate(node.value)
            self._validate(node.slice)

        elif isinstance(node, ast.Name):
            # Variable reference - OK
            pass

        elif isinstance(node, ast.Constant):
            # Literal value (Python 3.8+) - OK
            pass

        elif isinstance(node, ast.Num):
            # Numeric literal (Python 3.7) - OK
            pass

        elif isinstance(node, ast.Str):
            # String literal (Python 3.7) - OK
            pass

        elif isinstance(node, ast.List):
            # List literal - OK
            for elt in node.elts:
                self._validate(elt)

        elif isinstance(node, ast.Tuple):
            # Tuple literal - OK
            for elt in node.elts:
                self._validate(elt)

        elif isinstance(node, ast.Call):
            # Function calls - BLOCKED
            raise ValueError(
                "Function calls not allowed in query. "
                "Use attribute access instead (e.g., 'tensor.min' not 'min(tensor)')"
            )

        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            # Imports - BLOCKED
            raise ValueError("Import statements not allowed in query")

        elif isinstance(node, ast.Lambda):
            # Lambda expressions - BLOCKED
            raise ValueError("Lambda expressions not allowed in query")

        elif isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
            # Comprehensions - BLOCKED
            raise ValueError("Comprehensions not allowed in query")

        else:
            # Any other node type - BLOCKED
            raise ValueError(
                f"AST node type {type(node).__name__} not allowed in query"
            )

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """
        Safely evaluate the query with the given context.

        Args:
            context: Dictionary mapping variable names to their values

        Returns:
            Boolean result of query evaluation

        Raises:
            NameError: If query references undefined variables
            AttributeError: If query accesses non-existent attributes
            TypeError: If query performs invalid operations
        """
        try:
            result = self._eval_node(self.tree.body, context)
            # Ensure result is boolean
            return bool(result)
        except Exception as e:
            # Re-raise with more context
            raise type(e)(f"Error evaluating query '{self.query}': {e}") from e

    def _eval_node(self, node: ast.AST, context: Dict[str, Any]) -> Any:
        """
        Recursively evaluate an AST node.

        Args:
            node: AST node to evaluate
            context: Variable context

        Returns:
            Evaluation result
        """
        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left, context)
            right = self._eval_node(node.right, context)
            op_func = self.ALLOWED_BINARY_OPS[type(node.op)]
            return op_func(left, right)

        elif isinstance(node, ast.Compare):
            # Handle chained comparisons: a < b < c
            left = self._eval_node(node.left, context)
            for op, comparator in zip(node.ops, node.comparators):
                right = self._eval_node(comparator, context)
                op_func = self.ALLOWED_COMPARE_OPS[type(op)]
                if not op_func(left, right):
                    return False
                left = right
            return True

        elif isinstance(node, ast.BoolOp):
            # Short-circuit evaluation for and/or
            op_func = self.ALLOWED_BOOL_OPS[type(node.op)]
            result = self._eval_node(node.values[0], context)
            for value in node.values[1:]:
                result = op_func(result, self._eval_node(value, context))
                # Short-circuit
                if isinstance(node.op, ast.And) and not result:
                    return False
                if isinstance(node.op, ast.Or) and result:
                    return True
            return result

        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand, context)
            op_func = self.ALLOWED_UNARY_OPS[type(node.op)]
            return op_func(operand)

        elif isinstance(node, ast.Attribute):
            obj = self._eval_node(node.value, context)
            return getattr(obj, node.attr)

        elif isinstance(node, ast.Subscript):
            obj = self._eval_node(node.value, context)
            index = self._eval_node(node.slice, context)
            return obj[index]

        elif isinstance(node, ast.Name):
            if node.id not in context:
                raise NameError(f"Variable '{node.id}' not found in query context")
            return context[node.id]

        elif isinstance(node, ast.Constant):
            # Python 3.8+
            return node.value

        elif isinstance(node, ast.Num):
            # Python 3.7
            return node.n

        elif isinstance(node, ast.Str):
            # Python 3.7
            return node.s

        elif isinstance(node, ast.List):
            return [self._eval_node(elt, context) for elt in node.elts]

        elif isinstance(node, ast.Tuple):
            return tuple(self._eval_node(elt, context) for elt in node.elts)

        else:
            # Should never reach here if validation is correct
            raise ValueError(f"Cannot evaluate AST node type {type(node).__name__}")
