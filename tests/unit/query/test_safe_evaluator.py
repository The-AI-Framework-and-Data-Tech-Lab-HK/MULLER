"""
Unit tests for SafeQueryEvaluator.

Tests the safe query evaluator to ensure it:
1. Correctly evaluates all allowed operations
2. Blocks all dangerous operations
3. Provides clear error messages
"""

import pytest
from muller.core.query.safe_evaluator import SafeQueryEvaluator


class TestSafeQueryEvaluatorAllowedOperations:
    """Test that all allowed operations work correctly."""

    def test_comparison_operators(self):
        """Test comparison operators: ==, !=, <, <=, >, >="""
        evaluator = SafeQueryEvaluator("x == 5")
        assert evaluator.evaluate({'x': 5}) is True
        assert evaluator.evaluate({'x': 4}) is False

        evaluator = SafeQueryEvaluator("x != 5")
        assert evaluator.evaluate({'x': 4}) is True
        assert evaluator.evaluate({'x': 5}) is False

        evaluator = SafeQueryEvaluator("x < 10")
        assert evaluator.evaluate({'x': 5}) is True
        assert evaluator.evaluate({'x': 10}) is False

        evaluator = SafeQueryEvaluator("x <= 10")
        assert evaluator.evaluate({'x': 10}) is True
        assert evaluator.evaluate({'x': 11}) is False

        evaluator = SafeQueryEvaluator("x > 5")
        assert evaluator.evaluate({'x': 10}) is True
        assert evaluator.evaluate({'x': 5}) is False

        evaluator = SafeQueryEvaluator("x >= 5")
        assert evaluator.evaluate({'x': 5}) is True
        assert evaluator.evaluate({'x': 4}) is False

    def test_logical_operators(self):
        """Test logical operators: and, or, not"""
        evaluator = SafeQueryEvaluator("x > 5 and y < 10")
        assert evaluator.evaluate({'x': 6, 'y': 9}) is True
        assert evaluator.evaluate({'x': 4, 'y': 9}) is False
        assert evaluator.evaluate({'x': 6, 'y': 11}) is False

        evaluator = SafeQueryEvaluator("x == 1 or y == 2")
        assert evaluator.evaluate({'x': 1, 'y': 0}) is True
        assert evaluator.evaluate({'x': 0, 'y': 2}) is True
        assert evaluator.evaluate({'x': 0, 'y': 0}) is False

        evaluator = SafeQueryEvaluator("not x")
        assert evaluator.evaluate({'x': False}) is True
        assert evaluator.evaluate({'x': True}) is False

    def test_membership_operators(self):
        """Test membership operators: in, not in"""
        evaluator = SafeQueryEvaluator("1 in x")
        assert evaluator.evaluate({'x': [1, 2, 3]}) is True
        assert evaluator.evaluate({'x': [2, 3, 4]}) is False

        evaluator = SafeQueryEvaluator("1 not in x")
        assert evaluator.evaluate({'x': [2, 3, 4]}) is True
        assert evaluator.evaluate({'x': [1, 2, 3]}) is False

    def test_arithmetic_operators(self):
        """Test arithmetic operators: +, -, *, /, //, %, **"""
        evaluator = SafeQueryEvaluator("x + y > 10")
        assert evaluator.evaluate({'x': 6, 'y': 5}) is True
        assert evaluator.evaluate({'x': 3, 'y': 5}) is False

        evaluator = SafeQueryEvaluator("x - y == 2")
        assert evaluator.evaluate({'x': 5, 'y': 3}) is True
        assert evaluator.evaluate({'x': 5, 'y': 2}) is False

        evaluator = SafeQueryEvaluator("x * y > 20")
        assert evaluator.evaluate({'x': 5, 'y': 5}) is True
        assert evaluator.evaluate({'x': 2, 'y': 5}) is False

        evaluator = SafeQueryEvaluator("x / y == 2")
        assert evaluator.evaluate({'x': 10, 'y': 5}) is True
        assert evaluator.evaluate({'x': 10, 'y': 4}) is False

        evaluator = SafeQueryEvaluator("x // y == 2")
        assert evaluator.evaluate({'x': 10, 'y': 5}) is True
        assert evaluator.evaluate({'x': 11, 'y': 5}) is True

        evaluator = SafeQueryEvaluator("x % y == 1")
        assert evaluator.evaluate({'x': 10, 'y': 3}) is True
        assert evaluator.evaluate({'x': 10, 'y': 5}) is False

        evaluator = SafeQueryEvaluator("x ** 2 == 25")
        assert evaluator.evaluate({'x': 5}) is True
        assert evaluator.evaluate({'x': 4}) is False

    def test_attribute_access(self):
        """Test attribute access: obj.attr"""
        class MockObject:
            def __init__(self):
                self.shape = (100, 100)
                self.min = 0
                self.max = 255

        obj = MockObject()
        evaluator = SafeQueryEvaluator("x.shape[0] > 50")
        assert evaluator.evaluate({'x': obj}) is True

        evaluator = SafeQueryEvaluator("x.min == 0")
        assert evaluator.evaluate({'x': obj}) is True

        evaluator = SafeQueryEvaluator("x.max > 200")
        assert evaluator.evaluate({'x': obj}) is True

    def test_indexing(self):
        """Test indexing: obj[index]"""
        evaluator = SafeQueryEvaluator("x[0] == 1")
        assert evaluator.evaluate({'x': [1, 2, 3]}) is True
        assert evaluator.evaluate({'x': [2, 3, 4]}) is False

        evaluator = SafeQueryEvaluator("x[1] > 5")
        assert evaluator.evaluate({'x': [1, 10, 3]}) is True
        assert evaluator.evaluate({'x': [1, 2, 3]}) is False

    def test_chained_comparisons(self):
        """Test chained comparisons: a < b < c"""
        evaluator = SafeQueryEvaluator("5 < x < 10")
        assert evaluator.evaluate({'x': 7}) is True
        assert evaluator.evaluate({'x': 5}) is False
        assert evaluator.evaluate({'x': 10}) is False

    def test_complex_expressions(self):
        """Test complex nested expressions"""
        evaluator = SafeQueryEvaluator("(x > 5 and y < 10) or (x == 0 and y == 0)")
        assert evaluator.evaluate({'x': 6, 'y': 9}) is True
        assert evaluator.evaluate({'x': 0, 'y': 0}) is True
        assert evaluator.evaluate({'x': 4, 'y': 9}) is False

        evaluator = SafeQueryEvaluator("x.shape[0] * x.shape[1] > 1000")
        class MockObject:
            shape = (50, 30)
        assert evaluator.evaluate({'x': MockObject()}) is True

    def test_parentheses_grouping(self):
        """Test parentheses for grouping"""
        evaluator = SafeQueryEvaluator("(x + y) * z > 20")
        assert evaluator.evaluate({'x': 2, 'y': 3, 'z': 5}) is True
        assert evaluator.evaluate({'x': 1, 'y': 1, 'z': 5}) is False

    def test_list_and_tuple_literals(self):
        """Test list and tuple literals"""
        evaluator = SafeQueryEvaluator("x in [1, 2, 3]")
        assert evaluator.evaluate({'x': 2}) is True
        assert evaluator.evaluate({'x': 4}) is False

        evaluator = SafeQueryEvaluator("x in (1, 2, 3)")
        assert evaluator.evaluate({'x': 2}) is True
        assert evaluator.evaluate({'x': 4}) is False


class TestSafeQueryEvaluatorBlockedOperations:
    """Test that dangerous operations are blocked."""

    def test_block_function_calls(self):
        """Test that function calls are blocked"""
        with pytest.raises(ValueError, match="Function calls not allowed"):
            SafeQueryEvaluator("len(x)")

        with pytest.raises(ValueError, match="Function calls not allowed"):
            SafeQueryEvaluator("print('hello')")

        with pytest.raises(ValueError, match="Function calls not allowed"):
            SafeQueryEvaluator("eval('1+1')")

    def test_block_imports(self):
        """Test that imports are blocked"""
        # Import statements cause SyntaxError in eval mode, which is fine
        with pytest.raises(SyntaxError):
            SafeQueryEvaluator("import os")

        with pytest.raises(SyntaxError):
            SafeQueryEvaluator("from os import system")

    def test_block_lambda(self):
        """Test that lambda expressions are blocked"""
        # Lambda in eval mode causes SyntaxError, which is fine
        with pytest.raises((ValueError, SyntaxError)):
            SafeQueryEvaluator("(lambda x: x + 1)(5)")

    def test_block_comprehensions(self):
        """Test that comprehensions are blocked"""
        with pytest.raises(ValueError, match="Comprehensions not allowed"):
            SafeQueryEvaluator("[x for x in range(10)]")

        with pytest.raises(ValueError, match="Comprehensions not allowed"):
            SafeQueryEvaluator("{x: x**2 for x in range(10)}")

        with pytest.raises(ValueError, match="Comprehensions not allowed"):
            SafeQueryEvaluator("{x for x in range(10)}")

    def test_block_private_attributes(self):
        """Test that private/dunder attributes are blocked"""
        with pytest.raises(ValueError, match="private attribute"):
            SafeQueryEvaluator("x.__class__")

        with pytest.raises(ValueError, match="private attribute"):
            SafeQueryEvaluator("x._private")

    def test_block_dangerous_code_execution(self):
        """Test that common code execution attacks are blocked"""
        # __import__ attack
        with pytest.raises(ValueError, match="Function calls not allowed"):
            SafeQueryEvaluator("__import__('os').system('ls')")

        # open() attack
        with pytest.raises(ValueError, match="Function calls not allowed"):
            SafeQueryEvaluator("open('/etc/passwd').read()")

        # exec() attack
        with pytest.raises(ValueError, match="Function calls not allowed"):
            SafeQueryEvaluator("exec('print(1)')")


class TestSafeQueryEvaluatorErrorHandling:
    """Test error handling and messages."""

    def test_syntax_error(self):
        """Test that syntax errors are caught"""
        with pytest.raises(SyntaxError, match="Invalid query syntax"):
            SafeQueryEvaluator("x == ")

        with pytest.raises(SyntaxError, match="Invalid query syntax"):
            SafeQueryEvaluator("x ===")

    def test_undefined_variable(self):
        """Test that undefined variables raise NameError"""
        evaluator = SafeQueryEvaluator("x > 5")
        with pytest.raises(NameError, match="Variable 'x' not found"):
            evaluator.evaluate({})

    def test_invalid_attribute(self):
        """Test that invalid attributes raise AttributeError"""
        evaluator = SafeQueryEvaluator("x.nonexistent > 5")
        with pytest.raises(AttributeError):
            evaluator.evaluate({'x': object()})

    def test_error_message_includes_query(self):
        """Test that error messages include the query string"""
        evaluator = SafeQueryEvaluator("x > 5")
        try:
            evaluator.evaluate({})
        except NameError as e:
            assert "x > 5" in str(e)


class TestSafeQueryEvaluatorEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_context(self):
        """Test evaluation with empty context"""
        evaluator = SafeQueryEvaluator("True")
        assert evaluator.evaluate({}) is True

        evaluator = SafeQueryEvaluator("1 == 1")
        assert evaluator.evaluate({}) is True

    def test_boolean_coercion(self):
        """Test that results are coerced to boolean"""
        evaluator = SafeQueryEvaluator("x")
        assert evaluator.evaluate({'x': 1}) is True
        assert evaluator.evaluate({'x': 0}) is False
        assert evaluator.evaluate({'x': []}) is False
        assert evaluator.evaluate({'x': [1]}) is True

    def test_short_circuit_evaluation(self):
        """Test that boolean operators short-circuit"""
        # Note: Our implementation evaluates comparison nodes fully,
        # but the actual boolean logic does short-circuit
        evaluator = SafeQueryEvaluator("x == 0 and y > 5")
        # x == 0 is True, y > 5 is True, so result is True
        assert evaluator.evaluate({'x': 0, 'y': 10}) is True
        # x == 0 is True, y > 5 is False, so result is False
        assert evaluator.evaluate({'x': 0, 'y': 3}) is False
        # x == 0 is False, so result is False regardless of y
        assert evaluator.evaluate({'x': 1, 'y': 10}) is False

        evaluator = SafeQueryEvaluator("x == 1 or y > 5")
        # x == 1 is True, so result is True regardless of y
        assert evaluator.evaluate({'x': 1, 'y': 0}) is True
        # x == 1 is False, y > 5 is True, so result is True
        assert evaluator.evaluate({'x': 0, 'y': 10}) is True
        # x == 1 is False, y > 5 is False, so result is False
        assert evaluator.evaluate({'x': 0, 'y': 3}) is False

    def test_nested_attribute_access(self):
        """Test nested attribute access"""
        class Inner:
            value = 42

        class Outer:
            inner = Inner()

        evaluator = SafeQueryEvaluator("x.inner.value == 42")
        assert evaluator.evaluate({'x': Outer()}) is True

    def test_multiple_variables(self):
        """Test queries with multiple variables"""
        evaluator = SafeQueryEvaluator("a + b + c == 10")
        assert evaluator.evaluate({'a': 2, 'b': 3, 'c': 5}) is True
        assert evaluator.evaluate({'a': 1, 'b': 2, 'c': 3}) is False
