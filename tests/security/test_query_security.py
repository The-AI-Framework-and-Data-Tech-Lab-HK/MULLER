# SPDX-License-Identifier: MPL-2.0
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Xueling Lin

"""
Security tests for query system.

Tests that the query system properly blocks malicious queries and prevents
arbitrary code execution attacks.
"""

import pytest
from muller.core.query.safe_evaluator import SafeQueryEvaluator


class TestQuerySecurityCodeExecution:
    """Test that code execution attacks are blocked."""

    def test_block_os_system_via_import(self):
        """Test that os.system() attacks are blocked"""
        with pytest.raises(ValueError, match="not allowed"):
            SafeQueryEvaluator("__import__('os').system('ls')")

        with pytest.raises(ValueError, match="not allowed"):
            SafeQueryEvaluator("__import__('os').system('rm -rf /')")

    def test_block_subprocess_attacks(self):
        """Test that subprocess attacks are blocked"""
        with pytest.raises(ValueError, match="not allowed"):
            SafeQueryEvaluator("__import__('subprocess').call(['ls'])")

        with pytest.raises(ValueError, match="not allowed"):
            SafeQueryEvaluator("__import__('subprocess').Popen('ls')")

    def test_block_eval_exec_attacks(self):
        """Test that eval/exec attacks are blocked"""
        with pytest.raises(ValueError, match="not allowed"):
            SafeQueryEvaluator("eval('1+1')")

        with pytest.raises(ValueError, match="not allowed"):
            SafeQueryEvaluator("exec('print(1)')")

        with pytest.raises(ValueError, match="not allowed"):
            SafeQueryEvaluator("compile('1+1', '', 'eval')")

    def test_block_file_access(self):
        """Test that file access attacks are blocked"""
        with pytest.raises(ValueError, match="not allowed"):
            SafeQueryEvaluator("open('/etc/passwd').read()")

        with pytest.raises(ValueError, match="not allowed"):
            SafeQueryEvaluator("open('/etc/shadow', 'r')")

        with pytest.raises(ValueError, match="not allowed"):
            SafeQueryEvaluator("open('file.txt', 'w').write('data')")


class TestQuerySecurityReflectionAttacks:
    """Test that reflection-based attacks are blocked."""

    def test_block_class_access(self):
        """Test that __class__ access is blocked"""
        with pytest.raises(ValueError, match="private attribute"):
            SafeQueryEvaluator("x.__class__")

        with pytest.raises(ValueError, match="private attribute"):
            SafeQueryEvaluator("x.__class__.__bases__")

    def test_block_subclasses_access(self):
        """Test that __subclasses__ access is blocked"""
        # This is blocked by both private attribute check and function call check
        with pytest.raises(ValueError, match="not allowed"):
            SafeQueryEvaluator("x.__class__.__bases__[0].__subclasses__()")

    def test_block_globals_locals_access(self):
        """Test that globals()/locals() access is blocked"""
        with pytest.raises(ValueError, match="not allowed"):
            SafeQueryEvaluator("globals()")

        with pytest.raises(ValueError, match="not allowed"):
            SafeQueryEvaluator("locals()")

    def test_block_vars_dir_access(self):
        """Test that vars()/dir() access is blocked"""
        with pytest.raises(ValueError, match="not allowed"):
            SafeQueryEvaluator("vars(x)")

        with pytest.raises(ValueError, match="not allowed"):
            SafeQueryEvaluator("dir(x)")

    def test_block_getattr_setattr(self):
        """Test that getattr/setattr are blocked"""
        with pytest.raises(ValueError, match="not allowed"):
            SafeQueryEvaluator("getattr(x, 'attr')")

        with pytest.raises(ValueError, match="not allowed"):
            SafeQueryEvaluator("setattr(x, 'attr', 'value')")

    def test_block_delattr(self):
        """Test that delattr is blocked"""
        with pytest.raises(ValueError, match="not allowed"):
            SafeQueryEvaluator("delattr(x, 'attr')")


class TestQuerySecurityNetworkAttacks:
    """Test that network-based attacks are blocked."""

    def test_block_urllib_requests(self):
        """Test that urllib requests are blocked"""
        with pytest.raises(ValueError, match="not allowed"):
            SafeQueryEvaluator("__import__('urllib.request').urlopen('http://evil.com')")

    def test_block_requests_library(self):
        """Test that requests library usage is blocked"""
        with pytest.raises(ValueError, match="not allowed"):
            SafeQueryEvaluator("__import__('requests').get('http://evil.com')")

    def test_block_socket_access(self):
        """Test that socket access is blocked"""
        with pytest.raises(ValueError, match="not allowed"):
            SafeQueryEvaluator("__import__('socket').socket()")


class TestQuerySecurityDataExfiltration:
    """Test that data exfiltration attempts are blocked."""

    def test_block_pickle_attacks(self):
        """Test that pickle attacks are blocked"""
        with pytest.raises(ValueError, match="not allowed"):
            SafeQueryEvaluator("__import__('pickle').dumps(x)")

        with pytest.raises(ValueError, match="not allowed"):
            SafeQueryEvaluator("__import__('pickle').loads(data)")

    def test_block_json_dumps(self):
        """Test that json.dumps is blocked (could exfiltrate data)"""
        with pytest.raises(ValueError, match="not allowed"):
            SafeQueryEvaluator("__import__('json').dumps(x)")

    def test_block_base64_encoding(self):
        """Test that base64 encoding is blocked"""
        with pytest.raises(ValueError, match="not allowed"):
            SafeQueryEvaluator("__import__('base64').b64encode(x)")


class TestQuerySecurityEnvironmentAccess:
    """Test that environment access is blocked."""

    def test_block_os_environ_access(self):
        """Test that os.environ access is blocked"""
        with pytest.raises(ValueError, match="not allowed"):
            SafeQueryEvaluator("__import__('os').environ")

        with pytest.raises(ValueError, match="not allowed"):
            SafeQueryEvaluator("__import__('os').getenv('PATH')")

    def test_block_sys_access(self):
        """Test that sys module access is blocked"""
        with pytest.raises(ValueError, match="not allowed"):
            SafeQueryEvaluator("__import__('sys').exit()")

        with pytest.raises(ValueError, match="not allowed"):
            SafeQueryEvaluator("__import__('sys').path")


class TestQuerySecurityComplexAttacks:
    """Test complex multi-stage attacks."""

    def test_block_chained_reflection_attack(self):
        """Test complex reflection-based attack chain"""
        # Classic Python sandbox escape
        attack = "().__class__.__bases__[0].__subclasses__()[104].__init__.__globals__['sys'].modules['os'].system('ls')"
        with pytest.raises(ValueError, match="not allowed"):
            SafeQueryEvaluator(attack)

    def test_block_comprehension_with_side_effects(self):
        """Test comprehension with side effects"""
        with pytest.raises(ValueError, match="not allowed"):
            SafeQueryEvaluator("[print(x) for x in range(10)]")

    def test_block_lambda_with_import(self):
        """Test lambda with import"""
        with pytest.raises(ValueError, match="not allowed"):
            SafeQueryEvaluator("(lambda: __import__('os'))()")


class TestQuerySecurityAllowedOperations:
    """Test that legitimate operations are still allowed."""

    def test_allow_safe_comparisons(self):
        """Test that safe comparisons work"""
        evaluator = SafeQueryEvaluator("x > 5 and y < 10")
        assert evaluator.evaluate({'x': 6, 'y': 9}) is True

    def test_allow_safe_arithmetic(self):
        """Test that safe arithmetic works"""
        evaluator = SafeQueryEvaluator("x * y > 100")
        assert evaluator.evaluate({'x': 10, 'y': 11}) is True

    def test_allow_safe_attribute_access(self):
        """Test that safe attribute access works"""
        class MockObject:
            shape = (100, 100)
            min = 0
            max = 255

        evaluator = SafeQueryEvaluator("x.shape[0] > 50")
        assert evaluator.evaluate({'x': MockObject()}) is True

    def test_allow_safe_membership(self):
        """Test that safe membership tests work"""
        evaluator = SafeQueryEvaluator("1 in x")
        assert evaluator.evaluate({'x': [1, 2, 3]}) is True

    def test_allow_complex_safe_expressions(self):
        """Test complex but safe expressions"""
        evaluator = SafeQueryEvaluator(
            "(x.shape[0] * x.shape[1] > 1000) and (x.min >= 0) and (x.max <= 255)"
        )

        class MockObject:
            shape = (50, 30)
            min = 0
            max = 200

        assert evaluator.evaluate({'x': MockObject()}) is True


class TestQuerySecurityRegressionTests:
    """Regression tests for known attack vectors."""

    def test_cve_style_attacks(self):
        """Test various CVE-style attack patterns"""
        attacks = [
            # Code execution
            "__import__('os').system('whoami')",
            "exec('import os; os.system(\"ls\")')",
            "eval('__import__(\"os\").system(\"ls\")')",

            # File access
            "open('/etc/passwd', 'r').read()",
            "__import__('pathlib').Path('/etc/passwd').read_text()",

            # Network access
            "__import__('urllib.request').urlopen('http://evil.com').read()",

            # Reflection attacks
            "().__class__.__bases__[0].__subclasses__()",
            "[].__class__.__base__.__subclasses__()",
        ]

        for attack in attacks:
            with pytest.raises(ValueError, match="not allowed"):
                SafeQueryEvaluator(attack)

    def test_obfuscated_attacks(self):
        """Test obfuscated attack patterns"""
        # These should still be blocked even with obfuscation
        with pytest.raises(ValueError, match="not allowed"):
            # Using getattr to access __import__
            SafeQueryEvaluator("getattr(__builtins__, '__import__')('os')")

        with pytest.raises(ValueError, match="not allowed"):
            # Nested function calls
            SafeQueryEvaluator("eval(compile('1+1', '', 'eval'))")
