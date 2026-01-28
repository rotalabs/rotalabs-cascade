"""Tests for condition evaluator in rotalabs-cascade.

Tests cover:
- Comparison operators (==, !=, >, <, >=, <=)
- Collection operators (IN, NOT_IN, CONTAINS)
- Existence operators (EXISTS, IS_NULL)
- Pattern matching (MATCHES with regex)
- Composite conditions (AND, OR, NOT)
- Aggregation operators (ALL, ANY, NONE)
- Math operators (SUM, AVG, MIN, MAX, COUNT)
"""

import pytest

from rotalabs_cascade.evaluation.evaluator import ConditionEvaluator


class TestComparisonOperators:
    """Tests for basic comparison operators."""

    def test_equal_operator(self):
        """Test equality comparison."""
        evaluator = ConditionEvaluator()

        condition = {"field": "status", "operator": "==", "value": "active"}
        context = {"status": "active"}

        assert evaluator.evaluate(condition, context) is True

    def test_equal_operator_false(self):
        """Test equality comparison returns false."""
        evaluator = ConditionEvaluator()

        condition = {"field": "status", "operator": "==", "value": "active"}
        context = {"status": "inactive"}

        assert evaluator.evaluate(condition, context) is False

    def test_not_equal_operator(self):
        """Test not equal comparison."""
        evaluator = ConditionEvaluator()

        condition = {"field": "status", "operator": "!=", "value": "pending"}
        context = {"status": "active"}

        assert evaluator.evaluate(condition, context) is True

    def test_greater_than_operator(self):
        """Test greater than comparison."""
        evaluator = ConditionEvaluator()

        condition = {"field": "score", "operator": ">", "value": 50}
        context = {"score": 75}

        assert evaluator.evaluate(condition, context) is True

    def test_greater_than_or_equal_operator(self):
        """Test greater than or equal comparison."""
        evaluator = ConditionEvaluator()

        condition = {"field": "score", "operator": ">=", "value": 100}
        context = {"score": 100}

        assert evaluator.evaluate(condition, context) is True

    def test_less_than_operator(self):
        """Test less than comparison."""
        evaluator = ConditionEvaluator()

        condition = {"field": "age", "operator": "<", "value": 18}
        context = {"age": 15}

        assert evaluator.evaluate(condition, context) is True

    def test_less_than_or_equal_operator(self):
        """Test less than or equal comparison."""
        evaluator = ConditionEvaluator()

        condition = {"field": "count", "operator": "<=", "value": 10}
        context = {"count": 10}

        assert evaluator.evaluate(condition, context) is True

    def test_comparison_with_float(self):
        """Test comparisons with floating point numbers."""
        evaluator = ConditionEvaluator()

        condition = {"field": "confidence", "operator": ">", "value": 0.5}
        context = {"confidence": 0.75}

        assert evaluator.evaluate(condition, context) is True


class TestCollectionOperators:
    """Tests for collection operators."""

    def test_in_operator_list(self):
        """Test IN operator with list."""
        evaluator = ConditionEvaluator()

        condition = {"field": "status", "operator": "IN", "value": ["active", "pending"]}
        context = {"status": "active"}

        assert evaluator.evaluate(condition, context) is True

    def test_in_operator_not_present(self):
        """Test IN operator when value not in list."""
        evaluator = ConditionEvaluator()

        condition = {"field": "status", "operator": "IN", "value": ["active", "pending"]}
        context = {"status": "completed"}

        assert evaluator.evaluate(condition, context) is False

    def test_not_in_operator(self):
        """Test NOT_IN operator."""
        evaluator = ConditionEvaluator()

        condition = {"field": "status", "operator": "NOT_IN", "value": ["deleted", "archived"]}
        context = {"status": "active"}

        assert evaluator.evaluate(condition, context) is True

    def test_contains_operator_list(self):
        """Test CONTAINS operator with list."""
        evaluator = ConditionEvaluator()

        condition = {"field": "tags", "operator": "CONTAINS", "value": "urgent"}
        context = {"tags": ["urgent", "important", "review"]}

        assert evaluator.evaluate(condition, context) is True

    def test_contains_operator_string(self):
        """Test CONTAINS operator with string."""
        evaluator = ConditionEvaluator()

        condition = {"field": "message", "operator": "CONTAINS", "value": "error"}
        context = {"message": "An error occurred"}

        assert evaluator.evaluate(condition, context) is True

    def test_contains_operator_dict(self):
        """Test CONTAINS operator with dictionary."""
        evaluator = ConditionEvaluator()

        condition = {"field": "data", "operator": "CONTAINS", "value": "key1"}
        context = {"data": {"key1": "value1", "key2": "value2"}}

        assert evaluator.evaluate(condition, context) is True


class TestExistenceOperators:
    """Tests for existence operators."""

    def test_exists_operator_true(self):
        """Test EXISTS operator when field exists."""
        evaluator = ConditionEvaluator()

        condition = {"field": "optional_field", "operator": "EXISTS"}
        context = {"optional_field": "present"}

        assert evaluator.evaluate(condition, context) is True

    def test_exists_operator_false(self):
        """Test EXISTS operator when field missing."""
        evaluator = ConditionEvaluator()

        condition = {"field": "missing_field", "operator": "EXISTS"}
        context = {"other_field": "value"}

        assert evaluator.evaluate(condition, context) is False

    def test_is_null_operator_true(self):
        """Test IS_NULL operator when field is None."""
        evaluator = ConditionEvaluator()

        condition = {"field": "nullable_field", "operator": "IS_NULL"}
        context = {"nullable_field": None}

        assert evaluator.evaluate(condition, context) is True

    def test_is_null_operator_false(self):
        """Test IS_NULL operator when field has value."""
        evaluator = ConditionEvaluator()

        condition = {"field": "nullable_field", "operator": "IS_NULL"}
        context = {"nullable_field": "has_value"}

        assert evaluator.evaluate(condition, context) is False

    def test_is_null_operator_missing_field(self):
        """Test IS_NULL operator when field is missing."""
        evaluator = ConditionEvaluator()

        condition = {"field": "missing_field", "operator": "IS_NULL"}
        context = {"other_field": "value"}

        assert evaluator.evaluate(condition, context) is True


class TestPatternMatching:
    """Tests for pattern matching operators."""

    def test_matches_operator_simple(self):
        """Test MATCHES operator with simple pattern."""
        evaluator = ConditionEvaluator()

        condition = {"field": "email", "operator": "MATCHES", "value": r".*@.*\.com"}
        context = {"email": "user@example.com"}

        assert evaluator.evaluate(condition, context) is True

    def test_matches_operator_no_match(self):
        """Test MATCHES operator when pattern doesn't match."""
        evaluator = ConditionEvaluator()

        condition = {"field": "email", "operator": "MATCHES", "value": r".*@gmail\.com"}
        context = {"email": "user@yahoo.com"}

        assert evaluator.evaluate(condition, context) is False

    def test_matches_operator_numeric_pattern(self):
        """Test MATCHES operator with numeric pattern."""
        evaluator = ConditionEvaluator()

        condition = {"field": "code", "operator": "MATCHES", "value": r"^\d{3}-\d{4}$"}
        context = {"code": "123-4567"}

        assert evaluator.evaluate(condition, context) is True

    def test_matches_operator_caching(self):
        """Test that regex patterns are cached."""
        evaluator = ConditionEvaluator()

        condition = {"field": "text", "operator": "MATCHES", "value": r"test\d+"}

        # Compile and cache
        compiled = evaluator._compile_condition(condition)

        assert "regex" in compiled
        assert compiled["regex"] is not None


class TestCompositeConditions:
    """Tests for composite conditions (AND, OR, NOT)."""

    def test_and_operator_all_true(self):
        """Test AND operator when all conditions are true."""
        evaluator = ConditionEvaluator()

        rule = {
            "conditions": {
                "type": "composite",
                "operator": "AND",
                "conditions": [
                    {"field": "score", "operator": ">", "value": 50},
                    {"field": "status", "operator": "==", "value": "active"},
                ],
            }
        }

        compiled = evaluator.compile_rule(rule)
        context = {"score": 75, "status": "active"}

        assert evaluator.evaluate_compiled(compiled, context) is True

    def test_and_operator_one_false(self):
        """Test AND operator when one condition is false."""
        evaluator = ConditionEvaluator()

        rule = {
            "conditions": {
                "type": "composite",
                "operator": "AND",
                "conditions": [
                    {"field": "score", "operator": ">", "value": 50},
                    {"field": "status", "operator": "==", "value": "active"},
                ],
            }
        }

        compiled = evaluator.compile_rule(rule)
        context = {"score": 25, "status": "active"}

        assert evaluator.evaluate_compiled(compiled, context) is False

    def test_or_operator_one_true(self):
        """Test OR operator when at least one condition is true."""
        evaluator = ConditionEvaluator()

        rule = {
            "conditions": {
                "type": "composite",
                "operator": "OR",
                "conditions": [
                    {"field": "priority", "operator": "==", "value": "high"},
                    {"field": "urgent", "operator": "==", "value": True},
                ],
            }
        }

        compiled = evaluator.compile_rule(rule)
        context = {"priority": "low", "urgent": True}

        assert evaluator.evaluate_compiled(compiled, context) is True

    def test_or_operator_all_false(self):
        """Test OR operator when all conditions are false."""
        evaluator = ConditionEvaluator()

        rule = {
            "conditions": {
                "type": "composite",
                "operator": "OR",
                "conditions": [
                    {"field": "a", "operator": "==", "value": 1},
                    {"field": "b", "operator": "==", "value": 2},
                ],
            }
        }

        compiled = evaluator.compile_rule(rule)
        context = {"a": 0, "b": 0}

        assert evaluator.evaluate_compiled(compiled, context) is False

    def test_not_operator(self):
        """Test NOT operator."""
        evaluator = ConditionEvaluator()

        rule = {
            "conditions": {
                "type": "composite",
                "operator": "NOT",
                "conditions": [
                    {"field": "disabled", "operator": "==", "value": True},
                ],
            }
        }

        compiled = evaluator.compile_rule(rule)
        context = {"disabled": False}

        assert evaluator.evaluate_compiled(compiled, context) is True

    def test_nested_composite_conditions(self):
        """Test deeply nested composite conditions."""
        evaluator = ConditionEvaluator()

        rule = {
            "conditions": {
                "type": "composite",
                "operator": "AND",
                "conditions": [
                    {"field": "enabled", "operator": "==", "value": True},
                    {
                        "type": "composite",
                        "operator": "OR",
                        "conditions": [
                            {"field": "priority", "operator": "==", "value": "high"},
                            {"field": "score", "operator": ">", "value": 80},
                        ],
                    },
                ],
            }
        }

        compiled = evaluator.compile_rule(rule)
        context = {"enabled": True, "priority": "low", "score": 85}

        assert evaluator.evaluate_compiled(compiled, context) is True


class TestAggregationOperators:
    """Tests for aggregation operators (ALL, ANY, NONE)."""

    def test_all_operator_true(self):
        """Test ALL operator when all elements match."""
        evaluator = ConditionEvaluator()

        rule = {
            "conditions": {
                "type": "aggregation",
                "operator": "ALL",
                "field": "scores",
                "compare_operator": "==",
                "compare_value": 100,
            }
        }

        compiled = evaluator.compile_rule(rule)
        context = {"scores": [100, 100, 100]}

        assert evaluator.evaluate_compiled(compiled, context) is True

    def test_all_operator_false(self):
        """Test ALL operator when not all elements match."""
        evaluator = ConditionEvaluator()

        rule = {
            "conditions": {
                "type": "aggregation",
                "operator": "ALL",
                "field": "scores",
                "compare_operator": "==",
                "compare_value": 100,
            }
        }

        compiled = evaluator.compile_rule(rule)
        context = {"scores": [100, 90, 100]}

        assert evaluator.evaluate_compiled(compiled, context) is False

    def test_any_operator_true(self):
        """Test ANY operator when at least one element matches."""
        evaluator = ConditionEvaluator()

        rule = {
            "conditions": {
                "type": "aggregation",
                "operator": "ANY",
                "field": "statuses",
                "compare_operator": "==",
                "compare_value": "error",
            }
        }

        compiled = evaluator.compile_rule(rule)
        context = {"statuses": ["ok", "error", "ok"]}

        assert evaluator.evaluate_compiled(compiled, context) is True

    def test_any_operator_false(self):
        """Test ANY operator when no elements match."""
        evaluator = ConditionEvaluator()

        rule = {
            "conditions": {
                "type": "aggregation",
                "operator": "ANY",
                "field": "statuses",
                "compare_operator": "==",
                "compare_value": "error",
            }
        }

        compiled = evaluator.compile_rule(rule)
        context = {"statuses": ["ok", "ok", "ok"]}

        assert evaluator.evaluate_compiled(compiled, context) is False

    def test_none_operator_true(self):
        """Test NONE operator when no elements match."""
        evaluator = ConditionEvaluator()

        rule = {
            "conditions": {
                "type": "aggregation",
                "operator": "NONE",
                "field": "errors",
                "compare_operator": "==",
                "compare_value": "critical",
            }
        }

        compiled = evaluator.compile_rule(rule)
        context = {"errors": ["minor", "warning", "info"]}

        assert evaluator.evaluate_compiled(compiled, context) is True

    def test_none_operator_false(self):
        """Test NONE operator when some elements match."""
        evaluator = ConditionEvaluator()

        rule = {
            "conditions": {
                "type": "aggregation",
                "operator": "NONE",
                "field": "errors",
                "compare_operator": "==",
                "compare_value": "critical",
            }
        }

        compiled = evaluator.compile_rule(rule)
        context = {"errors": ["minor", "critical", "info"]}

        assert evaluator.evaluate_compiled(compiled, context) is False


class TestMathOperators:
    """Tests for math operators (SUM, AVG, MIN, MAX, COUNT)."""

    def test_sum_operator(self):
        """Test SUM operator."""
        evaluator = ConditionEvaluator()

        rule = {
            "conditions": {
                "type": "math",
                "operator": "SUM",
                "field": "values",
                "compare_operator": "==",
                "compare_value": 15,
            }
        }

        compiled = evaluator.compile_rule(rule)
        context = {"values": [5, 4, 3, 2, 1]}

        assert evaluator.evaluate_compiled(compiled, context) is True

    def test_avg_operator(self):
        """Test AVG operator."""
        evaluator = ConditionEvaluator()

        rule = {
            "conditions": {
                "type": "math",
                "operator": "AVG",
                "field": "scores",
                "compare_operator": ">=",
                "compare_value": 80,
            }
        }

        compiled = evaluator.compile_rule(rule)
        context = {"scores": [90, 85, 75]}

        assert evaluator.evaluate_compiled(compiled, context) is True

    def test_min_operator(self):
        """Test MIN operator."""
        evaluator = ConditionEvaluator()

        rule = {
            "conditions": {
                "type": "math",
                "operator": "MIN",
                "field": "temperatures",
                "compare_operator": ">",
                "compare_value": 0,
            }
        }

        compiled = evaluator.compile_rule(rule)
        context = {"temperatures": [10, 5, 15, 3]}

        assert evaluator.evaluate_compiled(compiled, context) is True

    def test_max_operator(self):
        """Test MAX operator."""
        evaluator = ConditionEvaluator()

        rule = {
            "conditions": {
                "type": "math",
                "operator": "MAX",
                "field": "scores",
                "compare_operator": "<=",
                "compare_value": 100,
            }
        }

        compiled = evaluator.compile_rule(rule)
        context = {"scores": [95, 88, 92, 100]}

        assert evaluator.evaluate_compiled(compiled, context) is True

    def test_count_operator(self):
        """Test COUNT operator."""
        evaluator = ConditionEvaluator()

        rule = {
            "conditions": {
                "type": "math",
                "operator": "COUNT",
                "field": "items",
                "compare_operator": "==",
                "compare_value": 5,
            }
        }

        compiled = evaluator.compile_rule(rule)
        context = {"items": [1, 2, 3, 4, 5]}

        assert evaluator.evaluate_compiled(compiled, context) is True

    def test_avg_empty_list(self):
        """Test AVG operator with empty list."""
        evaluator = ConditionEvaluator()

        rule = {
            "conditions": {
                "type": "math",
                "operator": "AVG",
                "field": "values",
                "compare_operator": "==",
                "compare_value": 0,
            }
        }

        compiled = evaluator.compile_rule(rule)
        context = {"values": []}

        assert evaluator.evaluate_compiled(compiled, context) is False


class TestNestedFieldAccess:
    """Tests for nested field access with dot notation."""

    def test_nested_field_access(self):
        """Test accessing nested fields."""
        evaluator = ConditionEvaluator()

        condition = {"field": "user.profile.age", "operator": ">", "value": 18}
        context = {"user": {"profile": {"age": 25}}}

        assert evaluator.evaluate(condition, context) is True

    def test_deeply_nested_fields(self):
        """Test accessing deeply nested fields."""
        evaluator = ConditionEvaluator()

        condition = {"field": "a.b.c.d.e", "operator": "==", "value": "deep"}
        context = {"a": {"b": {"c": {"d": {"e": "deep"}}}}}

        assert evaluator.evaluate(condition, context) is True

    def test_missing_nested_field(self):
        """Test that missing nested field returns false."""
        evaluator = ConditionEvaluator()

        condition = {"field": "user.profile.missing", "operator": "==", "value": "value"}
        context = {"user": {"profile": {"age": 25}}}

        assert evaluator.evaluate(condition, context) is False


class TestRuleCompilation:
    """Tests for rule compilation and caching."""

    def test_compile_rule(self):
        """Test compiling a rule."""
        evaluator = ConditionEvaluator()

        rule = {
            "conditions": {
                "field": "score",
                "operator": ">",
                "value": 50,
            }
        }

        compiled = evaluator.compile_rule(rule)

        assert "compiled_conditions" in compiled
        assert compiled["compiled_conditions"] is not None

    def test_compile_rule_no_conditions(self):
        """Test compiling a rule with no conditions."""
        evaluator = ConditionEvaluator()

        rule = {}

        compiled = evaluator.compile_rule(rule)

        assert compiled["compiled_conditions"] is None

    def test_evaluate_compiled_no_conditions(self):
        """Test evaluating compiled rule with no conditions returns True."""
        evaluator = ConditionEvaluator()

        compiled = {"compiled_conditions": None}
        context = {"any": "data"}

        assert evaluator.evaluate_compiled(compiled, context) is True

    def test_operator_function_caching(self):
        """Test that operator functions are cached in compiled conditions."""
        evaluator = ConditionEvaluator()

        condition = {"field": "x", "operator": ">", "value": 5}

        compiled = evaluator._compile_condition(condition)

        assert "op_func" in compiled
        assert compiled["op_func"] is not None
