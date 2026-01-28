"""
High-performance condition evaluator for cascade routing decisions.

This module provides optimized condition evaluation with support for:
- Basic comparisons (==, !=, >, >=, <, <=)
- Set operations (IN, NOT_IN, CONTAINS)
- Existence checks (EXISTS, IS_NULL)
- Composite conditions (AND, OR, NOT)
- Aggregation operators (ALL, ANY, NONE)
- Math operators (SUM, AVG, MIN, MAX, COUNT)
- Pattern matching (MATCHES)
"""

import logging
import operator
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ConditionEvaluator:
    """
    High-performance condition evaluator with rule compilation and caching.

    This evaluator pre-compiles rules for fast evaluation, caches compiled expressions,
    and provides special handling for stage results and nested field access.

    Examples:
        >>> evaluator = ConditionEvaluator()
        >>> rule = {
        ...     "conditions": {
        ...         "type": "simple",
        ...         "field": "confidence",
        ...         "operator": ">",
        ...         "value": 0.8
        ...     }
        ... }
        >>> compiled = evaluator.compile_rule(rule)
        >>> context = {"confidence": 0.9}
        >>> evaluator.evaluate_compiled(compiled, context)
        True
    """

    def __init__(self):
        """Initialize the condition evaluator with operator mappings and caches."""
        # Basic comparison operators
        self.operators = {
            "==": operator.eq,
            "!=": operator.ne,
            ">": operator.gt,
            ">=": operator.ge,
            "<": operator.lt,
            "<=": operator.le,
        }

        # Regex cache for pattern matching
        self._regex_cache: Dict[str, re.Pattern] = {}

        logger.debug("ConditionEvaluator initialized")

    def compile_rule(self, rule: Dict[str, Any]) -> Dict[str, Any]:
        """
        Pre-compile a rule for fast evaluation.

        This method compiles conditions into an optimized representation that can be
        evaluated quickly. It pre-processes field paths, compiles regex patterns,
        and caches operator functions.

        Args:
            rule: Rule dictionary with 'conditions' key

        Returns:
            Compiled rule dictionary ready for evaluation

        Examples:
            >>> evaluator = ConditionEvaluator()
            >>> rule = {
            ...     "conditions": {
            ...         "type": "composite",
            ...         "operator": "AND",
            ...         "conditions": [
            ...             {"field": "score", "operator": ">", "value": 0.5},
            ...             {"field": "status", "operator": "==", "value": "ready"}
            ...         ]
            ...     }
            ... }
            >>> compiled = evaluator.compile_rule(rule)
        """
        if "conditions" not in rule:
            logger.warning("Rule has no conditions, returning empty compiled rule")
            return {"compiled_conditions": None}

        compiled_conditions = self._compile_condition(rule["conditions"])

        logger.debug(
            "Compiled rule",
            extra={"has_conditions": compiled_conditions is not None}
        )

        return {"compiled_conditions": compiled_conditions}

    def _compile_condition(self, condition: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Compile a single condition or composite condition.

        This method pre-processes conditions for fast evaluation:
        - Splits field paths into parts for nested access
        - Caches regex patterns for MATCHES operator
        - Stores operator functions for direct invocation

        Args:
            condition: Condition dictionary with type, field, operator, value, etc.

        Returns:
            Compiled condition dictionary or None if invalid
        """
        if not condition:
            return None

        condition_type = condition.get("type", "simple")

        if condition_type == "composite":
            # Compile composite conditions (AND/OR/NOT)
            composite_op = condition.get("operator", "AND")
            sub_conditions = condition.get("conditions", [])

            compiled_subs = [
                self._compile_condition(sub)
                for sub in sub_conditions
            ]
            compiled_subs = [c for c in compiled_subs if c is not None]

            return {
                "type": "composite",
                "operator": composite_op,
                "conditions": compiled_subs
            }

        elif condition_type == "aggregation":
            # Compile aggregation conditions (ALL/ANY/NONE)
            return {
                "type": "aggregation",
                "operator": condition["operator"],
                "field": condition["field"],
                "field_parts": condition["field"].split("."),
                "compare_operator": condition["compare_operator"],
                "compare_value": condition["compare_value"]
            }

        elif condition_type == "math":
            # Compile math conditions (SUM/AVG/MIN/MAX/COUNT)
            return {
                "type": "math",
                "operator": condition["operator"],
                "field": condition["field"],
                "field_parts": condition["field"].split("."),
                "compare_operator": condition["compare_operator"],
                "compare_value": condition["compare_value"]
            }

        else:
            # Compile simple condition
            field = condition.get("field", "")
            op = condition.get("operator", "==")
            value = condition.get("value")

            compiled = {
                "type": "simple",
                "field": field,
                "field_parts": field.split(".") if field else [],
                "operator": op,
                "value": value
            }

            # Cache operator function for basic comparisons
            if op in self.operators:
                compiled["op_func"] = self.operators[op]

            # Pre-compile regex for MATCHES operator
            if op == "MATCHES" and isinstance(value, str):
                if value not in self._regex_cache:
                    try:
                        self._regex_cache[value] = re.compile(value)
                    except re.error as e:
                        logger.error(
                            "Invalid regex pattern",
                            extra={"pattern": value, "error": str(e)}
                        )
                        return None
                compiled["regex"] = self._regex_cache[value]

            return compiled

    def evaluate(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """
        Evaluate a condition against a context without pre-compilation.

        This is a convenience method for one-time evaluations. For repeated
        evaluations, use compile_rule() and evaluate_compiled() for better performance.

        Args:
            condition: Condition dictionary to evaluate
            context: Context dictionary with field values

        Returns:
            True if condition matches, False otherwise

        Examples:
            >>> evaluator = ConditionEvaluator()
            >>> condition = {"field": "score", "operator": ">", "value": 0.5}
            >>> context = {"score": 0.8}
            >>> evaluator.evaluate(condition, context)
            True
        """
        compiled = self._compile_condition(condition)
        if compiled is None:
            logger.warning("Failed to compile condition")
            return False

        return self._evaluate_compiled(compiled, context)

    def evaluate_compiled(self, compiled_rule: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """
        Evaluate a pre-compiled rule against a context.

        This is the fast path for evaluation. The rule should be compiled once
        using compile_rule() and then evaluated many times with different contexts.

        Args:
            compiled_rule: Compiled rule from compile_rule()
            context: Context dictionary with field values

        Returns:
            True if all conditions match, False otherwise

        Examples:
            >>> evaluator = ConditionEvaluator()
            >>> rule = {"conditions": {"field": "x", "operator": ">", "value": 5}}
            >>> compiled = evaluator.compile_rule(rule)
            >>> evaluator.evaluate_compiled(compiled, {"x": 10})
            True
        """
        compiled_conditions = compiled_rule.get("compiled_conditions")

        if compiled_conditions is None:
            logger.debug("No compiled conditions, returning True")
            return True

        try:
            result = self._evaluate_compiled(compiled_conditions, context)
            logger.debug(
                "Evaluated compiled rule",
                extra={"result": result}
            )
            return result
        except Exception as e:
            logger.error(
                "Error evaluating compiled rule",
                extra={"error": str(e)},
                exc_info=True
            )
            return False

    def _evaluate_compiled(self, compiled: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """
        Internal method to evaluate a compiled condition.

        Args:
            compiled: Compiled condition dictionary
            context: Context dictionary with field values

        Returns:
            True if condition matches, False otherwise
        """
        condition_type = compiled.get("type", "simple")

        if condition_type == "composite":
            return self._evaluate_composite(compiled, context)

        elif condition_type == "aggregation":
            op = compiled["operator"]
            field_parts = compiled["field_parts"]
            compare_op = compiled["compare_operator"]
            compare_value = compiled["compare_value"]

            field_value = self._get_field_value(field_parts, context)
            return self._evaluate_aggregation(op, field_value, compare_value, context, compare_op)

        elif condition_type == "math":
            op = compiled["operator"]
            field_parts = compiled["field_parts"]
            compare_op = compiled["compare_operator"]
            compare_value = compiled["compare_value"]

            field_value = self._get_field_value(field_parts, context)
            return self._evaluate_math(op, field_value, compare_value, compare_op)

        else:
            # Simple condition
            field_parts = compiled["field_parts"]
            op = compiled["operator"]
            compare_value = compiled["value"]

            field_value = self._get_field_value(field_parts, context)

            # Special operators
            if op == "EXISTS":
                return field_value is not None

            elif op == "IS_NULL":
                return field_value is None

            elif op == "IN":
                if not isinstance(compare_value, (list, tuple, set)):
                    return False
                return field_value in compare_value

            elif op == "NOT_IN":
                if not isinstance(compare_value, (list, tuple, set)):
                    return True
                return field_value not in compare_value

            elif op == "CONTAINS":
                if isinstance(field_value, (list, tuple, set)):
                    return compare_value in field_value
                elif isinstance(field_value, str):
                    return str(compare_value) in field_value
                elif isinstance(field_value, dict):
                    return compare_value in field_value
                return False

            elif op == "MATCHES":
                if "regex" not in compiled:
                    return False
                if not isinstance(field_value, str):
                    field_value = str(field_value)
                return bool(compiled["regex"].match(field_value))

            # Basic comparison operators
            elif "op_func" in compiled:
                try:
                    return compiled["op_func"](field_value, compare_value)
                except (TypeError, ValueError):
                    logger.warning(
                        "Type error in comparison",
                        extra={
                            "field_value": field_value,
                            "compare_value": compare_value,
                            "operator": op
                        }
                    )
                    return False

            else:
                logger.warning(
                    "Unknown operator",
                    extra={"operator": op}
                )
                return False

    def _evaluate_composite(self, compiled: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """
        Evaluate composite conditions with short-circuit logic.

        This method implements efficient short-circuit evaluation:
        - AND: stops at first False
        - OR: stops at first True
        - NOT: negates single condition

        Args:
            compiled: Compiled composite condition
            context: Context dictionary

        Returns:
            True if composite condition matches, False otherwise
        """
        composite_op = compiled["operator"]
        sub_conditions = compiled["conditions"]

        if composite_op == "AND":
            # Short-circuit: stop at first False
            for sub in sub_conditions:
                if not self._evaluate_compiled(sub, context):
                    return False
            return True

        elif composite_op == "OR":
            # Short-circuit: stop at first True
            for sub in sub_conditions:
                if self._evaluate_compiled(sub, context):
                    return True
            return False

        elif composite_op == "NOT":
            # Negate single condition
            if sub_conditions:
                return not self._evaluate_compiled(sub_conditions[0], context)
            return True

        else:
            logger.warning(
                "Unknown composite operator",
                extra={"operator": composite_op}
            )
            return False

    def _get_field_value(self, field_parts: List[str], context: Dict[str, Any]) -> Any:
        """
        Extract field value from context with nested field support.

        This method handles:
        - Nested field access (e.g., "user.profile.name")
        - Special handling for stage_results
        - Safe navigation returning None for missing fields

        Args:
            field_parts: List of field path parts (e.g., ["user", "profile", "name"])
            context: Context dictionary

        Returns:
            Field value or None if not found
        """
        if not field_parts:
            return None

        value = context

        for part in field_parts:
            if value is None:
                return None

            if isinstance(value, dict):
                # Special handling for stage_results
                if part == "stage_results" and "stage_results" in value:
                    value = value["stage_results"]
                elif part in value:
                    value = value[part]
                else:
                    return None
            else:
                # Try attribute access for objects
                try:
                    value = getattr(value, part)
                except AttributeError:
                    return None

        return value

    def _evaluate_aggregation(
        self,
        op: str,
        field_value: Any,
        compare_value: Any,
        context: Dict[str, Any],
        compare_op: str = "=="
    ) -> bool:
        """
        Evaluate aggregation operators (ALL, ANY, NONE).

        These operators work on collections and check if all, any, or no elements
        match the comparison condition.

        Args:
            op: Aggregation operator (ALL, ANY, NONE)
            field_value: Collection to evaluate
            compare_value: Value to compare against
            context: Context dictionary (unused)
            compare_op: Comparison operator for element checks

        Returns:
            True if aggregation condition matches, False otherwise
        """
        if not isinstance(field_value, (list, tuple, set)):
            logger.warning(
                "Aggregation operator requires collection",
                extra={"operator": op, "field_type": type(field_value).__name__}
            )
            return False

        if op == "ALL":
            return all(self._compare_value(v, compare_value) for v in field_value)

        elif op == "ANY":
            return any(self._compare_value(v, compare_value) for v in field_value)

        elif op == "NONE":
            return not any(self._compare_value(v, compare_value) for v in field_value)

        else:
            logger.warning(
                "Unknown aggregation operator",
                extra={"operator": op}
            )
            return False

    def _evaluate_math(
        self,
        op: str,
        field_value: Any,
        compare_value: Any,
        compare_op: str = "=="
    ) -> bool:
        """
        Evaluate math operators (SUM, AVG, MIN, MAX, COUNT).

        These operators compute aggregate values from collections and compare
        them against a threshold.

        Args:
            op: Math operator (SUM, AVG, MIN, MAX, COUNT)
            field_value: Collection to compute on
            compare_value: Threshold to compare against
            compare_op: Comparison operator

        Returns:
            True if math condition matches, False otherwise
        """
        if not isinstance(field_value, (list, tuple, set)):
            logger.warning(
                "Math operator requires collection",
                extra={"operator": op, "field_type": type(field_value).__name__}
            )
            return False

        try:
            if op == "SUM":
                result = sum(field_value)

            elif op == "AVG":
                if not field_value:
                    return False
                result = sum(field_value) / len(field_value)

            elif op == "MIN":
                if not field_value:
                    return False
                result = min(field_value)

            elif op == "MAX":
                if not field_value:
                    return False
                result = max(field_value)

            elif op == "COUNT":
                result = len(field_value)

            else:
                logger.warning(
                    "Unknown math operator",
                    extra={"operator": op}
                )
                return False

            # Compare result with threshold
            if compare_op in self.operators:
                return self.operators[compare_op](result, compare_value)
            else:
                logger.warning(
                    "Unknown comparison operator",
                    extra={"operator": compare_op}
                )
                return False

        except (TypeError, ValueError) as e:
            logger.error(
                "Error in math evaluation",
                extra={"operator": op, "error": str(e)}
            )
            return False

    def _compare_value(self, value: Any, compare: Any) -> bool:
        """
        Simple comparison helper for equality checks.

        Args:
            value: Value to compare
            compare: Value to compare against

        Returns:
            True if values are equal, False otherwise
        """
        return value == compare


@lru_cache(maxsize=128)
def compile_expression(expression: str) -> Optional[re.Pattern]:
    """
    Compile and cache a regex expression.

    This function provides LRU caching for regex pattern compilation to avoid
    recompiling the same patterns repeatedly.

    Args:
        expression: Regex pattern string

    Returns:
        Compiled regex pattern or None if invalid
    """
    try:
        return re.compile(expression)
    except re.error as e:
        logger.error(
            "Failed to compile regex expression",
            extra={"expression": expression, "error": str(e)}
        )
        return None
