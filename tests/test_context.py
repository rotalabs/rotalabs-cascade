"""Tests for execution context in rotalabs-cascade.

Tests cover:
- StageResult creation and serialization
- ExecutionContext initialization
- Get/set with dot notation paths
- Stage result management
- Stage enabling/disabling
- Termination flags
- Final result structure
"""

import pytest
import time

from rotalabs_cascade.core.context import ExecutionContext, StageResult


class TestStageResult:
    """Tests for StageResult class."""

    def test_stage_result_creation(self):
        """Test creating a basic stage result."""
        result = StageResult(
            stage_name="TEST_STAGE",
            result="success",
            confidence=0.95,
            time_ms=123.45,
        )

        assert result.stage_name == "TEST_STAGE"
        assert result.result == "success"
        assert result.confidence == 0.95
        assert result.time_ms == 123.45
        assert result.error is None
        assert result.data == {}

    def test_stage_result_with_data(self):
        """Test stage result with additional data."""
        result = StageResult(
            stage_name="DATA_STAGE",
            result="processed",
            data={"key": "value", "count": 42},
        )

        assert result.data["key"] == "value"
        assert result.data["count"] == 42

    def test_stage_result_with_error(self):
        """Test stage result with error."""
        result = StageResult(
            stage_name="FAILED_STAGE",
            error="Connection timeout",
            time_ms=5000.0,
        )

        assert result.error == "Connection timeout"
        assert result.result is None

    def test_stage_result_to_dict(self):
        """Test converting stage result to dictionary."""
        result = StageResult(
            stage_name="TEST",
            result="done",
            confidence=0.8,
            data={"x": 1},
            time_ms=100.0,
        )

        data = result.to_dict()

        assert data["stage_name"] == "TEST"
        assert data["result"] == "done"
        assert data["confidence"] == 0.8
        assert data["data"] == {"x": 1}
        assert data["time_ms"] == 100.0

    def test_stage_result_to_dict_with_error(self):
        """Test stage result with error serializes correctly."""
        result = StageResult(
            stage_name="FAILED",
            error="Something went wrong",
            time_ms=50.0,
        )

        data = result.to_dict()

        assert "error" in data
        assert data["error"] == "Something went wrong"

    def test_stage_result_repr(self):
        """Test string representation of stage result."""
        result = StageResult(
            stage_name="TEST",
            result="success",
        )

        repr_str = repr(result)

        assert "TEST" in repr_str
        assert "success" in repr_str


class TestExecutionContext:
    """Tests for ExecutionContext class."""

    def test_context_initialization(self, sample_data):
        """Test creating an execution context."""
        context = ExecutionContext(sample_data)

        assert context.data is sample_data
        assert len(context._stage_results) == 0
        assert len(context._errors) == 0
        assert context.should_terminate is False

    def test_context_data_is_reference(self, sample_data):
        """Test that context stores data by reference, not copy."""
        context = ExecutionContext(sample_data)

        # Modify original data
        sample_data["new_key"] = "new_value"

        # Should be reflected in context
        assert context.data["new_key"] == "new_value"

    def test_elapsed_ms(self, sample_data):
        """Test elapsed time calculation."""
        context = ExecutionContext(sample_data)

        time.sleep(0.1)  # Sleep 100ms

        elapsed = context.elapsed_ms
        assert elapsed >= 100  # At least 100ms elapsed
        assert elapsed < 200  # But not too much more

    def test_get_simple_field(self, sample_data):
        """Test getting a simple top-level field."""
        context = ExecutionContext(sample_data)

        assert context.get("user_id") == "user123"
        assert context.get("confidence") == 0.9

    def test_get_nested_field(self, sample_data):
        """Test getting nested field with dot notation."""
        context = ExecutionContext(sample_data)

        assert context.get("request.text") == "Sample request text"
        assert context.get("request.priority") == "high"
        assert context.get("request.metadata.source") == "api"

    def test_get_nonexistent_field(self, sample_data):
        """Test getting non-existent field returns default."""
        context = ExecutionContext(sample_data)

        assert context.get("nonexistent") is None
        assert context.get("nonexistent", "default") == "default"

    def test_get_nonexistent_nested_field(self, sample_data):
        """Test getting non-existent nested field returns default."""
        context = ExecutionContext(sample_data)

        assert context.get("request.missing.field") is None
        assert context.get("request.missing.field", "default") == "default"

    def test_get_caching(self, sample_data):
        """Test that get() results are cached."""
        context = ExecutionContext(sample_data)

        # First access
        value1 = context.get("request.text")

        # Modify underlying data
        sample_data["request"]["text"] = "Modified"

        # Second access should return cached value
        value2 = context.get("request.text")

        assert value1 == value2 == "Sample request text"

    def test_set_simple_field(self, sample_data):
        """Test setting a simple field."""
        context = ExecutionContext(sample_data)

        context.set("new_field", "new_value")

        assert context.data["new_field"] == "new_value"

    def test_set_nested_field(self, sample_data):
        """Test setting nested field with dot notation."""
        context = ExecutionContext(sample_data)

        context.set("result.status", "completed")

        assert context.data["result"]["status"] == "completed"

    def test_set_creates_intermediate_dicts(self, sample_data):
        """Test that set creates intermediate dictionaries."""
        context = ExecutionContext(sample_data)

        context.set("deeply.nested.value", 42)

        assert context.data["deeply"]["nested"]["value"] == 42

    def test_set_invalidates_cache(self, sample_data):
        """Test that set invalidates the path cache."""
        context = ExecutionContext(sample_data)

        # Cache the value
        old_value = context.get("confidence")
        assert old_value == 0.9

        # Set new value
        context.set("confidence", 0.5)

        # Get should return new value
        new_value = context.get("confidence")
        assert new_value == 0.5

    def test_set_on_non_dict_fails(self, sample_data):
        """Test that setting on non-dict raises error."""
        context = ExecutionContext(sample_data)

        with pytest.raises(ValueError, match="is not a dictionary"):
            context.set("confidence.subfield", "value")

    def test_add_stage_result(self, sample_data):
        """Test adding a stage result."""
        context = ExecutionContext(sample_data)

        result = StageResult(
            stage_name="TEST",
            result="success",
            time_ms=100.0,
        )

        context.add_stage_result(result)

        assert len(context._stage_results) == 1
        assert context._stage_results[0].stage_name == "TEST"

    def test_add_multiple_stage_results(self, sample_data):
        """Test adding multiple stage results."""
        context = ExecutionContext(sample_data)

        for i in range(3):
            result = StageResult(
                stage_name=f"STAGE_{i}",
                result=f"result_{i}",
            )
            context.add_stage_result(result)

        assert len(context._stage_results) == 3

    def test_get_stage_result(self, sample_data):
        """Test retrieving a specific stage result."""
        context = ExecutionContext(sample_data)

        result1 = StageResult(stage_name="STAGE_A", result="a")
        result2 = StageResult(stage_name="STAGE_B", result="b")

        context.add_stage_result(result1)
        context.add_stage_result(result2)

        retrieved = context.get_stage_result("STAGE_B")

        assert retrieved is not None
        assert retrieved.stage_name == "STAGE_B"
        assert retrieved.result == "b"

    def test_get_stage_result_returns_latest(self, sample_data):
        """Test that get_stage_result returns most recent result."""
        context = ExecutionContext(sample_data)

        # Add same stage twice
        context.add_stage_result(StageResult(stage_name="STAGE", result="first"))
        context.add_stage_result(StageResult(stage_name="STAGE", result="second"))

        retrieved = context.get_stage_result("STAGE")

        assert retrieved.result == "second"

    def test_get_stage_result_not_found(self, sample_data):
        """Test getting non-existent stage result returns None."""
        context = ExecutionContext(sample_data)

        result = context.get_stage_result("NONEXISTENT")

        assert result is None

    def test_add_stage_error(self, sample_data):
        """Test adding a stage error."""
        context = ExecutionContext(sample_data)

        context.add_stage_error("FAILED_STAGE", "Connection timeout")

        assert len(context._errors) == 1
        assert "FAILED_STAGE" in context._errors[0]
        assert "Connection timeout" in context._errors[0]

    def test_enable_stage(self, sample_data):
        """Test enabling a stage."""
        context = ExecutionContext(sample_data)

        context.enable_stage("STAGE_A")

        assert context.is_stage_enabled("STAGE_A")

    def test_disable_stage(self, sample_data):
        """Test disabling a stage."""
        context = ExecutionContext(sample_data)

        context.disable_stage("STAGE_B")

        assert not context.is_stage_enabled("STAGE_B")

    def test_enable_overrides_disable(self, sample_data):
        """Test that enabling a stage removes it from disabled set."""
        context = ExecutionContext(sample_data)

        context.disable_stage("STAGE")
        context.enable_stage("STAGE")

        assert context.is_stage_enabled("STAGE")

    def test_disable_overrides_enable(self, sample_data):
        """Test that disabling a stage removes it from enabled set."""
        context = ExecutionContext(sample_data)

        context.enable_stage("STAGE")
        context.disable_stage("STAGE")

        assert not context.is_stage_enabled("STAGE")

    def test_is_stage_enabled_default(self, sample_data):
        """Test that stages are enabled by default."""
        context = ExecutionContext(sample_data)

        assert context.is_stage_enabled("UNTRACKED_STAGE", default=True)

    def test_is_stage_enabled_custom_default(self, sample_data):
        """Test custom default for stage enabled check."""
        context = ExecutionContext(sample_data)

        assert not context.is_stage_enabled("UNTRACKED_STAGE", default=False)

    def test_set_termination_flag(self, sample_data):
        """Test setting termination flag."""
        context = ExecutionContext(sample_data)

        assert not context.should_terminate

        context.set_termination_flag("User requested abort")

        assert context.should_terminate

    def test_set_next_stage(self, sample_data):
        """Test setting next stage override."""
        context = ExecutionContext(sample_data)

        context.set_next_stage("FINAL_STAGE")

        assert context._next_stage == "FINAL_STAGE"

    def test_get_next_stage_clears(self, sample_data):
        """Test that get_next_stage clears the override."""
        context = ExecutionContext(sample_data)

        context.set_next_stage("FINAL")
        stage = context.get_next_stage()

        assert stage == "FINAL"
        assert context.get_next_stage() is None  # Cleared

    def test_metadata_operations(self, sample_data):
        """Test setting and getting metadata."""
        context = ExecutionContext(sample_data)

        context.set_metadata("key", "value")
        context.set_metadata("count", 42)

        assert context.get_metadata("key") == "value"
        assert context.get_metadata("count") == 42
        assert context.get_metadata("missing") is None

    def test_metadata_default_value(self, sample_data):
        """Test metadata get with default value."""
        context = ExecutionContext(sample_data)

        assert context.get_metadata("missing", "default") == "default"

    def test_get_result_success(self, sample_data):
        """Test getting final result for successful execution."""
        context = ExecutionContext(sample_data)

        result1 = StageResult(stage_name="STAGE_A", result="a", time_ms=100)
        result2 = StageResult(stage_name="STAGE_B", result="b", time_ms=200)

        context.add_stage_result(result1)
        context.add_stage_result(result2)

        final_result = context.get_result()

        assert final_result["success"] is True
        assert final_result["stages_executed"] == 2
        assert "STAGE_A" in final_result["stage_results"]
        assert "STAGE_B" in final_result["stage_results"]

    def test_get_result_with_errors(self, sample_data):
        """Test getting final result with errors."""
        context = ExecutionContext(sample_data)

        context.add_stage_error("STAGE_A", "Error occurred")

        final_result = context.get_result()

        assert final_result["success"] is False
        assert "errors" in final_result
        assert len(final_result["errors"]) == 1

    def test_get_result_with_metadata(self, sample_data):
        """Test that get_result includes metadata."""
        context = ExecutionContext(sample_data)

        context.set_metadata("custom_key", "custom_value")

        final_result = context.get_result()

        assert "metadata" in final_result
        assert final_result["metadata"]["custom_key"] == "custom_value"

    def test_get_result_includes_timeline(self, sample_data):
        """Test that get_result includes timeline."""
        context = ExecutionContext(sample_data)

        context.add_stage_result(StageResult(stage_name="STAGE", result="done"))

        final_result = context.get_result()

        assert "timeline" in final_result
        assert len(final_result["timeline"]) > 0

    def test_get_result_includes_routing_decisions(self, sample_data):
        """Test that get_result includes routing decisions."""
        context = ExecutionContext(sample_data)

        context.enable_stage("STAGE_A")
        context.set_next_stage("STAGE_B")

        final_result = context.get_result()

        assert "routing_decisions" in final_result
        assert len(final_result["routing_decisions"]) == 2

    def test_context_repr(self, sample_data):
        """Test string representation of execution context."""
        context = ExecutionContext(sample_data)

        context.add_stage_result(StageResult(stage_name="TEST", result="done"))

        repr_str = repr(context)

        assert "ExecutionContext" in repr_str
        assert "stages_executed=1" in repr_str
        assert "errors=0" in repr_str
