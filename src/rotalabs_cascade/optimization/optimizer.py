"""
Execution plan optimizer for cascade routing.

This module creates optimized execution plans by:
- Building dependency graphs from stage configurations
- Identifying stages that can run in parallel
- Creating initial plans with only starting stages (dynamic routing decides rest)
- Filtering stages based on input data properties
- Analyzing critical paths for performance optimization
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class ExecutionOptimizer:
    """
    Creates optimized execution plans for cascade routing.

    This optimizer analyzes stage dependencies and input data to create efficient
    execution plans. It identifies parallel execution opportunities and filters
    stages based on data properties.

    Key features:
    - Dependency graph construction
    - Parallel stage grouping
    - Data-driven stage filtering
    - Critical path analysis

    Examples:
        >>> config = {
        ...     "stages": {
        ...         "stage1": {"depends_on": []},
        ...         "stage2": {"depends_on": ["stage1"]},
        ...         "stage3": {"depends_on": ["stage1"]}
        ...     }
        ... }
        >>> optimizer = ExecutionOptimizer(config)
        >>> plan = optimizer.create_execution_plan({}, config)
        >>> len(plan)
        1  # Only starting stages
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the execution optimizer.

        Args:
            config: Cascade configuration with stages and dependencies
        """
        self.config = config
        self.stages = config.get("stages", {})

        # Build dependency graph
        self.dependency_graph = self._build_dependency_graph()

        # Identify parallel execution groups
        self.parallel_groups = self._identify_parallel_groups()

        logger.info(
            "ExecutionOptimizer initialized",
            extra={
                "num_stages": len(self.stages),
                "num_parallel_groups": len(self.parallel_groups)
            }
        )

    def _build_dependency_graph(self) -> Dict[str, Dict[str, Any]]:
        """
        Build a dependency graph from stage configurations.

        This method analyzes stage dependencies to create a graph structure
        showing which stages depend on which other stages.

        Returns:
            Dictionary mapping stage names to dependency information:
            {
                "stage_name": {
                    "depends_on": ["stage1", "stage2"],
                    "dependents": ["stage3", "stage4"]
                }
            }
        """
        graph = {}

        # Initialize graph nodes
        for stage_name in self.stages:
            graph[stage_name] = {
                "depends_on": [],
                "dependents": []
            }

        # Build edges
        for stage_name, stage_config in self.stages.items():
            depends_on = stage_config.get("depends_on", [])

            if isinstance(depends_on, str):
                depends_on = [depends_on]

            graph[stage_name]["depends_on"] = depends_on

            # Add reverse edges (dependents)
            for dep in depends_on:
                if dep in graph:
                    graph[dep]["dependents"].append(stage_name)
                else:
                    logger.warning(
                        "Stage depends on unknown stage",
                        extra={"stage": stage_name, "dependency": dep}
                    )

        logger.debug(
            "Built dependency graph",
            extra={"num_nodes": len(graph)}
        )

        return graph

    def _identify_parallel_groups(self) -> Dict[int, List[str]]:
        """
        Identify groups of stages that can run in parallel.

        Stages can run in parallel if they:
        - Have no dependencies between them
        - Have the same dependency depth
        - Are not connected through any dependency chain

        Returns:
            Dictionary mapping depth levels to lists of parallel stage names:
            {
                0: ["stage1", "stage2"],  # No dependencies
                1: ["stage3", "stage4"],  # Both depend on stage1
                2: ["stage5"]             # Depends on stage3
            }
        """
        parallel_groups = {}

        # Calculate depth for each stage
        depths = {}
        visited = set()

        def calculate_depth(stage_name: str) -> int:
            """Calculate the maximum depth from starting stages."""
            if stage_name in visited:
                # Already calculated
                return depths.get(stage_name, 0)

            visited.add(stage_name)

            depends_on = self.dependency_graph.get(stage_name, {}).get("depends_on", [])

            if not depends_on:
                # Starting stage
                depths[stage_name] = 0
                return 0

            # Depth is 1 + max depth of dependencies
            max_dep_depth = max(calculate_depth(dep) for dep in depends_on)
            depth = max_dep_depth + 1
            depths[stage_name] = depth

            return depth

        # Calculate depths for all stages
        for stage_name in self.stages:
            calculate_depth(stage_name)

        # Group stages by depth
        for stage_name, depth in depths.items():
            if depth not in parallel_groups:
                parallel_groups[depth] = []
            parallel_groups[depth].append(stage_name)

        logger.debug(
            "Identified parallel groups",
            extra={
                "num_levels": len(parallel_groups),
                "groups": {k: len(v) for k, v in parallel_groups.items()}
            }
        )

        return parallel_groups

    def create_execution_plan(
        self,
        data: Dict[str, Any],
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Create an optimized execution plan for the given data.

        IMPORTANT: This only returns starting stages (stages with no dependencies).
        Dynamic routing will decide which stages to execute next based on runtime results.

        Args:
            data: Input data for the cascade
            config: Cascade configuration

        Returns:
            List of starting stage configurations to execute first

        Examples:
            >>> optimizer = ExecutionOptimizer(config)
            >>> plan = optimizer.create_execution_plan({"type": "question"}, config)
            >>> [s["name"] for s in plan]
            ['initial_classifier', 'guard_check']
        """
        stages_config = config.get("stages", {})

        # Filter stages based on data properties
        filtered_stages = self._filter_stages_by_data(stages_config, data)

        # Get starting stages (no dependencies)
        starting_stages = []

        for stage_name, stage_config in filtered_stages.items():
            depends_on = stage_config.get("depends_on", [])

            if isinstance(depends_on, str):
                depends_on = [depends_on]

            if not depends_on:
                # This is a starting stage
                stage_plan = {
                    "name": stage_name,
                    "config": stage_config,
                    "parallel_eligible": True,
                    "estimated_cost": stage_config.get("cost_estimate", 1.0)
                }
                starting_stages.append(stage_plan)

        logger.info(
            "Created execution plan",
            extra={
                "num_filtered_stages": len(filtered_stages),
                "num_starting_stages": len(starting_stages)
            }
        )

        return starting_stages

    def _filter_stages_by_data(
        self,
        stages: Dict[str, Any],
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Filter stages based on input data properties.

        This method checks custom_properties in stage configurations to determine
        if stages should be included based on:
        - enable_conditions: Conditions that must be met
        - domains: Required domain values
        - min_value/max_value: Numeric range checks

        Args:
            stages: Stage configurations
            data: Input data

        Returns:
            Filtered dictionary of stages that match data properties
        """
        filtered = {}

        for stage_name, stage_config in stages.items():
            custom_props = stage_config.get("custom_properties", {})

            # Check enable conditions
            enable_conditions = custom_props.get("enable_conditions", {})
            if enable_conditions:
                if not self._evaluate_simple_conditions(enable_conditions, data):
                    logger.debug(
                        "Stage filtered out by enable conditions",
                        extra={"stage": stage_name}
                    )
                    continue

            # Check domain constraints
            required_domains = custom_props.get("domains")
            if required_domains:
                data_domain = data.get("domain")
                if isinstance(required_domains, str):
                    required_domains = [required_domains]
                if data_domain not in required_domains:
                    logger.debug(
                        "Stage filtered out by domain constraint",
                        extra={"stage": stage_name, "required": required_domains, "actual": data_domain}
                    )
                    continue

            # Check numeric range constraints
            min_value = custom_props.get("min_value")
            max_value = custom_props.get("max_value")

            if min_value is not None or max_value is not None:
                # Check if data has a numeric field to validate
                value_field = custom_props.get("value_field", "value")
                value = data.get(value_field)

                if value is not None:
                    try:
                        value = float(value)
                        if min_value is not None and value < min_value:
                            logger.debug(
                                "Stage filtered out by min_value",
                                extra={"stage": stage_name, "value": value, "min": min_value}
                            )
                            continue
                        if max_value is not None and value > max_value:
                            logger.debug(
                                "Stage filtered out by max_value",
                                extra={"stage": stage_name, "value": value, "max": max_value}
                            )
                            continue
                    except (TypeError, ValueError):
                        logger.warning(
                            "Could not convert value to float for range check",
                            extra={"stage": stage_name, "value": value}
                        )

            # Stage passed all filters
            filtered[stage_name] = stage_config

        return filtered

    def _topological_sort(self, stages: Dict[str, Any]) -> List[str]:
        """
        Perform topological sort on stages for dependency resolution.

        This method orders stages so that dependencies are executed before
        dependent stages. It uses Kahn's algorithm for topological sorting.

        Args:
            stages: Dictionary of stage configurations

        Returns:
            List of stage names in topological order

        Raises:
            ValueError: If circular dependencies are detected
        """
        # Build in-degree map
        in_degree = {}
        for stage_name in stages:
            in_degree[stage_name] = 0

        for stage_name, stage_config in stages.items():
            depends_on = stage_config.get("depends_on", [])
            if isinstance(depends_on, str):
                depends_on = [depends_on]

            for dep in depends_on:
                if dep in in_degree:
                    in_degree[stage_name] += 1

        # Find starting nodes (in-degree 0)
        queue = [name for name, degree in in_degree.items() if degree == 0]
        sorted_stages = []

        while queue:
            # Process node with no dependencies
            current = queue.pop(0)
            sorted_stages.append(current)

            # Reduce in-degree for dependents
            dependents = self.dependency_graph.get(current, {}).get("dependents", [])
            for dependent in dependents:
                if dependent in in_degree:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

        # Check for circular dependencies
        if len(sorted_stages) != len(stages):
            remaining = [name for name in stages if name not in sorted_stages]
            logger.error(
                "Circular dependency detected",
                extra={"remaining_stages": remaining}
            )
            raise ValueError(f"Circular dependency detected in stages: {remaining}")

        return sorted_stages

    def _can_run_parallel(
        self,
        stage_name: str,
        processed: Set[str],
        stage_config: Dict[str, Any]
    ) -> bool:
        """
        Check if a stage can run in parallel with already processed stages.

        A stage can run in parallel if:
        - All its dependencies are already processed
        - It has no custom properties preventing parallel execution

        Args:
            stage_name: Name of stage to check
            processed: Set of already processed stage names
            stage_config: Configuration of the stage

        Returns:
            True if stage can run in parallel, False otherwise
        """
        # Check if all dependencies are processed
        depends_on = stage_config.get("depends_on", [])
        if isinstance(depends_on, str):
            depends_on = [depends_on]

        for dep in depends_on:
            if dep not in processed:
                return False

        # Check custom properties for parallel execution flags
        custom_props = stage_config.get("custom_properties", {})
        allow_parallel = custom_props.get("allow_parallel", True)

        return allow_parallel

    def _evaluate_simple_conditions(
        self,
        conditions: Dict[str, Any],
        data: Dict[str, Any]
    ) -> bool:
        """
        Evaluate simple conditions for stage filtering.

        This is a lightweight condition evaluator for basic checks during
        plan creation. For full condition evaluation, use ConditionEvaluator.

        Args:
            conditions: Dictionary of field: value pairs to check
            data: Input data

        Returns:
            True if all conditions match, False otherwise

        Examples:
            >>> conditions = {"type": "question", "priority": "high"}
            >>> data = {"type": "question", "priority": "high"}
            >>> optimizer._evaluate_simple_conditions(conditions, data)
            True
        """
        for field, expected_value in conditions.items():
            # Handle nested field access
            field_parts = field.split(".")
            value = data

            for part in field_parts:
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    value = None
                    break

            # Simple equality check
            if value != expected_value:
                return False

        return True

    def analyze_critical_path(self, plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the critical path in an execution plan.

        The critical path is the longest sequence of dependent stages,
        which determines the minimum execution time for the cascade.

        Args:
            plan: List of stage configurations in execution order

        Returns:
            Dictionary with critical path analysis:
            {
                "critical_path": ["stage1", "stage2", "stage3"],
                "total_cost": 15.5,
                "parallel_opportunities": 3,
                "optimization_suggestions": ["suggestion1", "suggestion2"]
            }

        Examples:
            >>> optimizer = ExecutionOptimizer(config)
            >>> plan = optimizer.create_execution_plan({}, config)
            >>> analysis = optimizer.analyze_critical_path(plan)
            >>> analysis["total_cost"]
            10.5
        """
        if not plan:
            return {
                "critical_path": [],
                "total_cost": 0.0,
                "parallel_opportunities": 0,
                "optimization_suggestions": []
            }

        # Build stage name to config mapping
        stage_map = {stage["name"]: stage for stage in plan}

        # Find critical path using dynamic programming
        # For each stage, calculate max cost to reach it
        max_costs = {}
        predecessors = {}

        def calculate_max_cost(stage_name: str) -> float:
            """Calculate maximum cost to reach this stage."""
            if stage_name in max_costs:
                return max_costs[stage_name]

            if stage_name not in stage_map:
                return 0.0

            stage = stage_map[stage_name]
            stage_cost = stage.get("estimated_cost", 1.0)

            # Get dependencies
            depends_on = self.dependency_graph.get(stage_name, {}).get("depends_on", [])

            if not depends_on:
                # Starting stage
                max_costs[stage_name] = stage_cost
                return stage_cost

            # Find max cost through dependencies
            dep_costs = [(dep, calculate_max_cost(dep)) for dep in depends_on]
            max_dep, max_dep_cost = max(dep_costs, key=lambda x: x[1])

            total_cost = max_dep_cost + stage_cost
            max_costs[stage_name] = total_cost
            predecessors[stage_name] = max_dep

            return total_cost

        # Calculate max costs for all stages
        for stage in plan:
            calculate_max_cost(stage["name"])

        # Find ending stage with max cost (critical path end)
        if not max_costs:
            return {
                "critical_path": [],
                "total_cost": 0.0,
                "parallel_opportunities": 0,
                "optimization_suggestions": []
            }

        end_stage, total_cost = max(max_costs.items(), key=lambda x: x[1])

        # Reconstruct critical path
        critical_path = []
        current = end_stage

        while current:
            critical_path.append(current)
            current = predecessors.get(current)

        critical_path.reverse()

        # Count parallel opportunities (stages not on critical path)
        parallel_opportunities = len(plan) - len(critical_path)

        # Generate optimization suggestions
        suggestions = []

        if parallel_opportunities > 0:
            suggestions.append(
                f"Consider running {parallel_opportunities} stages in parallel "
                "to reduce total execution time"
            )

        # Check for expensive stages on critical path
        expensive_threshold = total_cost / len(critical_path) * 1.5
        for stage_name in critical_path:
            stage = stage_map.get(stage_name)
            if stage and stage.get("estimated_cost", 0) > expensive_threshold:
                suggestions.append(
                    f"Stage '{stage_name}' is expensive and on critical path. "
                    "Consider optimizing or caching its results."
                )

        # Check for long critical path
        if len(critical_path) > 5:
            suggestions.append(
                "Critical path has many sequential stages. "
                "Consider restructuring dependencies to enable more parallelism."
            )

        logger.info(
            "Critical path analysis complete",
            extra={
                "path_length": len(critical_path),
                "total_cost": total_cost,
                "parallel_opportunities": parallel_opportunities
            }
        )

        return {
            "critical_path": critical_path,
            "total_cost": total_cost,
            "parallel_opportunities": parallel_opportunities,
            "optimization_suggestions": suggestions
        }
