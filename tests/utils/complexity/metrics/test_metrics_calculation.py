"""Tests for all metrics using the standard test log and expected values."""

import json
import pytest
import numpy as np
from pathlib import Path
from typing import Dict, Any

from utils.complexity.metrics.registry import discover_metrics, available_metric_names
from utils.complexity.metrics.metric_orchestrator import MetricOrchestrator
from utils.windowing.window import Window
from utils.constants import ALL_METRIC_NAMES
from tests.conftest import assert_close, standard_test_log


class TestMetricsCalculation:
    """Golden tests using the standard test log and expected values from JSON."""
    
    @pytest.fixture
    def expected_values(self) -> Dict[str, Any]:
        """Load expected values from JSON file."""
        # Use a more robust path resolution that works regardless of file location
        current_file = Path(__file__)
        # Go up from tests/utils/complexity/metrics/ to tests/ directory
        tests_dir = current_file.parent.parent.parent.parent
        json_path = tests_dir / "expected_metric_values.json"
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    
    @pytest.fixture
    def computed_metrics(self, standard_test_log) -> Dict[str, Any]:
        """Compute all metrics on the standard test log."""
        # Discover all metrics
        discover_metrics()
        
        # Create a window object
        window = Window(
            id="test_window",
            size=len(standard_test_log),
            traces=standard_test_log,
            population_distributions=None
        )
        
        # Create orchestrator and compute all metrics
        orchestrator = MetricOrchestrator(strict=False, prefer="trace")
        store, info = orchestrator.compute_many_metrics(ALL_METRIC_NAMES, window)
        
        # Extract results
        results = {}
        for name in ALL_METRIC_NAMES:
            if store.has(name):
                measure = store.get(name)
                results[name] = {
                    "value": measure.value,
                    "hidden": measure.hidden,
                    "meta": measure.meta
                }
            else:
                results[name] = {
                    "value": None,
                    "hidden": None,
                    "meta": None,
                    "error": "Metric not computed"
                }
        
        return results, info
    
    def test_all_metrics_computed(self, computed_metrics):
        """Test that all metrics were computed successfully."""
        results, info = computed_metrics
        computed_count = len([r for r in results.values() if r['value'] is not None])
        total_count = len(results)
        
        print(f"Computed {computed_count}/{total_count} metrics successfully")
        print(f"Skipped metrics: {info.get('skipped', [])}")
        
        # All metrics should be computed
        assert computed_count == total_count, f"Only {computed_count}/{total_count} metrics computed"
        assert len(info.get('skipped', [])) == 0, f"Skipped metrics: {info.get('skipped', [])}"
    
    def test_metric_values_match_expected(self, expected_values, computed_metrics):
        """Test that computed metric values match expected values."""
        results, _ = computed_metrics
        expected_metrics = expected_values["computed_metrics"]
        
        # Tolerance map for different metric types
        tolerances = {
            'default': 1e-10,
            'float': 1e-6,
            'lempel_ziv': 1e-6,
            'edit_distance': 1e-6,
            'affinity': 1e-6,
            'deviation': 1e-6,
            'structure': 1e-6,
            'percentage': 1e-6,
        }
        
        mismatches = []
        
        for metric_name, expected_value in expected_metrics.items():
            if metric_name not in results:
                mismatches.append(f"Missing metric: {metric_name}")
                continue
                
            computed_data = results[metric_name]
            if computed_data['value'] is None:
                mismatches.append(f"Metric not computed: {metric_name}")
                continue
            
            computed_value = computed_data['value']
            
            # Determine tolerance based on metric type
            if 'Lempel-Ziv' in metric_name:
                tolerance = tolerances['lempel_ziv']
            elif 'Edit Distance' in metric_name:
                tolerance = tolerances['edit_distance']
            elif 'Affinity' in metric_name:
                tolerance = tolerances['affinity']
            elif 'Deviation' in metric_name:
                tolerance = tolerances['deviation']
            elif 'Structure' in metric_name:
                tolerance = tolerances['structure']
            elif 'Percentage' in metric_name:
                tolerance = tolerances['percentage']
            elif isinstance(expected_value, float):
                tolerance = tolerances['float']
            else:
                tolerance = tolerances['default']
            
            # Compare values
            if isinstance(expected_value, (int, float)) and isinstance(computed_value, (int, float)):
                if not np.isclose(expected_value, computed_value, rtol=0, atol=tolerance):
                    mismatches.append(
                        f"{metric_name}: expected {expected_value}, got {computed_value} "
                        f"(diff: {abs(expected_value - computed_value)})"
                    )
            elif expected_value != computed_value:
                mismatches.append(
                    f"{metric_name}: expected {expected_value}, got {computed_value}"
                )
        
        if mismatches:
            print("Metric value mismatches:")
            for mismatch in mismatches:
                print(f"  - {mismatch}")
            assert False, f"Found {len(mismatches)} metric value mismatches"
    
    def test_hand_computed_values_verification(self, expected_values, computed_metrics):
        """Test that hand-computed values match the computed values."""
        results, _ = computed_metrics
        hand_computed = expected_values["hand_computed_values"]
        
        for metric_name, expected_value in hand_computed.items():
            if metric_name not in results:
                continue
                
            computed_data = results[metric_name]
            if computed_data['value'] is None:
                continue
            
            computed_value = computed_data['value']
            
            # For hand-computed values, we expect exact matches
            if isinstance(expected_value, (int, float)) and isinstance(computed_value, (int, float)):
                assert np.isclose(expected_value, computed_value, rtol=0, atol=1e-10), \
                    f"{metric_name}: hand-computed {expected_value} != computed {computed_value}"
            else:
                assert expected_value == computed_value, \
                    f"{metric_name}: hand-computed {expected_value} != computed {computed_value}"
    
    def test_requested_metrics_not_hidden(self, computed_metrics):
        """Test that all requested metrics are visible (hidden=False)."""
        results, _ = computed_metrics
        
        for metric_name, data in results.items():
            if data['value'] is None:
                continue
            
            # All computed metrics should not be hidden
            assert not data['hidden'], f"Metric {metric_name} should not be hidden"
    
    def test_metric_metadata_consistency(self, computed_metrics):
        """Test that metric metadata is consistent."""
        results, _ = computed_metrics
        
        for metric_name, data in results.items():
            if data['value'] is None:
                continue
            
            # All computed metrics should have metadata
            assert data['meta'] is not None, f"Metric {metric_name} should have metadata"
            
            # Trace-based metrics should have "basis": "traces" in metadata
            # Some metrics may have different basis values (e.g., "observation distribution", "derived")
            if 'basis' in data['meta']:
                basis = data['meta']['basis']
                assert basis in ['traces', 'observation distribution', 'derived'], \
                    f"Metric {metric_name} should have valid basis, got '{basis}'"
    
    def test_metric_properties(self, computed_metrics):
        """Test that metrics satisfy expected properties."""
        results, _ = computed_metrics
        
        # Test non-negative properties
        non_negative_metrics = [
            "Number of Traces", "Number of Events", "Number of Distinct Traces",
            "Number of Distinct Activities", "Number of Distinct Activity Transitions",
            "Min. Trace Length", "Max. Trace Length", "Avg. Trace Length",
            "Percentage of Distinct Traces", "Average Distinct Activities per Trace",
            "Lempel-Ziv Complexity", "Structure", "Estimated Number of Acyclic Paths",
            "Number of Ties in Paths to Goal"
        ]
        
        for metric_name in non_negative_metrics:
            if metric_name in results and results[metric_name]['value'] is not None:
                value = results[metric_name]['value']
                assert value >= 0, f"Metric {metric_name} should be non-negative, got {value}"
        
        # Test range properties
        if "Percentage of Distinct Traces" in results and results["Percentage of Distinct Traces"]['value'] is not None:
            value = results["Percentage of Distinct Traces"]['value']
            assert 0 <= value <= 1, f"Percentage should be in [0,1], got {value}"
        
        if "Structure" in results and results["Structure"]['value'] is not None:
            value = results["Structure"]['value']
            assert 0 <= value <= 1, f"Structure should be in [0,1], got {value}"
        
        # Test trace length relationships
        if all(metric in results and results[metric]['value'] is not None 
               for metric in ["Min. Trace Length", "Max. Trace Length", "Avg. Trace Length"]):
            min_len = results["Min. Trace Length"]['value']
            max_len = results["Max. Trace Length"]['value']
            avg_len = results["Avg. Trace Length"]['value']
            
            assert min_len <= avg_len <= max_len, \
                f"Trace length relationship violated: {min_len} <= {avg_len} <= {max_len}"
    
    def test_metric_consistency_relationships(self, computed_metrics):
        """Test that related metrics have consistent relationships."""
        results, _ = computed_metrics
        
        # Number of distinct traces should be <= number of traces
        if (all(metric in results and results[metric]['value'] is not None 
                for metric in ["Number of Traces", "Number of Distinct Traces"])):
            total_traces = results["Number of Traces"]['value']
            distinct_traces = results["Number of Distinct Traces"]['value']
            assert distinct_traces <= total_traces, \
                f"Distinct traces ({distinct_traces}) should be <= total traces ({total_traces})"
        
        # Number of distinct activities should be <= number of events
        if (all(metric in results and results[metric]['value'] is not None 
                for metric in ["Number of Events", "Number of Distinct Activities"])):
            total_events = results["Number of Events"]['value']
            distinct_activities = results["Number of Distinct Activities"]['value']
            assert distinct_activities <= total_events, \
                f"Distinct activities ({distinct_activities}) should be <= total events ({total_events})"
        
        # Average distinct activities per trace should be <= max trace length
        if (all(metric in results and results[metric]['value'] is not None 
                for metric in ["Average Distinct Activities per Trace", "Max. Trace Length"])):
            avg_distinct = results["Average Distinct Activities per Trace"]['value']
            max_length = results["Max. Trace Length"]['value']
            assert avg_distinct <= max_length, \
                f"Avg distinct activities per trace ({avg_distinct}) should be <= max trace length ({max_length})"


class TestMetricRegistryIntegration:
    """Test integration with the metric registry system."""
    
    def test_all_metrics_discoverable(self):
        """Test that all metrics can be discovered through the registry."""
        discover_metrics()
        metric_names = list(available_metric_names())
        
        assert len(metric_names) > 0, "No metrics discovered"
        
        # Check that we have expected metric categories
        expected_categories = [
            "Number of Traces", "Number of Events", "Number of Distinct Traces",
            "Number of Distinct Activities", "Number of Distinct Activity Transitions",
            "Min. Trace Length", "Max. Trace Length", "Avg. Trace Length",
            "Percentage of Distinct Traces", "Average Distinct Activities per Trace",
            "Average Edit Distance", "Average Affinity", "Deviation from Random",
            "Lempel-Ziv Complexity", "Structure", "Estimated Number of Acyclic Paths",
            "Number of Ties in Paths to Goal"
        ]
        
        for expected_metric in expected_categories:
            assert expected_metric in metric_names, f"Expected metric {expected_metric} not found in registry"
    
    def test_metric_orchestrator_integration(self, standard_test_log):
        """Test that the metric orchestrator can compute all metrics."""
        discover_metrics()
        metric_names = list(available_metric_names())
        
        # Create a window object
        window = Window(
            id="test_window",
            size=len(standard_test_log),
            traces=standard_test_log,
            population_distributions=None
        )
        
        # Test with strict mode
        orchestrator_strict = MetricOrchestrator(strict=True, prefer="trace")
        store_strict, info_strict = orchestrator_strict.compute_many_metrics(metric_names, window)
        
        # All metrics should be computed in strict mode
        computed_count = len([name for name in metric_names if store_strict.has(name)])
        assert computed_count == len(metric_names), \
            f"Strict mode should compute all metrics, got {computed_count}/{len(metric_names)}"
        
        # Test with non-strict mode
        orchestrator_non_strict = MetricOrchestrator(strict=False, prefer="trace")
        store_non_strict, info_non_strict = orchestrator_non_strict.compute_many_metrics(metric_names, window)
        
        # Should also compute all metrics in non-strict mode for this simple case
        computed_count = len([name for name in metric_names if store_non_strict.has(name)])
        assert computed_count == len(metric_names), \
            f"Non-strict mode should also compute all metrics, got {computed_count}/{len(metric_names)}"
