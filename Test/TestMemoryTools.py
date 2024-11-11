import unittest
import pandas as pd

from Utils.MemoryTrackingTools import MemoryTrackingTools


class TestMemoryTrackingTools(unittest.TestCase):
    def setUp(self):
        """Set up before each test."""
        self.memory_tool = MemoryTrackingTools()

    def test_get_available_memory(self):
        """Test if the method returns a non-negative available memory value."""
        available_memory = self.memory_tool.get_available_memory()
        self.assertGreaterEqual(available_memory, 0, "Available memory should be non-negative.")

    def test_calculate_target_memory_usage(self):
        """Test if the target memory usage is approximately 90% of available memory."""
        available_memory = self.memory_tool.get_available_memory()
        target_memory_usage = self.memory_tool.calculate_target_memory_usage()
        expected_memory_usage = available_memory * 0.90

        # Allow for a 1% tolerance in the calculation
        tolerance = 0.01 * expected_memory_usage  # 1% tolerance
        self.assertTrue(
            abs(target_memory_usage - expected_memory_usage) <= tolerance,
            f"Target memory usage should be approximately 90% of available memory. "
            f"Expected {expected_memory_usage}, but got {target_memory_usage}."
        )

    def test_get_memory_usage_empty_dataframe(self):
        """Test memory usage calculation with an empty DataFrame."""
        chunk = pd.DataFrame()
        memory_usage = self.memory_tool.get_memory_usage(chunk)
        self.assertEqual(memory_usage, 0, "Memory usage for an empty dataframe should be 0.")

    def test_get_memory_usage_non_empty_dataframe(self):
        """Test memory usage calculation with a non-empty DataFrame."""
        data = {'col1': ['data1', 'data2', 'data3'], 'col2': ['info1', 'info2', 'info3']}
        chunk = pd.DataFrame(data)
        memory_usage = self.memory_tool.get_memory_usage(chunk)
        self.assertGreater(memory_usage, 0, "Memory usage for a non-empty dataframe should be greater than 0.")

    def test_get_total_memory(self):
        """Test if the method returns a non-negative total memory value."""
        total_memory = self.memory_tool.get_total_memory()
        self.assertGreater(total_memory, 0, "Total memory should be non-negative.")

    def test_calculate_target_memory_usage_vs_total_memory(self):
        """Test that the target memory usage is less than the total memory."""
        target_memory_usage = self.memory_tool.calculate_target_memory_usage()
        total_memory = self.memory_tool.get_total_memory()
        self.assertLessEqual(target_memory_usage, total_memory,
                             "Target memory usage should be less than or equal to total system memory.")


if __name__ == "__main__":
    unittest.main()
