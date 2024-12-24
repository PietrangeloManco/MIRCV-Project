import unittest

from Utils.MemoryTrackingTools import MemoryTrackingTools


class TestMemoryTrackingTools(unittest.TestCase):
    def setUp(self):
        """Set up before each test."""
        self.memory_tool = MemoryTrackingTools()

    def test_get_available_memory(self):
        """Test if the method returns a non-negative available memory value."""
        available_memory = self.memory_tool.get_available_memory()
        self.assertGreaterEqual(available_memory, 0, "Available memory should be non-negative.")

    def test_get_total_memory(self):
        """Test if the method returns a non-negative total memory value."""
        total_memory = self.memory_tool.get_total_memory()
        self.assertGreater(total_memory, 0, "Total memory should be non-negative.")


if __name__ == "__main__":
    unittest.main()
