import psutil
from pandas import DataFrame


class MemoryTrackingTools:
    def __init__(self):
        pass

    @staticmethod
    def get_available_memory() -> int:
        """Get available memory on the system (in bytes)."""
        available_memory = psutil.virtual_memory().available
        return available_memory

    def calculate_target_memory_usage(self) -> int:
        """Calculate the target memory usage to leave 10% of memory free."""
        available_memory = self.get_available_memory()
        target_memory_usage = round(available_memory * 0.90)  # Leave 10% free
        return target_memory_usage

    @staticmethod
    def get_memory_usage(chunk: DataFrame) -> int:
        """Estimate the memory usage of the chunk."""
        return sum([len(str(doc)) for doc in chunk])

    @staticmethod
    def get_total_memory() -> int:
        """
        Get the total system memory in bytes.
        """
        return psutil.virtual_memory().total
