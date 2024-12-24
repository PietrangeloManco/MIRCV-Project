import psutil


class MemoryTrackingTools:
    def __init__(self):
        """Initialize the MemoryTrackingTools class."""
        pass

    @staticmethod
    def get_available_memory() -> int:
        """
        Get available memory on the system (in bytes).

        Returns:
            int: Available system memory in bytes.
        """
        available_memory = psutil.virtual_memory().available
        return available_memory

    @staticmethod
    def get_total_memory() -> int:
        """
        Get the total system memory in bytes.

        Returns:
            int: Total system memory in bytes.
        """
        return psutil.virtual_memory().total
