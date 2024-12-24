from dataclasses import dataclass


@dataclass
class MemoryProfile:
    """
    Memory profile data class.
    """
    memory_per_doc: float
    estimated_chunk_size: int
