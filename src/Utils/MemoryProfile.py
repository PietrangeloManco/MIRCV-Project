from dataclasses import dataclass


@dataclass
class MemoryProfile:
    memory_per_doc: float
    estimated_chunk_size: int
