from dataclasses import dataclass
from typing import Any

# Posting list structure
@dataclass
class Posting:
    doc_id: int
    payload: Any = None