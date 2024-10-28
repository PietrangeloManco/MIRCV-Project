from dataclasses import dataclass
from typing import Any

@dataclass
class Posting:
    doc_id: int
    payload: Any = None