from dataclasses import dataclass
from typing import Any

@dataclass
class Posting:
    doc_id: int
    payload: Any = None  # Store term frequency or any other payload

    def set_payload(self, payload: Any) -> None:
        """Set the payload (e.g., term frequency)."""
        self.payload = payload

    def get_payload(self) -> Any:
        """Get the payload (e.g., term frequency)."""
        return self.payload
