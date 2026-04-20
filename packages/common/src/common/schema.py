import json
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any


@dataclass
class RouterExample:
    id: str
    source_dataset: str
    question: str
    context: str = ""
    gold_answer: str = ""
    gold_label: str = ""
    unit_tests: str = ""
    skill_tags: List[str] = None
    hallucination_subset: bool = False
    split: str = ""
    original_id: str = ""
    metadata: Dict[str, Any] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        if d["skill_tags"] is None:
            d["skill_tags"] = []
        if d["metadata"] is None:
            d["metadata"] = {}

        # Stringify the dictionary so Parquet just sees a normal string column
        d["metadata"] = json.dumps(d["metadata"])
        return d
