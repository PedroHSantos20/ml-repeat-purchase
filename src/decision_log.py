# src/decision_log.py -- small, project-specific decision diary (adapted idea, own code)
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Decision:
    step: str                         
    column: str
    action: str
    reason: str
    rows_train: int
    rows_test: int                    # 0 for train-only steps
    carries: dict = field(default_factory=dict)  


class DecisionLog:
    def __init__(self):
        self.items: list[Decision] = []

    def record(self, step, column, action, reason, rows_train, rows_test, carries=None):
        if not reason.strip():
            raise ValueError("every decision needs a reason")  
        self.items.append(
            Decision(step, column, action, reason,
                     int(rows_train), int(rows_test), carries or {})
        )

    def __len__(self):
        return len(self.items)

    def __repr__(self):
        return f"DecisionLog(decisions={len(self.items)})"

    def to_frame(self):
        # One row per decision, for display in the notebook (carries column hidden).
        if not self.items:
            return pd.DataFrame(
                columns=["step", "column", "action", "reason", "rows_train", "rows_test"]
            )
        return pd.DataFrame([asdict(d) for d in self.items]).drop(columns="carries")

    def to_json(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps([asdict(d) for d in self.items], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path):
        log = cls()
        for d in json.loads(Path(path).read_text(encoding="utf-8")):
            log.items.append(Decision(**d))
        return log