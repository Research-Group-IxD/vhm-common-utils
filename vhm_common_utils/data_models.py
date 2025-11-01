from pydantic import BaseModel, Field
from typing import Dict, Any, List
import datetime

class Anchor(BaseModel):
    anchor_id: str
    text: str
    stored_at: datetime.datetime
    salience: float = Field(default=1.0)
    meta: Dict[str, Any] = Field(default_factory=dict)

class IndexedAnchorResponse(BaseModel):
    anchor_id: str
    ok: bool
    reason: str | None = None
    detail: str | None = None
