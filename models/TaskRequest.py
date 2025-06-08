from pydantic import BaseModel
from typing import List

class TaskRequest(BaseModel):
    src_dir: str
    output_dir: str
    threshold: float
    size: str
    format: str
    types: List[str]
    center: bool
    blank: str