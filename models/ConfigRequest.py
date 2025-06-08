from pydantic import BaseModel
from typing import List

class ConfigRequest(BaseModel):
    name: str
    src_dir: str
    output_dir: str
    threshold: float
    size: str
    format: str
    types: List[str]
    center: bool
    blank: str