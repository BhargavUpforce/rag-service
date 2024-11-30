from typing import List

from pydantic import BaseModel


class UploadResponse(BaseModel):
    message: str
    stored_chunks_ids: List[str]