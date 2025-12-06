from typing import TypedDict

class CVSchema(TypedDict):
    """
    Schema for CV metadata in the Vector DB.
    """
    name: str
    years_experience: int
    skills: list[str]
    session_id: str
