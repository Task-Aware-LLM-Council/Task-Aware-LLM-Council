import os

def to_text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)

def get_current_user() -> str:
    username = os.getenv("USER") or os.getenv("USERNAME")  # For cross-platform compatibility
    return username
