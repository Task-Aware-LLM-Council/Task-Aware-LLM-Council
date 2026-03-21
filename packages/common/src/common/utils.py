def to_text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)
