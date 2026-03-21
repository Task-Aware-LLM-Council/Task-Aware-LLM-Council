import hashlib

def make_id(source: str, orig_id) -> str:
    raw = f"{source}_{orig_id}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]