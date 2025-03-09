import os

def make_path(path):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path