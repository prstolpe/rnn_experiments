
def flatten(some_list):
    """Flatten a python list."""
    return [some_list] if not isinstance(some_list, list) else [x for X in some_list for x in flatten(X)]
