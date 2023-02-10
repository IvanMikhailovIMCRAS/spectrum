def filter_opus(path):
    """
    Check whether the file is a valid opus one.
    params:
    path: str - the path to the destination file
    rtype: bool
    """
    ext = path[path.rfind('.') + 1:]
    if not ext.isdigit():
        return False
    with open(path, 'r') as f:
        try:
            f.read()
            return False
        except:
            return True
        
        