import os.path
import hashlib

class Path(object):
    canonical = None
    digest = None

    def __init__(self, path):
        self.canonical = os.path.abspath(path)
        self.digest = hashlib.sha256(self.canonical.encode('utf-8')).hexdigest()
    
    def __str__(self): 
        return self.canonical

    def __repr__(self): 
        return self.canonical
        