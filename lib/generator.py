import logging

from .ellington_library import EllingtonLibrary, Track
from .audio import AudioTrack

class DataGenerator: 

    library = None

    # Initialise a generator from a library
    def __init__(library): 
        self.library = library
    
    # def iter(self): 
    #     for track in library: 


