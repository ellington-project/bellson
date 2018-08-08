import logging

from ellington_library import EllingtonLibrary, Track
from audio import AudioTrack

def main():
    logging.basicConfig(format='%(asctime)s %(levelname)s %(module)s %(lineno)d : %(message)s', level=logging.DEBUG)
    el = EllingtonLibrary.from_file("example.el")
    count = str(len(el.tracks))
    ix = 0
    for t in el.tracks: 
        ix = ix + 1
        print("Track: " + str(t.trackname) + " " + str(ix) + " / " + count)
        print("\t Testing audio data: ")
        
        audiotrack = AudioTrack(t)

        for ad in audiotrack.audio_intervals(True): 
            logging.debug("Audio data recieved")
            # print(ad[0:10])

        print("\t Training audio data: ")

        for ad in audiotrack.audio_intervals(): 
            logging.debug("Audio data recieved")
            # print(ad[0:10])

if __name__ == '__main__':
    main()