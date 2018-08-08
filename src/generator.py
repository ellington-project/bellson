from ellington_library import EllingtonLibrary, Track
from audio import AudioTrack

def main():
    print("Hello world!")
    el = EllingtonLibrary.from_file("example.el")
    for t in el.tracks: 
        print("Track: " + str(t.trackname))
        for i in AudioTrack(t).intervals(): 
            print("\t\t Interval: " + str(i))
        

if __name__ == '__main__':
    main()