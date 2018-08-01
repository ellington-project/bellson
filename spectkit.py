#!/usr/bin/env python3
# On OSX, this must be invoked with: 
# export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
# in order to disable certain checks that stop multiprocessing from working
from __future__ import print_function
from multiprocessing import Pool
from pprint import pprint, pformat
import librosa
import librosa.display

import numpy as np
import json
import os
from pathlib import Path

# Extract audio data from a sample
#     audiodata = numpy float array
#     samplerate = int
#     start = seconds
#     length = seconds
def extract_audio(audiodata, samplerate, start=60, length=60):
    if start == None or length == None: 
        return audiodata
    # get the length of the audio data.
    duration = librosa.core.get_duration(audiodata)
    # Check to see that start + length < duration 
    if start + length > duration: 
        raise ValueError('Requested time longer than audio duration', str(start), str(length), str(duration)) 
    # Calculate where the start would be, based on the sample rate
    startframe = start * samplerate
    # Calculate where the end would be, based on the sample rate
    endframe = (start + length) * samplerate
    # Return a slice of the numpy array
    return audiodata[startframe:endframe]

def save_spectrogram(audio_dat, filename):
    import matplotlib.pyplot as plt
    print("Saving spectrogram to " + filename)
    fig = plt.figure(figsize=(12, 12))
    S = librosa.stft(audio_dat)
    M = librosa.core.magphase(S)[0]
    D = librosa.amplitude_to_db(M, ref=np.max)
    ax = plt.subplot(111)
    ax.set_frame_on(False)
    plt.axis('off')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    librosa.display.specshow(D, y_axis='linear')
    fig.savefig(filename, dpi=100, bbox_inches='tight', pad_inches=0.0)
    plt.close(fig)

class Track: 
    bpm=None
    filename=None
    spect_path=None
    @classmethod
    def from_json(cls,json):
        metadata = json['metadata']
        filename = json['location']
        if metadata is not None and filename is not None: 
            bpm = metadata['bpm']
            if bpm is not None and bpm is not 0: 
                return Track(bpm, filename)
        return None
    
    def __init__(self, bpm, filename): 
        self.bpm = bpm
        self.filename = filename
        self.spect_path = "tmp/spect/" + str(bpm) + "/"
        
    def __str__(self):
        return "T["+str(self.bpm) + "," + str(self.filename) + "]"
    def __repr__(self):
        return "T["+str(self.bpm) + "," + str(self.filename) + "]"

    def write_spectrogram(self): 
        (y, sr) = librosa.load(self.filename)
        try:
            os.makedirs(self.spect_path)
        except: 
            print("Path already exists: " + self.spect_path)
        t = extract_audio(y, sr)
        print("Original length: " + str(librosa.core.get_duration(y)))
        print("Reduced length: " + str(librosa.core.get_duration(t)))
        imgname = Path(self.filename).stem + ".png"
        save_spectrogram(t, self.spect_path + "/" + imgname)



class EllingtonData: 
    tracks=[]
    @classmethod
    def from_file(cls,filename): 
        with open(filename) as f: 
            json_data = json.load(f)
            return EllingtonData(json_data)
            
    def __init__(self, json): 
        self.tracks = list(filter(lambda t: t is not None, map(Track.from_json, json['tracks'])))
    def __str__(self): 
        return str(self.tracks)
    def __repr__(self):
        return pformat(self.tracks)
     
def proc(tp):
    i = tp[0]
    track = tp[1]
    track.write_spectrogram()
    print("["+str(i) + "]")

def main():
    ed = EllingtonData.from_file('example.json')

    pprint(ed)

    tracks = list(enumerate(ed.tracks))

    pool = Pool()
    pool.map(proc, tracks)

    # tc = str(len(ed.tracks))
    # ix = 0
    # for t in ed.tracks: 
    #     print("["+str(ix) + "/" + tc + "]")
    #     t.write_spectrogram()
    #     ix = ix + 1


if __name__ == '__main__': 
    main()
