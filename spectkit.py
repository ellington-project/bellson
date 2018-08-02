#!/usr/bin/env python3
# On OSX, this must be invoked with: 
# export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
# in order to disable certain checks that stop multiprocessing from working
from __future__ import print_function
from multiprocessing import Pool
from pprint import pprint, pformat
import librosa
import librosa.display
import hashlib
import numpy as np
import json
import os
import math
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
    

# Save a segment of audio data to a spectrogram file
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

def spectrogram_interval(filename, audiodata, samplerate, start, length): 
    # get the length of the audio data.
    duration = librosa.core.get_duration(audiodata)

    # If we are not passed a start, assume zero
    if start == None:
        start = 0

    # If we are not passed a length, assume that it's the rest of the track
    if length == None: 
        length = duration - start
        
    # Check to see that start + length < duration 
    if start + length > duration: 
        raise ValueError('Requested time longer than audio duration', str(start), str(length), str(duration)) 

    # Calculate where the start would be, based on the sample rate
    startframe = start * samplerate
    # Calculate where the end would be, based on the sample rate
    endframe = (start + length) * samplerate
    # Extract a  slice of the numpy array
    audioslice = audiodata[startframe:endframe]

    # Save the slice as a file
    import matplotlib.pyplot as plt
    print("Saving spectrogram to " + filename)

    # Create a figure, and configure it
    fig = plt.figure(figsize=(12, 12))
    ax = plt.subplot(111)
    ax.set_frame_on(False)
    plt.axis('off')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    # Perform some magnitue-to-frequency calculations, and write the result to the figure
    S = librosa.stft(audioslice)
    M = librosa.core.magphase(S)[0]
    D = librosa.amplitude_to_db(M, ref=np.max)
    librosa.display.specshow(D, y_axis='linear')

    # Save the figure, and close it
    fig.savefig(filename, dpi=100, bbox_inches='tight', pad_inches=0.0)
    plt.close(fig)



class Track: 
    bpm=None
    filename=None
    trackname=None
    digest=None
    spect_path=None
    @classmethod
    def from_json(cls,json):
        metadata = json['metadata']
        filename = json['location']
        if metadata is not None and filename is not None: 
            bpm = metadata['bpm']
            trackname = metadata['name']
            if bpm is not None and bpm is not 0 and trackname is not None: 
                return Track(bpm, filename, trackname)

        return None
    
    def __init__(self, bpm, filename, trackname): 
        self.bpm = bpm
        self.filename = filename
        self.trackname = trackname
        self.digest = hashlib.sha256(trackname.encode('utf-8')).hexdigest()
        self.spect_path = "tmp/spect/" + str(bpm) + "/"
        try:
            os.makedirs(self.spect_path)
        except: 
            print("Path already exists: " + self.spect_path)
        
    def __str__(self):
        return "T["+str(self.bpm) + "," + str(self.filename) + "," + str(self.trackname) + "]"
    def __repr__(self):
        return "T["+str(self.bpm) + "," + str(self.filename) + "," + str(self.trackname) + "]"

    def interval_spect_path(self, start, end): 
        return self.spect_path + self.digest + "-"+ str(start) + "-" + str(end) + ".png"

    # Write spectrograms of this file, of 1 minute length, starting from 1 minute into the track, 
    # at an interval of 1 second
    def write_spectrograms(self): 
        # try:
        # Get the audio and sample rate
        (y, sr) = librosa.load(self.filename)
        # Get the length of the track
        duration = librosa.core.get_duration(y)
        if duration < 60: 
            print("File to short to extract 1 minute of audio")
            return 
        start = int(math.floor(min(60, duration-60)))
        stop = int(math.floor(duration))-60
        # Check for tracks < 1m
        # Iterate over 1s intervals 
        for s in range(start, stop, 1): 
            print("Creating spectrogram for file " + self.digest +" of interval [" + str(s) + "," + str(s+60) + "]")
            imgpath = self.interval_spect_path(s, s+60)
            if os.path.exists(imgpath): 
                print("File already exists: " + imgpath)
                continue
            spectrogram_interval(imgpath, y, sr, s, 60)

        # except Exception as e: 
        #     pprint(e)


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
    print("Processing track ["+str(i)+"]: ")
    pprint(track)
    track.write_spectrograms()

def main():
    ed = EllingtonData.from_file('example.json')

    pprint(ed)

    tracks = list(enumerate(ed.tracks))

    pool = Pool(8)
    pool.map(proc, tracks)


if __name__ == '__main__': 
    main()
