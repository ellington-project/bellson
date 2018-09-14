import librosa.core

# Configuration for IO from audio files
class Configuration:
    start = 60.0
    samplelen = 20.0
    lowf=64
    highf=320
    samplerate = 44100
    nfft = 2048
    def __init__(self, start=60, samplelen=20, lowf=64, highf=320, samplerate=44100, nfft=2048):
        self.start = start
        self.samplelen = samplelen
        self.lowf = lowf
        self.highf = highf
        self.samplerate = samplerate
        self.nfft = nfft 

    def seconds_to_frames(self, time): 
        return librosa.core.time_to_frames(time, self.samplerate, self.nfft/4, self.nfft)

    def sample_frames(self): 
        return self.seconds_to_frames(self.samplelen)

    def sample_shape(self): 
        h = int(self.highf-self.lowf)
        w = int(self.sample_frames())
        return (w,h)

def main(time): 
    # Config
    config = Configuration()
    logging.info("Frames from sample: " + str(config.seconds_to_frames(float(time))))
    logging.info("Config sample frames: " + str(config.sample_frames()))
    logging.info("Config sample shape: " + str(config.sample_shape()))

if __name__ == '__main__':
    import logging
    import argparse
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(module)s %(lineno)d : %(message)s', level=logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument('--time', required=True, help='Time period to convert to frame count')
    args = parser.parse_args()
    arguments = args.__dict__
    main(**arguments)
