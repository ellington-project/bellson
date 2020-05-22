# Bellson 
An experiment in automated swing music BPM detection using neural network techniques. Named for one of Duke Ellington's most talented drummers (and Pearl Bailey's husband), Louie Bellson.

Bellson is designed as a companion tool for the `Ellington` tool, and is part of the ellington-project. There are some features of bellson (e.g. training) that require that the user have an ellington-generated libraryfile. 

Please get in touch if you're interested in using this, and having trouble getting it working. 

## Using Bellson 

Bellson has a number of different modes/applications that should be invoked before, during, and after the training process. The most usful three are: 

- bellson.apps.util.preprep - Preprep preprepares audio data by computing spectrograms and saving them to disk. This is not mandatory for training/inference, but can produce a large speedup for the first epoch of training. 
- bellson.apps.tf.train - This app trains a neural network on an ellington library. The resulting network can be used with the inference tool to estimate the bpm of an unseen track.
- bellson.apps.tf.infer - This app acts as a "oneshot" inference tool, estimating the tempo/bpm of a single track. 

### Docker and scripts. 

For training, I strongly advise using the included dockerfile and `run_docker.sh` script. The latter is a convenience script for building the Docker container and running a (specified) script from `scripts`. For example, training a neural network can be as simple as invoking: `./run_docker train.sh`. Note that these scripts currently hardcode a number of directories (e.g. `/mnt/bigboi`) for convenience of the author. Please amend these before using the docker scripts. 

### Python examples: 

#### Preparing data: 

Prepare spectrograms for tracks in `~/library.json`, and write them to `~/bellson_cache/`

    λ → python3 -m bellson.apps.util.prepare \
        --cache-dir ~/bellson_cache/ \
        --ellington-lib ~/library.json

#### Training

Train a fresh model on tracks in `~/library.json`, storing models and tensorboard logs to `~/bellson_logs/` and reading/writing spectrogram caches from `~/bellson_cache/` (which may have been prepopulated using `bellson.apps.util.prepare`). 

    λ → python3 -m bellson.apps.tf.train \
        --ellington-lib ~/library.json \
        --job-dir ~/bellson_logs/ \
        --cache-dir ~/bellson_cache/ 

#### Inference 

Use a pre-trained model stored at `~/bellson_logs/` to compute tempo information for a single audio track `~/Music/track.mp3`.

    λ → python3 -m bellson.apps.tf.infer \ 
        --modelfile ~/bellson_logs/latest-model.h5 \
        --audiofile ~/Music/track.mp3 

## Q&A 

Some common/predictable questions, and their answers. 

### Why Louis Bellson? Why not X, who also played drums with Duke Ellington?

This is a great question - the answer is that it was a completely arbitrary choice. I wanted to name this project after one of Ellington's drummers, as it was a core "tempo" computing component, and the drummer is the core of a rhythm section in a swing band. Louis Bellson was simply the drummer with the longest wikipedia article, who seemed to have collaborated the most with Ellington, and been most vital to Ellington's success. My original choice was "Sonny Greer", who is also closely associated with Ellington, but who sadly parted ways with him in unpleasant circumstances. At the end of the day, the name is arbitrary. 

### Who is Bellson for? 

Bellson should be used by anyone who already uses (but is not happy with) tempo estimation algorithms for classifying and/or sorting their library of swing music. This might include (but is not limited to) swing dance teachers or DJs. 

### Does Bellson mean that I don't need to manually BPM my music library? 

Although Bellson is designed to give higher quality estimates than other algorithms, it should not be considered a replacement for a DJ/teacher manually listening to and BPM'ing their own library. As with all metrics, the BPM of a track only represents a single aspect, and by far not one of the most important. I suggest that users of Bellson apply it as a *guide* to their library - helping them to find tracks that they may not have previously listened to or had time to BPM. 

### Why not use X algorithm? 

Bellson came into being because I was dissatisfied with the "traditional" algorithms available for computing tempo information, as applied to swing music. Traditional algorithms are quite rigid in their understanding of what a "beat" is, and the way rhythms are expressed in music. Generally they either expect clear "dum, dum, dum" patterns, or "dum-tish-dum-tish-dum-tish" patterns where each beat is clear and equal in energy. Swing jazz, by contrast, revels in syncopation and strong and weak beats. This adds a huge amount of variety and interest to the resulting music, but causes difficulties for traditional algorithms. Instead of recognising the beat correctly, traditional algorithms often compute *multiples* of the tempo, either only noting every other beat (and computing a 0.5x multiple), or mistaking "shuffles" or syncopated rhythms as beats (and computing a 2x or greater multiple).

An alternative approach is to train a machine learning to the structure of beats and tempo in a particular genre of music, and use that model to recognise where beats fall in a particular track. Once the beats have been identified in a track the bpm can be easily calculated by taking the average of the time between each beat. In all the literature I could find on this approach, the main input to this process is a set of tracks with *beat placement* information - i.e. when exactly (in time) beats fall. This is a huge corpus of data, as each track must have the beats methodically mapped out, and such corpii simply do not exist for swing music, making this approach untenable. 

Bellson is a *third* approach: Instead of learning exact beat placements, we train an algorithm to recognise an *overall* bpm, with the details of finding beat placements hidden deep in the model. This approach requires only the data that most swing DJ's already have - a set of tracks with associated tempo/bpm information. Whether or not this approach is effective/accurate is still an open question, and a question that Bellson aims to answer. 

### How does Bellson relate to the Ellington project? 

The Ellington project is an experimental testbet/toolkit for inspecting and classifying swing music libraries. To this end, it provides a "library" abstraction for organising metrics on tracks and passing track information to other tools in the project, and an "estimator" abstraction for calculating metrics for a given track. 

Bellson uses the Ellington library abstraction/format for discovering tracks and ground truth BPM data for training/validation, and the `bellson.apps.tf.infer` app is an Ellington estimator that the toolkit can use to calculate tempo metrics. 

### Why do I need to use Ellington with Bellson? 

Ellington (the parent project) is not required for using Bellson in *inference* mode, but is only required for *training* or *validation* of the model. Ellington libraries are used to drive the training mode, directing Bellson to tracks for training and validation. Removing this dependency is a potential path for Bellson, but will require the re-implementation of a large amount of code that has been reliably developed and tested in Ellington. 