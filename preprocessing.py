import essentia
import essentia.standard
import os
import scipy.io.wavfile
import scipy.io
from essentia.standard import *
import numpy
from sklearn import preprocessing

numpy.seterr(all='warn')

def getSegment(frames, start, end):
    rows = frames[start : end+1, ]
    return numpy.hstack(rows)

def convertManyToOne(Y):
    newY = numpy.empty((0, 1))
    for i in xrange(len(Y)):
        for j in xrange(len(Y[i])):
            if Y[i][j] == 1:
                newY = numpy.vstack([newY, j])
                break
    return newY


folder_path = '/Users/shamitlal/Desktop/shamit/sem 6/minor/download/wav/'
audios = []
emotions = []
emotion_class_map = {'W' : 0, 'L' : 1, 'E' : 2, 'A' : 3, 'F' : 4, 'T' : 5, 'N' : 6}
num_emotions = len(emotion_class_map)

i=0

for filename in os.listdir(folder_path):
    if i==0:
        i=1
        continue
    filepath = folder_path + filename
    # print filepath
    #rate, data = scipy.io.wavfile.read(filename = filepath)
    loader = essentia.standard.MonoLoader(filename = filepath)
    data = loader()
    emotion = filepath[-6] #2nd last character of file exluding extension name wav
    emotion_class = emotion_class_map[emotion]
    audios.append(data)
    emotions.append(emotion_class)

sample_rate = 16000 # in hertz
frameDuration = 0.025 #duration in seconds
hopDuration = 0.010
frameSize = int(sample_rate*frameDuration)
hopSize = int(sample_rate*hopDuration)
featuresPerFrame = 13 + 1
framesPerSegment = 25
featuresPerSegment = featuresPerFrame * framesPerSegment
segmentHop = 13

w = Windowing(type = 'hann')
spectrum = Spectrum()
mfcc = MFCC()

X = numpy.empty((0, featuresPerSegment))
Y = numpy.empty((0, num_emotions))
energy_func = essentia.standard.Energy()

for i in range(len(audios)):
    print i
    audio = audios[i]
    output = emotions[i]
    output_vec = numpy.zeros((1, num_emotions))
    output_vec[0][output] = 1
    frames = numpy.empty((featuresPerFrame, )) #each index stores a list of 53 features for that frame
    # frames = numpy.matrix(frames)
    #for frame in FrameGenerator(essentia.array(audio), frameSize = frameSize, hopSize = hopSize):
    for frame in FrameGenerator(audio, frameSize = frameSize, hopSize = hopSize):
        mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame))) #40 mfcc bands and 13 mfcc_coeffs
        # frame_features = numpy.concatenate((mfcc_bands,mfcc_coeffs))
        frame_energy = energy_func(audio)
        frame_features = numpy.append(mfcc_coeffs, frame_energy)
        if numpy.isnan(frame_features).any() :
            print "nan\nnan\n"
            exit()
        frames = numpy.vstack([frames,frame_features])

    start_segment = 0
    while True:
        end_segment = start_segment + framesPerSegment - 1 #center = 13, left=1-12, right=14-25
        if end_segment >= len(frames) :
            break
        segment = getSegment(frames, start_segment, end_segment)
        start_segment = start_segment + segmentHop -1 #segmentSize = 13
        X = numpy.vstack([X, segment])
        Y = numpy.vstack([Y, output_vec])

X_scaled = preprocessing.scale(X)
scipy.io.savemat('X.mat', {'X' : X})
scipy.io.savemat('Y.mat', {'Y' : Y})
scipy.io.savemat('X_scaled.mat', {'X_scaled' : X_scaled})

