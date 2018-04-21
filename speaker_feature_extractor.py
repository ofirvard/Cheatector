from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import os
import json
import numpy as np


features = {}

datadir = "/home/ron/Downloads/Sounds/"
frame_size = 0.1
frame_stepsize = 0.05

total = 0
for i, dirname in enumerate(os.listdir(datadir)):
	speaker = dirname
	speaker_count = 0
	st_features = 0
	for filename in os.listdir(datadir + dirname):
		if "Lie" in filename:
			continue
		else:
			filepath = datadir + dirname + '/' + filename
			[Fs, x] = audioBasicIO.readAudioFile(filepath)
			audio_features = audioFeatureExtraction.stFeatureExtraction(x, Fs, frame_size*Fs, frame_stepsize*Fs)
			audio_features = np.mean(audio_features, axis=1)
			st_features += audio_features
			speaker_count += 1
			total += 1
	features[speaker] = (st_features / speaker_count).tolist()

with open('speaker_features.json', 'w') as feature_file:
	json.dump(features,feature_file)
