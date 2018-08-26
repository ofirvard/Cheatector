from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import os
import json
import numpy as np

features = {}
answers = {}
frame_size = 0.1
frame_stepsize = 0.05
datadir = "./Sounds/"
speaker_file = 'speaker_features.json'

with open(speaker_file, 'r') as data:
	speaker_features = json.load(data)
total = 0
for i, dirname in enumerate(os.listdir(datadir)):
	speaker = dirname 
	for filename in os.listdir(datadir + dirname):
        	answers[filename] = 1 if "lie" in filename else 0
		[Fs, x] = audioBasicIO.readAudioFile(datadir + dirname + "/" + filename)
		speaker_feature = speaker_features[dirname]
		st_features = audioFeatureExtraction.stFeatureExtraction(x, Fs, frame_size*Fs, frame_stepsize*Fs)
		num_features, num_windows = st_features.shape
		new_features = np.zeros((num_features, num_windows))
		for i in range(num_features):
			new_features[i] = (st_features[i] - speaker_feature[i]) / speaker_feature[i]
		st_features = np.concatenate((st_features, new_features))
		features[filename] = st_features.tolist()
		total += 1
with open('labels.json', 'w') as label_file:
	json.dump(answers,label_file)
with open('features.json','w') as feature_file:
	json.dump(features,feature_file)
