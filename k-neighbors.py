from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import skew
import numpy as np
import json

summarized_features = []
labels = []

summarized_features_by_id = {}
labels_by_id = {}

with open('features.json', 'r') as feature_file:
	with open('labels.json', 'r') as label_file:
		features = json.load(feature_file)
		labels = json.load(label_file)
		for speaker, data in features.iteritems():
			if speaker not in labels_by_id:
				labels_by_id[speaker] = []
				summarized_features_by_id[speaker] = []
			for filename, segment in features[speaker].iteritems():
				labels_by_id[speaker].append(labels[speaker][filename])

				segment = np.array(segment)
				avgs = np.mean(segment,axis=1)
				maxs = np.max(segment,axis=1)
				mins = np.min(segment,axis=1)
				skews = skew(segment,axis=1)
				std = np.std(segment,axis=1)
				summary = np.hstack([avgs,maxs,mins,skews,std])
				summarized_features_by_id[speaker].append(summary)

labels = np.array(labels)

all_preds = []
all_test_y = []

for speaker, data in summarized_features_by_id.iteritems():
	x = np.stack(summarized_features_by_id[speaker])
	y = np.array(labels_by_id[speaker])
	indices = np.random.permutation(len(y))
	lastTraining = 7*len(x)/10
	training_idx, test_idx = indices[:lastTraining], indices[lastTraining:]
	training_x, test_x = x[training_idx,:], x[test_idx,:]
	training_y, test_y = y[training_idx], y[test_idx]
	neigh = KNeighborsClassifier(n_neighbors=6) #default k=5, seems fine to me
	neigh.fit(training_x, training_y) 
	preds = neigh.predict(test_x)
	all_preds.append(preds)
	all_test_y.append(test_y)
	print "results for speaker " + str(speaker)
	print "accuracy = " + str(np.sum(preds == test_y)/float(len(preds)))
	print classification_report(test_y,preds)
all_preds = np.hstack(all_preds)
all_test_y = np.hstack(all_test_y)
print "results for all subjects:"
print "accuracy = " + str(np.sum(all_preds == all_test_y)/float(len(all_preds)))
print classification_report(all_test_y,all_preds)
