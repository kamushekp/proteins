import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from ResourceProviders.TrainPathProvider import TrainPathProvider


class TestClassifierCreator:

    def create_classifier(self):
        features, labels = self.get_features_and_labels()
        clf = KNeighborsClassifier(n_neighbors=10, weights='distance', algorithm='ball_tree', n_jobs=-1)
        clf.fit(features, labels)
        return clf

    def get_features_and_labels(self):
        features_file = TrainPathProvider().get_search_space_features_path()
        with open(features_file, 'rb') as f:
            features = np.load(f, allow_pickle=True)

        labels_file = TrainPathProvider().get_search_space_labels_path()
        with open(labels_file, 'rb') as f:
            labels = np.load(f, allow_pickle=True)
        return features, labels
