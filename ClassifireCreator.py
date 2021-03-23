from TrainMarksProvider import TrainMarksProvider
from PathProvider import PathProvider
from ImageFeaturesDto import ImageFeaturesDto
import numpy as np
import pandas as pd
import json
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
from pandas.core.common import SettingWithCopyWarning
import warnings
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

class ClassifireCreator:
    def __init__(self):
        self.train_marks_provider = TrainMarksProvider()
        self.path_provider = PathProvider()
        self.search_space = self.load_search_space()
            
    def load_search_space(self):
        path = self.path_provider.get_train_features_path()
        with open(path) as file_in:
            dtos = []
            for line in file_in:
                featuresDto = ImageFeaturesDto(**json.loads(line))
                dtos.append(featuresDto)
        return pd.DataFrame(dtos)
        
    def create_search_space_with_rank(self, rank):
        allowed = set(self.train_marks_provider._get_by_rank(rank))
        mask = self.search_space['image_number'].isin(allowed)
        
        df = self.search_space[mask]
        df['vector'] = df.apply(lambda r: np.array(r['vector']), axis=1)
        df['labels'] = df.apply(lambda r: r['labels'], axis=1)
        
        features = np.vstack(df['vector'].values)
        labels = np.vstack(np.ravel(df['labels'].values))
        return features, labels
    
    def create_classifier(self, features, labels):
        clf = KNeighborsClassifier(n_neighbors = 10, weights='distance', algorithm= 'ball_tree', n_jobs=-1)
        clf.fit(features, labels)
        return clf
    
    def create_full_classifier(self):
        features, labels = self.create_search_space_with_rank(1)
        classifire = self.create_classifier(features, labels)
        for rank in range(2,6):
            many_label_features, many_label_labels = self.create_search_space_with_rank(rank)
            one_label_features, one_label_labels = self.reformat_non_one_lable_sample_to_one_lable(classifire, labels, many_label_features, many_label_labels)
            print((features.shape,one_label_features.shape, labels.shape, one_label_labels.shape))
            features = np.concatenate((features, one_label_features))
            labels = np.concatenate((labels, one_label_labels))    
            classifire = self.create_classifier(features, labels)
            print(f'rank {rank} done')
            
        features_file = PathProvider().get_searchspace_features_path()
        with open(features_file, 'wb') as f:
            np.save(f, features)
            
        labels_file = PathProvider().get_searchspace_labels_path()
        with open(labels_file, 'wb') as f:
            np.save(f, labels)
        
    def _reformat_non_one_lable_sample_to_one_lable(self, classifier, classifier_labels, features, labels):
        distances, indices = classifier.kneighbors(features,  n_neighbors=10)
        kneighbours = classifier_labels[indices] #тут должен получится (k=10, 1) массив подходящих лейблов по возрастанию расстояния
        for i, feature_neighbours in enumerate(kneighbours):
            for vector in feature_neighbours:
                label = vector[0]
                if label in labels[i]:
                    yield features[i], label
                    break

    def reformat_non_one_lable_sample_to_one_lable(self, classifier, classifier_labels, features, labels):
        res = np.asarray(list(self._reformat_non_one_lable_sample_to_one_lable(classifier, classifier_labels, features, labels)))
        features, labels = np.vstack(np.ravel(res[:,0])), res[:,1].reshape(-1, 1)
        return features, labels