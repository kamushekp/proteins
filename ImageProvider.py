import cv2
from os import path
import pandas as pd
import numpy as np
from PathProvider import PathProvider

class ImageProvider:
    def __init__(self):
        self.path_provider = PathProvider()
        self.train_df = pd.read_csv(self.path_provider.get_train_df_path(), index_col=0)
        self.train_df['ID'] = self.train_df.index
    
    def _get_train_id(self, num):
        return self.train_df.iloc[num].ID
    
    def load_train_sample_by_number(self, num, colors = ['red', 'green', 'blue', 'yellow']):
        return self.load_train_sample(self._get_train_id(num), colors)
    
    def load_train_sample(self, image_id, colors):
        labels = self._load_train_labels(image_id)
        image_path = self.path_provider.get_train_image_path(image_id)
        image = self._load_RGBY_image(image_path, colors)
        return image, labels
    
    def _load_RGBY_image(self, image_id_path, colors):
        def load_image(suffix):
            full_path = image_id_path + suffix
            if not path.isfile(full_path):
                raise FileNotFoundError(full_path)

            return cv2.imread(full_path, cv2.IMREAD_UNCHANGED)

        images = [load_image(f"_{color}.png") for color in colors]

        red_image = load_image("_red.png")
        green_image = load_image("_green.png")
        blue_image = load_image("_blue.png")
        yellow_image = load_image("_yellow.png")
        finalImage = np.stack(images, axis=-1)
        return finalImage

    def _load_train_labels(self, image_id):
        if not image_id in self.train_df.index:
            raise FileNotFoundError(image_id)
        return self.train_df.loc[image_id].Label.split(sep='|')
    
    