from SplittedImageHighLighter import SplittedImageHighLighter
from EllipseImageFeatureExtractor import EllipseImageFeatureExtractor
from PathProvider import PathProvider
from ImageProvider import ImageProvider
import numpy as np
import os
import codecs, json 
from ImageFeaturesDto import ImageFeaturesDto

class ImageFeaturesProvider:
    
    def __init__(self, image_number):
        self.image_number = image_number
        self.path_provider = PathProvider()
        self.image_provider = ImageProvider()
        
    def get(self):
        try:
            labels = self.image_provider.loat_train_label_by_number(self.image_number)

            for img, ellipse_number in SplittedImageHighLighter(self.image_number).get_all_ellipse_images():
                feature = EllipseImageFeatureExtractor().get_hue_hist(img).tolist()
                yield ImageFeaturesDto(self.image_number, feature, labels, ellipse_number)
        except Exception as e:
            pass
    