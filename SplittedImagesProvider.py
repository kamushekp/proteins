import json
from PathProvider import PathProvider
from EllipseMetaDbo import EllipseMetaDbo
from NuclearEllipse import NuclearEllipse
import cv2
import numpy as np

class SplittedImagesProvider:
    def __init__(self, image_number):
        self.path_provider = PathProvider()
        self.image_number = image_number
        self.meta = self._get_meta(image_number)
    
    def get_ellipses_count(self):
        return len(self.meta)
    
    def get_ellipse_meta(self, ellipse_number):
        return self.meta[ellipse_number]
    
    def get_ellipse_img(self, ellipse_number, color):
        path = self.path_provider.get_splitted_image_path(self.image_number, color, ellipse_number)
        return cv2.imread(path, cv2.IMREAD_UNCHANGED)
        
    def _get_meta(self, image_number):
        meta_path = self.path_provider.get_meta_path(image_number)

        with open(meta_path, 'r') as data_file:
            json_data = data_file.read()
            
        ellipses = json.loads(json_data)
        
        return [EllipseMetaDbo(**e) for e in ellipses]
    
    