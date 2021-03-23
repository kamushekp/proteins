import cv2
import numpy as np

from SplittedImagesProvider import SplittedImagesProvider
from NuclearEllipse import NuclearEllipse

class SplittedImageHighLighter:
    
    def __init__(self, image_number):
        self.image_number = image_number
        self.splitted_image_provider = SplittedImagesProvider(image_number)
        self.ellipses_count = self.splitted_image_provider.get_ellipses_count()
        
    def get_all_ellipse_images(self):
        for ellipse_number in range(self.ellipses_count):
            img = [self.splitted_image_provider.get_ellipse_img(ellipse_number, color) for color in ['red', 'green', 'blue']]
            img = np.stack(img, axis=-1)
            img = self._process_ellipse_image(img, ellipse_number)
            yield img, ellipse_number
        
    def _process_ellipse_image(self, img, ellipse_number):
        meta = self.splitted_image_provider.get_ellipse_meta(ellipse_number)
        ellipse = NuclearEllipse((meta['x'], meta['y']), meta['long'], meta['short'], meta['angle'])
        ellipse_exterior = np.asarray(ellipse.initial_ellipse.exterior.xy, dtype=int).T
        stencil = np.zeros(img.shape).astype(img.dtype)
        contours = [ellipse_exterior]
        color = [255, 255, 255]
        cv2.fillPoly(stencil, contours, color)
        highlighted = cv2.bitwise_and(img, stencil)
        return highlighted