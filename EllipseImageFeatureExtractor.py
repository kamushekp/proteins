import cv2
import numpy as np

class EllipseImageFeatureExtractor:
    
    def get_hue_hist(self, img):
        hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        hue = hsv_img[:,:,0].flatten()
        hue_range = 180
        values, bin_edges = np.histogram(hue, bins=hue_range, density=True)
        return values
