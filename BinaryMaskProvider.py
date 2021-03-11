import numpy as np
import cv2
class BinaryMaskProvider:
    def Get(self, img):
        '''
        Turn the RGB image into grayscale before
        applying an Otsu threshold to obtain a
        binary segmentation
        '''
    
        blurred_img = cv2.GaussianBlur(img,(15,15),0)
        if (len(blurred_img.shape) > 2): #if many chanells
            blurred_img = cv2.cvtColor(blurred_img, cv2.COLOR_RGBA2GRAY)
        ret, otsu = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        kernel = np.ones((40,40),np.uint8)
        closed_mask = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
        return closed_mask