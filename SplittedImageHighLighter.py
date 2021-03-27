import cv2
import numpy as np

from ImageSplitting.BinaryMaskProvider import BinaryMaskProvider
from ImageSplitting.NuclearEllipse import NuclearEllipse

from skimage.util import img_as_bool

class SplittedImageHighLighter:

    def highlight_ellipse(self, ellipse_image, ellipse_meta):
        ellipse = NuclearEllipse((ellipse_meta['x'], ellipse_meta['y']), ellipse_meta['long'], ellipse_meta['short'], ellipse_meta['angle'])
        ellipse_exterior = np.asarray(ellipse.initial_ellipse.exterior.xy, dtype=int).T
        stencil = np.zeros(ellipse_image.shape).astype(ellipse_image.dtype)
        contours = [ellipse_exterior]
        color = [255, 255, 255]
        cv2.fillPoly(stencil, contours, color)
        highlighted = cv2.bitwise_and(ellipse_image, stencil)
        return highlighted

    def highlight_ellipse_on_full_image(self, full_image_shape, ellipse_image, minx, miny, maxx, maxy):
        import matplotlib.pyplot as plt
        stencil = np.zeros(full_image_shape).astype(ellipse_image.dtype)
        stencil[int(miny):int(maxy), int(minx):int(maxx)] = ellipse_image

        binary_mask = BinaryMaskProvider().get_binary_mask(stencil)

        contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True) for cnt in contours]
        areas = np.asarray([cv2.contourArea(c) for c in contours])
        index_of_max = np.argmax(areas)

        empty_image = np.zeros(full_image_shape[:2]).astype(ellipse_image.dtype)

        cv2.drawContours(empty_image, [contours[index_of_max]], 0, (255), cv2.FILLED)

        empty_image = img_as_bool(empty_image)
        return empty_image
