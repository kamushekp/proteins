import cv2
import numpy as np


class ContourFinder:
    @staticmethod
    def find_contours(binary_mask: np.array):

        contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True) for cnt in contours]
        areas = np.asarray([cv2.contourArea(c) for c in contours])
        points_count = np.asarray([len(c) for c in contours])
        suitable_by_area = (areas > 200).nonzero()[0]
        suitable_by_ellipse_possibility = (points_count > 5).nonzero()[0]
        suitable = list(set(suitable_by_area).intersection(set(suitable_by_ellipse_possibility)))

        params_filtered = np.take(contours, suitable, axis=0)

        def border_cond(contour):
            # works properly only for square images!!!
            return np.any(contour == 0) or np.any(contour == binary_mask.shape[0] - 1)

        return [contour for contour in params_filtered if not border_cond(contour)]
