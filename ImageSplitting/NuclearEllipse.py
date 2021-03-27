import cv2
import numpy as np
import shapely.affinity
from shapely.geometry.point import Point


class NuclearEllipse:

    @staticmethod
    def build_from_contour(contour):
        (center, (long_length, short_length), angle) = cv2.fitEllipse(contour)
        long_length /= 2.0
        short_length /= 2.0
        return NuclearEllipse(center, long_length, short_length, angle)

    def get_overlap_ellipse(self):
        return NuclearEllipse.create_ellipse(self.center[0], self.center[1], self.long_length, self.short_length,
                                             self.angle)

    def __init__(self, center, long_length, short_length, angle):
        self.center = center
        self.long_length = long_length
        self.short_length = short_length
        self.angle = angle
        self.initial_ellipse = self.get_overlap_ellipse()

    def create_d_ellipse(self, d_long_length, d_short_length, d_angle):
        return NuclearEllipse(self.center, self.long_length + d_long_length, self.short_length + d_short_length,
                              self.angle + d_angle)

    @staticmethod
    def create_ellipse(x, y, long_length, short_length, angle):
        circle = shapely.geometry.Point((x, y)).buffer(1)
        ellipse = shapely.affinity.scale(circle, long_length, short_length)
        ellipse_rotated = shapely.affinity.rotate(ellipse, angle)
        return ellipse_rotated

    @staticmethod
    def get_d_angles():
        return np.arange(0, 100, 10)

    def try_transform(self, d_long_length, d_short_length, d_angle, nuclear_overlap_ellipse):
        if self.long_length + d_long_length <= 0 or self.short_length + d_short_length <= 0:
            return False
        suggestion = NuclearEllipse.create_ellipse(self.center[0], self.center[1], self.long_length + d_long_length,
                                                   self.short_length + d_short_length, self.angle + d_angle)
        if suggestion.contains(nuclear_overlap_ellipse):
            return suggestion
        return False

    def try_find_transform(self, d_long_length, d_short_length, intersection_func, nuclear_overlap_ellipse):
        for d_angle in NuclearEllipse.get_d_angles():
            suggestion = self.try_transform(d_long_length, d_short_length, d_angle, nuclear_overlap_ellipse)
            if suggestion and intersection_func(suggestion):
                return self.create_d_ellipse(d_long_length, d_short_length, d_angle)

        return False
