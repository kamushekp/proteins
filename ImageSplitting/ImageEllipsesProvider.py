import random

from shapely.geometry import Polygon
from shapely.strtree import STRtree

from ImageSplitting.NuclearEllipse import NuclearEllipse


class ImageEllipsesProvider:
    def __init__(self, contours, image_size):
        self.nuclearEllipses = [NuclearEllipse.build_from_contour(cnt) for cnt in contours]
        self.nuclear_overlap_ellipses = [e.get_overlap_ellipse() for e in self.nuclearEllipses]

        self.transformable_long = set(range(0, len(contours)))
        self.transformable_short = set(range(0, len(contours)))

        self.image_size = image_size / 5.0
        image_polygon = Polygon(((0, 0), (0, image_size), (image_size, image_size), (image_size, 0)))
        self.image_tree = STRtree([image_polygon])

    def build_ellipses(self):
        while True:
            call_conditions = {self._transform_field_long: len(self.transformable_long) > 0,
                               self._transform_field_short: len(self.transformable_short) > 0}
            availible_func = [key for key in call_conditions if call_conditions[key]]
            if len(availible_func) == 0:
                break

            func = random.choice(availible_func)
            func(10)
        return self.nuclearEllipses

    def _get_ellipse(self, index):
        return self.nuclearEllipses[index].get_overlap_ellipse()

    def _try_transform_ellipse(self, index, d_long_length, d_short_length):
        if self.nuclearEllipses[index].long_length > self.image_size or\
                self.nuclearEllipses[index].short_length > self.image_size:
            return False

        def intersection_func(suggestion):
            shape_tree = STRtree([self._get_ellipse(i) for i in range(len(self.nuclearEllipses)) if i != index])
            shape_intersection_query = shape_tree.query(suggestion)
            intersection = [o for o in shape_intersection_query if o.intersects(suggestion)]
            return len(intersection) == 0

        return self.nuclearEllipses[index].try_find_transform(d_long_length, d_short_length, intersection_func,
                                                              self.nuclear_overlap_ellipses[index])

    def _transform_field(self, transformable, d, func):
        untransformable = set()

        for j in transformable:
            sugg = func(j, d)
            if sugg:
                self.nuclearEllipses[j] = sugg
            else:
                untransformable.add(j)
        return untransformable

    def _transform_field_long(self, d_long):
        unstransformable = self._transform_field(self.transformable_long, d_long,
                                                 lambda j, d: self._try_transform_ellipse(j, d, 0))
        self.transformable_long = self.transformable_long - unstransformable
        return len(self.transformable_long) > 0

    def _transform_field_short(self, d_short):
        unstransformable = self._transform_field(self.transformable_short, d_short,
                                                 lambda j, d: self._try_transform_ellipse(j, 0, d))
        self.transformable_short = self.transformable_short - unstransformable
        return len(self.transformable_short) > 0
