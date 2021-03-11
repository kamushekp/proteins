from Field import Field
from ContourFinder import ContourFinder
class EllipseExtractor:
    
    def extract_ellipses(self, binary):
        if not (len(binary.shape) == 2 and binary.shape[0] == binary.shape[1]):
            raise ArgumentException()            
        image_size = binary.shape[0]
        
        contours = ContourFinder().find_contours(binary)
        field = Field(contours, image_size)
        ellipses = field.build_ellipses()
        result = []
        
        for ellipse in ellipses:
            minx, miny, maxx, maxy = ellipse.get_overlap_ellipse().bounds
            center = (ellipse.center[0] - minx, ellipse.center[1] - miny)
            long_length, short_length = ellipse.long_length, ellipse.short_length
            angle = ellipse.angle
            minx, miny, maxx, maxy = [self.norm_index(value, image_size) for value in (minx, miny, maxx, maxy)]
            
            ellipse_params = ((center[0], center[1], long_length, short_length, angle), (minx, miny, maxx, maxy))
            result.append(ellipse_params)
        return result
    
    def norm_index(self, value, image_size):
        if value < 0:
            return 0
        if value > image_size - 1:
            return  image_size - 1
        return value