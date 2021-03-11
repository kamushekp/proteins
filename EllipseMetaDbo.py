class EllipseMetaDbo(dict):
    def __init__(self, image_number, ellipse_number, x, y, long, short, angle):
        dict.__init__(self,
                      image_number = image_number,
                      ellipse_number = ellipse_number,
                      x = x,
                      y = y,
                      long = long,
                      short = short,
                      angle = angle)