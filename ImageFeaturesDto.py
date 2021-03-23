class ImageFeaturesDto(dict):
    def __init__(self, image_number, vector, labels, ellipse_number):
        dict.__init__(self, image_number = image_number, ellipse_number = ellipse_number, vector = vector, labels = labels)