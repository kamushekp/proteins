from ImageSplitting.EllipseImageFeatureExtractor import EllipseImageFeatureExtractor
from SplittedImageHighLighter import SplittedImageHighLighter


class ImageFeaturesProvider:

    def get(self, ellipse_image, meta):
        try:
            high_lighted_ellipse_image = SplittedImageHighLighter().highlight_ellipse(ellipse_image, meta)
            feature = EllipseImageFeatureExtractor().get_hue_hist(high_lighted_ellipse_image).tolist()
            yield feature

        except Exception as e:
            pass
