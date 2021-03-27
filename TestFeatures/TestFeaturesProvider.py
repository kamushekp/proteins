from ImageSplitting.BinaryMaskProvider import BinaryMaskProvider
from ImageSplitting.EllipseExtractor import EllipseExtractor
from ImageSplitting.EllipseImageFeatureExtractor import EllipseImageFeatureExtractor
from ImageSplitting.EllipseMetaDbo import EllipseMetaDbo
from SplittedImageHighLighter import SplittedImageHighLighter


class TestFeaturesProvider:

    def get(self, nuclear_image, image):
        binary = BinaryMaskProvider().get_binary_mask(nuclear_image)
        extract_ellipses_result = EllipseExtractor().extract_ellipses(binary)

        for ellipse_params in extract_ellipses_result:
            (center_x, center_y, long_length, short_length, angle), (minx, miny, maxx, maxy) = ellipse_params
            ellipse_image = image[int(miny):int(maxy), int(minx):int(maxx)]
            meta = EllipseMetaDbo(image_number=None, ellipse_number=None, x=center_x, y=center_y, long=long_length,
                                  short=short_length, angle=angle)

            high_lighted_ellipse_image = SplittedImageHighLighter().highlight_ellipse(ellipse_image, meta)
            feature = EllipseImageFeatureExtractor().get_hue_hist(high_lighted_ellipse_image).tolist()

            contoured_image = SplittedImageHighLighter().highlight_ellipse_on_full_image(image.shape, ellipse_image,
                                                                                         minx, miny, maxx, maxy)
            yield feature, contoured_image
