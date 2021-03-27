import traceback

from ImageSplitting.BinaryMaskProvider import BinaryMaskProvider
from ImageSplitting.EllipseExtractor import EllipseExtractor
from ResourceProviders.ImageProvider import ImageProvider
from ResourceProviders.TrainPathProvider import TrainPathProvider
from ResultWriter import ResultWriter


class ImageNuclearSplitter:
    def __init__(self):
        self.result_writer = ResultWriter()
        self.path_provider = TrainPathProvider()

    def split_train_sample(self, nuclear_image, red, green, blue, yellow, image_number):
        try:

            binary = BinaryMaskProvider().get_binary_mask(nuclear_image)
            extract_ellipses_result = EllipseExtractor().extract_ellipses(binary)
            metas = []

            for ellipse_number, e in enumerate(extract_ellipses_result):
                metas.append(self.result_writer.create_ellipse_meta_dbo(e, image_number, ellipse_number))

                for color in zip(['red', 'green', 'blue', 'yellow'], [red, green, blue, yellow]):
                    full_image, _ = ImageProvider().load_train_sample_by_number(image_number, colors=[color])

                    self.result_writer.process_extracted_ellipse_image(full_image, e, image_number, color, ellipse_number)

            self.result_writer.write_ellipse_metas(metas, image_number)

        except Exception as e:
            self.result_writer.write_exception(traceback.format_exc(), image_number)
