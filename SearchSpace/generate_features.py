import warnings

import numpy as np

from ImageFeaturesDto import ImageFeaturesDto
from ImageFeaturesProvider import ImageFeaturesProvider
from ImageSplitting.ResultWriter import ResultWriter
from ImageSplitting.SplittedImagesProvider import SplittedImagesProvider
from ResourceProviders.ImageProvider import ImageProvider

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

from tqdm import tqdm

if __name__ == "__main__":
    total = 21806
    for image_number in tqdm(range(total)):

        image_provider = SplittedImagesProvider(image_number)
        labels = ImageProvider().load_train_label_by_number(image_number)

        for ellipse_number in range(image_provider.get_ellipses_count()):
            colored_ellipse_images = [image_provider.get_ellipse_img(ellipse_number, color) for color in
                                      ['red', 'green', 'blue']]
            ellipse_image = np.stack(colored_ellipse_images, axis=-1)

            meta = image_provider.get_ellipse_meta(ellipse_number)
            feature = ImageFeaturesProvider().get(ellipse_image, meta)

            dto = ImageFeaturesDto(image_number, feature, labels, ellipse_number)

            ResultWriter().write_train_features(dto)
