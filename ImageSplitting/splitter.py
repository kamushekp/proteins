from ImageSplitting.ImageNuclearSplitter import ImageNuclearSplitter
from ResourceProviders.ImageProvider import ImageProvider
from ResourceProviders.TrainPathProvider import TrainPathProvider


def _foo(image_number):
    if TrainPathProvider().check_if_image_processed(image_number):
        return

    nuclear_colors = ['blue']
    nuclear_image, _ = ImageProvider().load_train_sample_by_number(image_number, colors=nuclear_colors)

    full_image, _ = ImageProvider().load_train_sample_by_number(image_number)
    red, green, blue, yellow = full_image[:,:,0], full_image[:,:,1], full_image[:,:,2], full_image[:,:,3]

    ImageNuclearSplitter().split_train_sample(nuclear_image, red, green, blue, yellow, image_number)
    return
