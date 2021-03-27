import os
import pathlib


class TestPathProvider:
    def __init__(self):
        self.dataset_folder = "F:\datasets\hpa-single-cell-image-classification"

    def get_dataset_folder(self):
        return self.dataset_folder

    def get_test_image_folder(self):
        return os.path.join(self.get_dataset_folder(), "test")

    def get_sample_submission_file(self):
        return os.path.join(self.get_dataset_folder(), "sample_submission.csv")

    def get_train_image_path(self, image_id):
        return os.path.join(self.get_test_image_folder(), image_id)
