import os

import numpy as np
import pandas as pd

from ResourceProviders.TestPathProvider import TestPathProvider


class TestImageDfProvider:

    @staticmethod
    def get() -> np.array:
        sample_submission = pd.read_csv(TestPathProvider().get_sample_submission_file())

        def get_path_for_color(Id, color):
            return f"{os.path.join(TestPathProvider().get_test_image_folder(), Id)}_{color}.png"

        sample_submission['RedPath'] = sample_submission['ID'].apply(lambda x: get_path_for_color(x, 'red'))
        sample_submission['GreenPath'] = sample_submission['ID'].apply(lambda x: get_path_for_color(x, 'green'))
        sample_submission['BluePath'] = sample_submission['ID'].apply(lambda x: get_path_for_color(x, 'blue'))
        sample_submission['YellowPath'] = sample_submission['ID'].apply(lambda x: get_path_for_color(x, 'yellow'))

        return sample_submission
