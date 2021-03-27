
import pandas as pd
from cv2 import cv2
import numpy as np
from ResourceProviders.TestPathProvider import TestPathProvider
from ResourceProviders.TrainPathProvider import TrainPathProvider
from TestFeatures.TestClassifierCreator import TestClassifierCreator
from TestFeatures.TestFeaturesProvider import TestFeaturesProvider
from TestFeatures.TestImageDfProvider import TestImageDfProvider
from skimage.util import img_as_ubyte
import base64
import numpy as np
from pycocotools import _mask as coco_mask
import typing as t
import zlib

class SubmissionBuilder:

    def Build(self):
        sample_submission = TestImageDfProvider().get()

        classifire = TestClassifierCreator().create_classifier()
        sample_submission = sample_submission[:3]

        sample_submission['PredictionString'] = sample_submission.apply(lambda x: self.get_prediction_suffix(classifire, x), axis=1)

        columns_to_drop = ['RedPath', 'GreenPath', 'BluePath', 'YellowPath']
        sample_submission.drop(columns_to_drop, inplace=True, axis=1)

        sample_submission.to_csv('F:\datasets\hpa-single-cell-image-classification\submission.csv', index=False)

    def get_prediction_suffix(self, classifire, first):
        images = [cv2.imread(full_path, cv2.IMREAD_UNCHANGED) for full_path in
                  [first['RedPath'], first['GreenPath'], first['BluePath']]]
        fullImage = img_as_ubyte(np.stack(images, axis=-1))
        blue_image = fullImage[:, :, 2]
        prediction_string = ""
        for feature, contour_binary in TestFeaturesProvider().get(blue_image, fullImage):
            encoding_string = str(self.encode_binary_mask(contour_binary))
            prediction = classifire.predict([feature])
            prediction_string += f"{prediction[0]} 1.0 {encoding_string}"
        return prediction_string

    def encode_binary_mask(self, mask: np.ndarray) -> t.Text:
        """Converts a binary mask into OID challenge encoding ascii text."""

        # check input mask --
        if mask.dtype != np.bool:
            raise ValueError(
                "encode_binary_mask expects a binary mask, received dtype == %s" %
                mask.dtype)

        mask = np.squeeze(mask)
        if len(mask.shape) != 2:
            raise ValueError(
                "encode_binary_mask expects a 2d mask, received shape == %s" %
                mask.shape)

        # convert input mask to expected COCO API input --
        mask_to_encode = mask.reshape(mask.shape[0], mask.shape[1], 1)
        mask_to_encode = mask_to_encode.astype(np.uint8)
        mask_to_encode = np.asfortranarray(mask_to_encode)

        # RLE encode mask --
        encoded_mask = coco_mask.encode(mask_to_encode)[0]["counts"]

        # compress and base64 encoding --
        binary_str = zlib.compress(encoded_mask, zlib.Z_BEST_COMPRESSION)
        base64_str = base64.b64encode(binary_str)
        return base64_str.decode('ascii')

if __name__ == "__main__":
    print(SubmissionBuilder().Build())