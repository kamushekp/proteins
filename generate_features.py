from ImageFeaturesProvider import ImageFeaturesProvider
from ResultWriter import ResultWriter
import warnings
import numpy as np
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

from tqdm import tqdm
        
        
if __name__ == "__main__":
    total = 21806
    for image_number in tqdm(range(total)):
        for dto in ImageFeaturesProvider(image_number).get():
            ResultWriter().write_train_features(dto)