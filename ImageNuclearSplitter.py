import csv
from os import path
from ImageProvider import ImageProvider
from BinaryMaskProvider import BinaryMaskProvider
import cv2
import numpy as np
from Field import Field
from EllipseExtractor import EllipseExtractor
from ResultWriter import ResultWriter
from PathProvider import PathProvider
import traceback

class ImageNuclearSplitter:
    def __init__(self):        
        self.result_writer = ResultWriter()
        self.path_provider = PathProvider()
    
    def split_train_sample(self, image_number):
        try:
            if self.path_provider.check_if_image_processed(image_number):
                return
            
            nuclear_colors = ['blue']
            image, labels = ImageProvider().load_train_sample_by_number(image_number, colors = nuclear_colors)
            
            binary = BinaryMaskProvider().Get(image)
            extract_ellipses_result = EllipseExtractor().extract_ellipses(binary)
            metas = []
            
            for ellipse_number, e in enumerate(extract_ellipses_result):            
                metas.append(self.result_writer.create_ellipse_meta_dbo(e, image_number, ellipse_number))

                for color in ['red', 'green', 'blue', 'yellow']: 
                    image, labels = ImageProvider().load_train_sample_by_number(image_number, colors = [color])
                    
                    self.result_writer.process_extracted_ellipse_image(image, e, image_number, color, ellipse_number)
                    
            self.result_writer.write_ellipse_metas(metas, image_number)
            
        except Exception as e:
            self.result_writer.write_exception(traceback.format_exc(), image_number)
