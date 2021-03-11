import json
from EllipseMetaDbo import EllipseMetaDbo
from os import path
import pathlib
from PathProvider import PathProvider
import cv2

class ResultWriter:
    def __init__(self):
        self.path_provider = PathProvider()
    
    def process_extracted_ellipse_image(self, image, params, image_number, color, ellipse_number):
        minx, miny, maxx, maxy = params[1]
        cropped_image = image[int(miny):int(maxy), int(minx):int(maxx)]
        
        image_path = self.path_provider.get_splitted_image_path(image_number, color, ellipse_number)
        
        save_result = cv2.imwrite(image_path, cropped_image)
        
        if not save_result:
             raise Exception(f"Could not write image to path {image_path}")
        
    def create_ellipse_meta_dbo(self, params, image_number, ellipse_number):        

        ellipse_params = params[0]
        dbo = EllipseMetaDbo(image_number, ellipse_number,
                             ellipse_params[0], ellipse_params[1],
                             ellipse_params[2], ellipse_params[3],
                             ellipse_params[4])        
        return dbo
    
    def write_ellipse_metas(self, dbos, image_number):
        json_path =  self.path_provider.get_meta_path(image_number)
        with open(json_path, 'w+', encoding='utf-8') as f:
            json.dump(dbos, f, ensure_ascii=False, indent=4)
            
    def write_exception(self, message, image_number):
        exception_path = self.path_provider.get_error_path(image_number)
        with open(exception_path, 'w+', encoding='utf-8') as f:
            f.write(message)