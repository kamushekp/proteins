import os
import pathlib

class PathProvider:
    def __init__(self):        
        self.dataset_folder = "F:\datasets\hpa-single-cell-image-classification"
        self.splitted_result_folder = os.path.join(self.dataset_folder, 'train_splitted')
        
    def get_dataset_folder(self):
        return self.dataset_folder
    
    def get_training_image_folder(self):
        return os.path.join(self.get_dataset_folder(), "train")
    
    def get_train_df_path(self):
        return os.path.join(self.get_dataset_folder(), "train.csv")
    
    def get_train_image_path(self, image_id):
        return os.path.join(self.get_training_image_folder(), image_id)
        
    def check_if_image_processed(self, image_number):
        folder = self.get_splitted_image_folder(image_number, False)
        return os.path.exists(folder)
    
    def get_splitted_image_folder(self, image_number, need_create):
        folder = os.path.join(self.splitted_result_folder, str(image_number))
        if need_create:
            pathlib.Path(folder).mkdir(parents=True, exist_ok=True)     
        return folder
    
    def get_splitted_image_path(self, image_number, color, ellipse_number):
        image_to_save_name = f'{color}_{ellipse_number}.png'
        folder = self.get_splitted_image_folder(image_number, True)
        
        pathlib.Path(folder).mkdir(parents=True, exist_ok=True)        
        return os.path.join(folder, image_to_save_name)
    
    def get_meta_path(self, image_number):
        return os.path.join(self.get_splitted_image_folder(image_number, True), 'meta.json')
    
    def get_error_path(self, image_number):
        return os.path.join(self.get_splitted_image_folder(image_number, True), 'errors.txt')