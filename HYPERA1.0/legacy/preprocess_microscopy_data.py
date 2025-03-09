import os
class PreprocessMicroscopyData:
    def __init__(self):
        self.image_data_dir = os.path.join(os.getcwd(), "HYPERA/BBBC039 /images/")
        self.label_data_dir = os.path.join(os.getcwd(), "HYPERA/BBBC039 masks/")
        self.training_data_filename =  os.path.join(os.getcwd(), "HYPERA/BBBC039/metadata/training.txt")
        self.validation_data_filename =  os.path.join(os.getcwd(), "HYPERA/BBBC039/metadata/validation.txt")
    
    def _createDataDictionaries(self, filename):
        training_data_dicts = []
        with open(filename, 'r') as file:
            # Read each line in the file
            for line in file:
                label_file_name = line.strip()
                image_file_name = line.split('.')[0]+".tif"
                training_data_dicts.append({
                    "image": self.image_data_dir + image_file_name,
                    "label": self.label_data_dir + label_file_name,
                })
        return training_data_dicts
    
    def getFiles(self, filename):
        return self._createDataDictionaries(filename)[:]
    
    def getTrainingFiles(self):
        return self.getFiles(self.training_data_filename)
    
    def getValidationFiles(self):
        return self.getFiles(self.validation_data_filename)
