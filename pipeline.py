from collections import namedtuple
from inferenz_smartapp import handwriting_model
from inferenz_bbox import bbox_model

class pipeline:
    def __init__(self):
        '''
        Initialize the pipeline with the models. Unfinished!
        Depending on how the bbox model class is built,
        a lot of things dont have to be here
        e.g cropping, preprocessing, etc. 
        Exact Structure still needs to be discussed
        
        This class can then directly be used for the webapi.
        '''
        self.bbox_model = bbox_model()
        self.handwriting_model = handwriting_model()
        
    def predict_bounding_boxes(self, image):
        boxes = self.bbox_model.inference(image)
        return boxes
    
    def predict_handwriting(self, image):
        result = self.handwriting_model.inference(image)
        return result
    
    def __call__(self, image):
        # Predict bounding boxes and crop the text regions
        cropped_images = self.predict_bounding_boxes(image)
        
        # Extract text from each bounding boxes
        for image in cropped_images:
            predictions = self.predict_handwriting(image)
        
        return predictions
