from inferenz_smartapp import handwriting_model
# from inferenz_yolo...doesnt exist yet..  import bbox model

class pipeline:
    def __init__(self):
        '''
        Initialize the pipeline with the models. Unfinished!
        Depending on how the bbox model class is built,
        a lot of things dont have to be here
        e.g cropping, preprocessing, etc. 
        Structure still needs to be discussed
        
        This class can then be used for the webapi.
        '''
        # self.text_extraction_model = text_extraction_model()
        self.handwriting_model = handwriting_model()
        
        
    def predict_bounding_boxes(self, image):
        # Preprocess the image
        image = self.preprocess_image(image)
        # Predict the bounding boxes
        boxes = self.text_extraction_model.predict(image)
        return boxes
    
    # def cropping_function---> to crop the bounding boxes from the image if not already in the bbox model class
    
    def predict_handwriting(self, image):
        result = self.handwriting_model.inference(image)
        return result
    
    def __call__(self, image):
        # Predict bounding boxes
        cropped_images = self.predict_bounding_boxes(image)
        
        # if cropping is done here then
        # cropped_images = self.cropping_function(image, boxes)
        
        # Extract text from the bounding boxes
        predictions = self.predict_handwriting(cropped_images)
        return predictions
