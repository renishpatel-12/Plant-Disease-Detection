# Importing Necessary Libraries
import tensorflow as tf
import numpy as np
from PIL import Image


# Cleaning image    
def clean_image(image):
    image = np.array(image)
    
    # Resizing the image
    image = np.array(Image.fromarray(
        image).resize((512, 512), Image.LANCZOS))
        
    # Adding batch dimensions to the image
    # You are setting :3, that's because sometimes user upload 4 channel image,
    image = image[np.newaxis, :, :, :3]
    # So we just take first 3 channels
    
    return image
    
    
def get_prediction(model, image):
    # Normalize the image
    image = image.astype(np.float32) / 255.0
    
    # Predict from the image
    predictions = model.predict(image)
    predictions_arr = np.argmax(predictions, axis=1)[0]
    
    return predictions, predictions_arr
    

# Making the final results 
def make_results(predictions, predictions_arr):
    
    result = {}
    if int(predictions_arr) == 0:
        result = {"status": " is Healthy ",
                    "prediction": f"{predictions[0][0]*100:.1f}%"}
    elif int(predictions_arr) == 1:
        result = {"status": ' has Multiple Diseases ',
                    "prediction": f"{predictions[0][1]*100:.1f}%"}
    elif int(predictions_arr) == 2:
        result = {"status": ' has Rust ',
                    "prediction": f"{predictions[0][2]*100:.1f}%"}
    elif int(predictions_arr) == 3:
        result = {"status": ' has Scab ',
                    "prediction": f"{predictions[0][3]*100:.1f}%"}
    return result   
