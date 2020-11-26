import numpy as np
from keras.models import load_model
from keras.preprocessing import image

class xray:
    def __init__(self,filename):
        self.filename =filename


    def predictionxray(self):
        # load model
        model = load_model('Lungs.h5')

        # summarize model
        #model.summary()
        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)

        if result[0][0] == 1:
            prediction = 'Normal'
            return [{ "x-RAY Report" : prediction}]
        else:
            prediction = 'Pneumonia'
            return [{ "x-RAY Report" : prediction}]
