from keras.models import load_model
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import numpy as np

from keras.models import load_model

model = load_model('model_saved.h5')

image = load_img('validation_set/modern/modern.jpg', target_size=(512, 512))
img = np.array(image)
img = img / 255.0
img = img.reshape(1, 512, 512, 3)
label = model.predict(img)
print("Predicted Class (0 -> Modern , 1 -> Classic): ", label[0][0])
