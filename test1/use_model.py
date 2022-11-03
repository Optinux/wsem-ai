from keras.models import load_model
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import numpy
from keras.models import load_model

model = load_model('wsem_model.h5')

i = 1
while i < 25:  # 25x bilder predicten und dann den gesamnt value davon errechnen

    image = load_img(
        'validation_set/modern/idString=[0K4zaXF4bu]_year=[1851-1900].jpg', target_size=(256, 256))
    img = numpy.array(image)
    img = img / 255.0
    img = img.reshape(1, 256, 256, 3)
    label = model.predict(img)
    print(i, ". Predicted Class (0 -> Classic , 1 -> Modern): ", label[0][0])

    i += 1

print("Insgesamte Accuracy: ", 25 / i)
