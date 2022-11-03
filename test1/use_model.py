from keras.models import load_model
from keras.utils import load_img
from keras.utils import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import numpy
import os
from keras.models import load_model
import matplotlib.pyplot as plt

accuracy_insg = 0  # gesamte accuracy zu beginn auf null
accuracy_plot = []  # leere plot list

model = load_model('wsem_model.h5')

# erstellt liste aller files im ausgewählten ordner -> classic oder modern, um auf diese zu predicten
path = r'G:\GitHub\wsem-ai\test1\validation_set\classic'  # !!!WICHTIG
list_files = []  # leere file list
for root, dirs, files in os.walk(path):
    for file in files:
        list_files.append(os.path.join(root, file))

i = 1
for name in list_files:
    image = load_img(  # resize and reshape entsprechend des models
        name, target_size=(256, 256))
    img = numpy.array(image)
    img = img / 255.0
    img = img.reshape(1, 256, 256, 3)
    label = model.predict(img)
    accuracy = label[0][0]
    print(i, ". Predicte Bild... (0 -> Classic , 1 -> Modern): ", accuracy)

    # insgesamte accuracy -> accuracy alle predicteten bilder zusammen
    accuracy_insg = accuracy_insg + accuracy
    # fürs spätere plotten ein array befüllen
    accuracy_plot.append(accuracy)
    # print(accuracy_plot)    # debug

    i += 1

    if i > 400:  # wie oft er predicten soll                                              !!!WICHTIG
        i = i - 1  # -1 damit auch die richtige anzahl geteilt wird
        break

print("Insgesamte Accuracy von", i, "predicteten Bildern: ", accuracy_insg / i)

# Prediction als witzige Graphen plotten, zeigt Accuracy der Prediction aller predicteten Bilder in einem Graphen
# print(accuracy_plot)    # debug
anzahl_predictions = range(0, i)
# print(anzahl_predictions)   # debug
plt.plot(anzahl_predictions, accuracy_plot)
plt.title(accuracy_insg / i)
plt.suptitle('Accuracy aller Predictions, wobei durchschnittliche Accuracy: ')
plt.ylabel('Accuracy | Genauigkeit')
plt.xlabel('Predictions | Anzahl an Vorhersagen')
plt.show()
