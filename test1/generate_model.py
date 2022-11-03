from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

import matplotlib.pyplot as plt
from keras.utils import plot_model


# festlegen auf 256x256 da optimale performance zu aufwand ratio
img_width, img_height = 256, 256

train_data_dir = 'training_set'
validation_data_dir = 'validation_set'
nb_train_samples = 2048  # anzahl an training images 1024
nb_validation_samples = 1024  # anzahl an validation images 512
epochs = 10  # epochen aka wie viele iterationen 10

# samples geteilt durch batch size -> wie viel pro schritt in der epoche durch network geboxt werden 32
batch_size = 32

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    # resize falls nicht stimmt (stimmt eigentlich nie)
    input_shape = (img_width, img_height, 3)


# aufbau des models
model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)


# Generiert das Model als Grafik um Aufbau zu verdeutlichen
plot_model(
    model,
    to_file="wsem_model.png",
    show_shapes=False,
    show_dtype=False,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=False,
    dpi=96,
    layer_range=None,
    show_layer_activations=False,
)


# Trainingsverlauf als witzige Graphen plotten, zeigt Accuracy von Training und Validation
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy im Verlauf des Trainings')
plt.ylabel('Accuracy | Genauigkeit')
plt.xlabel('Epochs | Iterationen')
plt.legend(['train', 'validate'], loc='upper left')
plt.show()

# Trainingsverlauf als witzige Graphen plotten, zeigt Loss von Training und Validation
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss im Verlauf des Trainings')
plt.ylabel('Loss | "wie falsch war die Prediction"')
plt.xlabel('Epochs | Iterationen')
plt.legend(['train', 'validate'], loc='upper left')
plt.show()

model.save('wsem_model.h5')
