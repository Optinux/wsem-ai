import matplotlib.pyplot as plt
from keras import backend as K
from keras.layers import (Activation, Conv2D, Dense, Dropout, Flatten,
                          MaxPooling2D)
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
                          
img_w, img_h = 256, 256 # 256x256 optimale performance <> cost ratio
train_dir = 'training_set'
validation_dir = 'validation_set'
nb_train_samples = 4096  # anzahl images training
nb_validation_samples = 2048  # anzahl images validation
epochs = 50  # epochen | iterationen
batch_size = 32 # anzahl images per step (samples / batch size = steps per epoch)

if K.image_data_format() == 'channels_first':   # überprüfen des datenformats und falls nicht stimmt anpassen
    input_shape = (3, img_w, img_h)
else:
    input_shape = (img_w, img_h, 3)


model = Sequential() # 1 input & output pro layer
model.add(Conv2D(32, (2, 2), input_shape=input_shape)) # input bild in stücke aufteilen
model.add(Activation('relu')) # "aktiviert" gewisse künstliche Neuronen nach relu
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (2, 2))) # 2tes Layer
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (2, 2))) # 3tes Layer
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())   
model.add(Dense(64))    # Hidden Layer
model.add(Activation('relu'))
model.add(Dropout(0.5))  # Damit es kein Overfitting / Overlearning gibt
model.add(Dense(1)) # Am Ende nur 1 Neuron als Ouput um zwischen Classic oder Modern zu entscheiden, Aufbau ist quasi wie eine invertierte Sanduhr
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',   # compiler macht magisches zeugs um das model zu optimieren
              optimizer='adam',
              metrics=['accuracy'])


train_datagen = ImageDataGenerator(     # resize, zoom und co. -> input vorbereiten (train)
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_gen = ImageDataGenerator(rescale=1. / 255)     # resize, zoom und co. -> input vorbereiten (test) 

train_gen = train_datagen.flow_from_directory(      # weiter vorbereiten (training)
    train_dir,
    target_size=(img_w, img_h),
    batch_size=batch_size,
    class_mode='binary')

validation_gen = test_gen.flow_from_directory(      # weiter vorbereiten (validation)
    validation_dir,
    target_size=(img_w, img_h),
    batch_size=batch_size,
    class_mode='binary')

history = model.fit_generator(  # füttert die vorbereiten bilder in das model + recorded die history beim trainen des models -> plotten
    train_gen,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_gen,
    validation_steps=nb_validation_samples // batch_size)


# Generiert das Model als Grafik um Aufbau zu verdeutlichen
plot_model(
    model,
    to_file="wsem_model.png",
    show_shapes=True,
    show_dtype=True,
    show_layer_names=True,
    rankdir="TB",
    dpi=64,
    layer_range=None,
    show_layer_activations=True,
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

model.save('wsem_model.h5') # model speichern !!! WICHTIG
