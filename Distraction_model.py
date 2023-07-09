import numpy as np
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Sequential
from tensorflow.keras import regularizers
from keras.layers import Conv2D, AvgPool2D, Flatten, Dense, InputLayer, BatchNormalization, Dropout

model = Sequential()
model.add(InputLayer(input_shape=(48, 48, 3)))
model.add(Conv2D(64, kernel_size=3,activation='relu',padding='same', kernel_regularizer=regularizers.l1_l2(l1=1e-15, l2=1e-15), bias_regularizer=regularizers.l1_l2(l1=1e-15, l2=1e-15)))
model.add(BatchNormalization(momentum=0.1, epsilon=0.1))
model.add(AvgPool2D(pool_size=(2, 2), padding='same'))
model.add(Flatten())    
model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-15, l2=1e-15), bias_regularizer=regularizers.l1_l2(l1=1e-15, l2=1e-15)))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer="rmsprop",loss='categorical_crossentropy', metrics=['accuracy'])


train_dir = "./PETA Dataset/train/"
val_dir = "./PETA Dataset/val/"
data = ImageDataGenerator()
train = data.flow_from_directory(train_dir,target_size=(48,48),class_mode='categorical',batch_size=100,shuffle=False)
validation = data.flow_from_directory(val_dir,target_size=(48,48),class_mode='categorical',batch_size=100,shuffle=False)
label_map = (train.class_indices)

hist= model.fit(train,epochs=10,verbose=1,validation_data=validation)
print(hist.history.keys())


from sklearn.metrics import classification_report, confusion_matrix
from matplotlib import pyplot

Y_pred = model.predict(train)
y_pred = np.argmax(Y_pred, axis=1)
print('Train-Confusion Matrix')
print(confusion_matrix(train.classes, y_pred))
print('\nTrain-Classification Report')
target_names = ['Distracted', 'Not Distracted']
print(classification_report(train.classes, y_pred, target_names=target_names))
print("\n")

Y_pred = model.predict(validation)
y_pred = np.argmax(Y_pred, axis=1)
print('Validation-Confusion Matrix')
print(confusion_matrix(validation.classes, y_pred))
print('\nValidation-Classification Report')
target_names = ['Distracted', 'Not Distracted']
print(classification_report(validation.classes, y_pred, target_names=target_names))

pyplot.figure(figsize=(15,5))
pyplot.subplot(1, 2, 1)
pyplot.plot(hist.history['loss'], 'r', label='Training loss')
pyplot.plot(hist.history['val_loss'], 'g', label='Validation loss')
pyplot.legend()
pyplot.subplot(1, 2, 2)
pyplot.plot(hist.history['accuracy'], 'r', label='Training accuracy')
pyplot.plot(hist.history['val_accuracy'], 'g', label='Validation accuracy')
pyplot.legend()
pyplot.show()

# model.save("Distract02.h5")

# This is for prediction
path = "./001.png"
img = image.load_img(path, target_size=(48, 48))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = np.vstack([x])
cl = model.predict(x)
print("Image prediction->",np.argmax(cl)) 
print(cl)