# -*- coding: utf-8 -*-
"""
Created on Thu May 28 11:07:30 2020

@author: erdem
"""

import numpy as np 
import os
for dirname, _, filenames in os.walk('/kaggle_dataset'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
import cv2 
import os
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix
import random

uniq_labels=['yes','no']
directory="./kaggle_dataset"

 

def load_images(directory,uniq_labels):
    images = []
    labels = []
    for idx, label in enumerate(uniq_labels):
        for file in os.listdir(directory + "/" + label):
            filepath = directory + "/" + label + "/" + file
            image = cv2.resize(cv2.imread(filepath), (128, 128))
            images.append(image)
            labels.append(idx)
    images = np.array(images)
    labels = np.array(labels)        
    return images,labels  

def draw_confusion_matrix(true,preds):
    # Compute confusion matrix
    conf_matx = confusion_matrix(true, preds)
    print("Confusion matrix : \n")
    sns.heatmap(conf_matx, annot=True,annot_kws={"size": 12},fmt='g', cbar=False, cmap="viridis")
    plt.show()
    return conf_matx
#load images and labels
images,labels=load_images(directory,uniq_labels)

#preprosessing (one hot encoding for labels & scaling for images)
labels = keras.utils.to_categorical(labels)
images = images.astype("float32")/ 255.0

images.shape


shuffler = list(zip(images, labels))

random.shuffle(shuffler)

Images, Labels = zip(*shuffler)
images1 = []
labels1 = []
for image in Images:
    images1.append(image)
for label in Labels:
    labels1.append(label)

images1 = np.array(images1,dtype=np.float32)
labels1 = np.array(labels1,dtype=np.float32)

fig = plt.figure(figsize=(10, 10))
i=0

for img in enumerate(images1):
    if i<12:
      plt.title(labels1[i])
      plt.imshow(img[1])
      i = i + 1
      plt.show()
  



kf = KFold(n_splits=5, random_state=None, shuffle=False)
for train_index, test_index in kf.split(images1):
    model = keras.models.Sequential()
    model.add( keras.layers.Conv2D(filters=32, kernel_size=5, strides=1, padding="same", activation="relu", input_shape= (128,128,3)))#64
    model.add(keras.layers.MaxPool2D(pool_size=2))
    model.add(keras.layers.Dropout(0.4))
    model.add( keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same", activation="relu"))#128
    model.add(keras.layers.MaxPool2D(pool_size=2))
    model.add(keras.layers.Dropout(0.4))
    model.add( keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding="same", activation="relu"))
    model.add(keras.layers.MaxPool2D(pool_size=2))
    model.add(keras.layers.Dropout(0.4))
    
    model.add(keras.layers.Flatten())
    layer0 = keras.layers.Dense(512, activation="relu",kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l2(0.01))#512
    layer1 = keras.layers.Dense(128, activation="relu",kernel_initializer="he_normal",kernel_regularizer=keras.regularizers.l2(0.01))
    layer_output = keras.layers.Dense(2, activation="sigmoid",kernel_initializer="glorot_uniform")#sigmoid, softmax
    model.add(layer0)
    model.add(keras.layers.Dropout(0.2))
    model.add(layer1)
    model.add(keras.layers.Dropout(0.2))
    model.add(layer_output)
    
    # The model’s summary() method displays all the model’s layers
      
    
    opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(loss="binary_crossentropy", optimizer= opt, metrics=["accuracy"])#categorical_crossentropy,binary_crossentropy
    
    #print("TRAIN:", train_index, "\n" , "TEST:", test_index)
    X_train, X_test = images1[train_index], images1[test_index]
    y_train, y_test = labels1[train_index], labels1[test_index]
    history = model.fit(X_train, y_train, epochs=30, batch_size=18, validation_split=0.2)
    """
    pd.DataFrame(history.history["acc"]).plot(figsize=(12, 8))
    pd.DataFrame(history.history["val_acc"]).plot(figsize=(12, 8))
    plt.grid(True)
    plt.gca().set_ylim(0, 1) # set the vertical range to [0-2]
    plt.show()"""
    
    
    model_evaluate = model.evaluate(X_test, y_test)
    print("Loss     : ",model_evaluate[0])
    print("accuracy : ",model_evaluate[1])
    
    #prediction for test images
    y_pred = model.predict_classes(X_test)
    #real values for test images
    y_test_=np.argmax(y_test, axis=1)
    
    
    print("Classification report : \n",classification_report(y_test_, y_pred,target_names=["Pos", "Neg"]))
    
    cm = confusion_matrix(y_test_, y_pred)
    cm
    
    con_mat = draw_confusion_matrix(y_test_, y_pred)
    
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
    
print(model.summary())   


