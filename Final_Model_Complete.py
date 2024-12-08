# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 15:48:32 2022

@author: Farahan
"""

# Importing Necessary Libraries

import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from nn_model import my_model
import tensorflow as tf
from sklearn import preprocessing
import time


# Loading the file that contains locations and labels file that has all the labels with respect to each file. 
all_locs = np.load("D:/project_ascent_recognition/dataset/all_files_loc.npy", allow_pickle = True)

all_labels= np.load("D:/project_ascent_recognition/dataset/all_lables_loc.npy", allow_pickle = True)

# Normalize the dataset 
def data_normalization(data):
    data = data.reshape(-1,1)
    data = preprocessing.StandardScaler().fit_transform(data)
    return data



# Extract features and labels to make a complete dataset from all samples

feat = []
labels = []
start_time = time.time()

for loc in range(len(all_locs)-10):       
        data, sr= librosa.load(all_locs[loc], mono = False,offset=1.0, duration=1.0)
        print('Obtaining Sample:  ', loc)
        if len(data) >= 22050: # Minimum samples accepted 
            mfcc = librosa.feature.mfcc(y=data, sr=sr, hop_length=1024, htk=True)
            
            data = data_normalization(data)
            mfcc  = np.asarray(mfcc, dtype=np.float32)
            mfcc = np.expand_dims(mfcc, axis=2)
            feat.append(mfcc)
            
            label = all_labels[loc]
            labels.append(label)
        
# Convert the features to array that can be passed to the NN           
X = np.asarray(feat) 

labels = np.asarray(labels)
Y = to_categorical(labels) # One Hot encoding 

# Initialize the model 
model = my_model()

# Train test split 
train_X,valid_X,train_label,valid_label = train_test_split(X, Y, test_size=0.20, random_state=42)

# Training the model 
history = model.fit(train_X, train_label, verbose=1,epochs=500,validation_data=(valid_X, valid_label))

end_time = time.time()
total_time = start_time - end_time
print("Time Training: ", total_time)


# Ploting the model

plt.plot(history.history['loss'])
plt.show()
plt.plot(history.history['accuracy'])
plt.show()

plt.plot(history.history['val_loss'])
plt.show()
plt.plot(history.history['val_accuracy'])
plt.show()

# Save loss and accuracy values
np.save("val_accuracy.npy", history.history['val_accuracy'],allow_pickle=True )
np.save("val_loss.npy", history.history['val_loss'],allow_pickle=True )
np.save("accuracy.npy", history.history['accuracy'],allow_pickle=True )
np.save("loss.npy", history.history['loss'],allow_pickle=True )


# Save the model weights
model.save_weights('Model_Weights1.h5')

