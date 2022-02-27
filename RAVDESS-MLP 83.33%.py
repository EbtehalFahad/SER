#!/usr/bin/env python
# coding: utf-8

# In[97]:


import soundfile
import numpy as np
import librosa
import pickle
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix

# all emotions on RAVDESS dataset
allemotion = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}
gendertype = {
    "01": "m",
    "03": "m",
    "05": "m",
    "07": "m",
    "09": "m",
    "11": "m",
    "13": "m",
    "15": "m",
    "17": "m",
    "19": "m",
    "21": "m",
    "23": "m",
    "02": "f",
    "04": "f",
    "06": "f",
    "08": "f",
    "10": "f",
    "12": "f",
    "14": "f",
    "16": "f",
    "18": "f",
    "20": "f",
    "22": "f",
    "24": "f"
}
# Emotions to observe
observed_emotions={
    "angry",
    "sad",
    "neutral",
    "happy"
}


# In[98]:


#Extract features (mfcc, chroma, mel, contrast, tonnetz) from a sound file
def extract_features(file_name, **kwargs):
    """
    Extract feature from audio file `file_name`
        Features supported:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            - Contrast (contrast)
            - Tonnetz (tonnetz)
        e.g:
        `features = extract_feature(path, mel=True, mfcc=True)`
    """
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma or contrast:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result = np.hstack((result, mel))
        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, contrast))
        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
            result = np.hstack((result, tonnetz))
    return result


# In[99]:


# Load the data
def load_data(g, test_size=0.2):
    x, y = [], []
    try :
        for file in glob.glob("ravdess/Actor_*/*.wav"):
            # get the base name of the audio file
            basename = os.path.basename(file)
            # get the emotion label 
            emotion = allemotion[basename.split("-")[2]]
            # get the gender label
            gender = gendertype[basename.split("-")[6].split(".")[0]]
            # we allow only observed_emotions we set for both gender
            if emotion not in observed_emotions:
                continue
            if g != gender and g != "both":
                continue
            # extract speech features
            features = extract_features(file, mfcc=True, chroma=True, mel=True)
            # add to data
            x.append(features)
            y.append(emotion)
    except :
         pass
    # split the data to training and testing and return it
    return train_test_split(np.array(x), y, test_size=test_size, random_state=7)


# In[100]:


#loading_data
x_train,x_test,y_train,y_test = load_data("both", test_size=0.25)

# print some details
print("Both:")
# number of samples in training data
print("[+] Number of training samples:", x_train.shape[0])
# number of samples in testing data
print("[+] Number of testing samples:", x_test.shape[0])
# number of features used
# this is a vector of features extracted 
# using utils.extract_features() method
print("[+] Number of features:", x_train.shape[1])

#________________________________________________________________________
#FIRST MODEL
model_params = {
    'alpha': 0.01,
    'batch_size': 256,
    'epsilon': 1e-08, 
    'hidden_layer_sizes': (300,), 
    'learning_rate': 'adaptive', 
    'max_iter': 500, 
}
    
# initialize Multi Layer Perceptron classifier
# with best parameters ( so far )
model = MLPClassifier(**model_params)

# train the model
print("[*] Training the model...")
model.fit(x_train, y_train)

# predict 25% of data to measure how good we are
y_pred = model.predict(x_test)

# calculate the accuracy
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

print("Accuracy first model: {:.2f}%".format(accuracy*100))
#print(classification_report(y_test,y_pred))
#print(confusion_matrix(y_test,y_pred))

#________________________________________________________________________
#SECOND MODEL
m_params = {
    'alpha': 0.01,
    'batch_size': 200,
    'epsilon': 1e-08, 
    'hidden_layer_sizes': (300,), 
    'learning_rate': 'adaptive', 
    'max_iter': 500, 
}
# initialize Multi Layer Perceptron classifier
# with best parameters ( so far )
m1 = MLPClassifier(**m_params)

# train the model
print("[*] Training the model...")
m1.fit(x_train, y_train)

# predict 25% of data to measure how good we are
y_p = m1.predict(x_test)

# calculate the accuracy
accuracy = accuracy_score(y_true=y_test, y_pred=y_p)

print("Accuracy second model: {:.2f}%".format(accuracy*100))
#print(classification_report(y_test,y_p))
#print(confusion_matrix(y_test,y_p))


# In[101]:


#loading_data
print("Male emotion")
x_train,x_test,y_train,y_test = load_data("m", test_size=0.20)

# print some details
# number of samples in training data
print("[+] Number of training samples:", x_train.shape[0])
# number of samples in testing data
print("[+] Number of testing samples:", x_test.shape[0])
# number of features used
# this is a vector of features extracted 
# using utils.extract_features() method
print("[+] Number of features:", x_train.shape[1])


#________________________________________________________________________
#FIRST MODEL
model_params = {
    'alpha': 0.01,
    'batch_size': 256,
    'epsilon': 1e-08, 
    'hidden_layer_sizes': (300,), 
    'learning_rate': 'adaptive', 
    'max_iter': 500, 
}
    
# initialize Multi Layer Perceptron classifier
# with best parameters ( so far )
model = MLPClassifier(**model_params)

# train the model
print("[*] Training the model...")
model.fit(x_train, y_train)

# predict 25% of data to measure how good we are
y_pred = model.predict(x_test)

# calculate the accuracy
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

print("Accuracy first model: {:.2f}%".format(accuracy*100))
#print(classification_report(y_test,y_pred))
#print(confusion_matrix(y_test,y_pred))

#________________________________________________________________________
#SECOND MODEL
m_params = {
    'alpha': 0.01,
    'batch_size': 200,
    'epsilon': 1e-08, 
    'hidden_layer_sizes': (300,), 
    'learning_rate': 'adaptive', 
    'max_iter': 500, 
}
# initialize Multi Layer Perceptron classifier
# with best parameters ( so far )
m1 = MLPClassifier(**m_params)

# train the model
print("[*] Training the model...")
m1.fit(x_train, y_train)

# predict 25% of data to measure how good we are
y_p = m1.predict(x_test)

# calculate the accuracy
accuracy = accuracy_score(y_true=y_test, y_pred=y_p)

print("Accuracy second model: {:.2f}%".format(accuracy*100))
#print(classification_report(y_test,y_p))
#print(confusion_matrix(y_test,y_p))


# In[102]:


#loading_data
print("Female emotion")
x_train,x_test,y_train,y_test = load_data("f", test_size=0.20)

# print some details
# number of samples in training data
print("[+] Number of training samples:", x_train.shape[0])
# number of samples in testing data
print("[+] Number of testing samples:", x_test.shape[0])
# number of features used
# this is a vector of features extracted 
# using utils.extract_features() method
print("[+] Number of features:", x_train.shape[1])


#________________________________________________________________________
#FIRST MODEL
model_params = {
    'alpha': 0.01,
    'batch_size': 256,
    'epsilon': 1e-08, 
    'hidden_layer_sizes': (300,), 
    'learning_rate': 'adaptive', 
    'max_iter': 500, 
}
    
# initialize Multi Layer Perceptron classifier
# with best parameters ( so far )
model = MLPClassifier(**model_params)

# train the model
print("[*] Training the model...")
model.fit(x_train, y_train)

# predict 25% of data to measure how good we are
y_pred = model.predict(x_test)

# calculate the accuracy
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

print("Accuracy first model: {:.2f}%".format(accuracy*100))
#print(classification_report(y_test,y_pred))
#print(confusion_matrix(y_test,y_pred))

#________________________________________________________________________
#SECOND MODEL
m_params = {
    'alpha': 0.01,
    'batch_size': 200,
    'epsilon': 1e-08, 
    'hidden_layer_sizes': (300,), 
    'learning_rate': 'adaptive', 
    'max_iter': 500, 
}
# initialize Multi Layer Perceptron classifier
# with best parameters ( so far )
m1 = MLPClassifier(**m_params)

# train the model
print("[*] Training the model...")
m1.fit(x_train, y_train)

# predict 25% of data to measure how good we are
y_p = m1.predict(x_test)

# calculate the accuracy
accuracy = accuracy_score(y_true=y_test, y_pred=y_p)

print("Accuracy second model: {:.2f}%".format(accuracy*100))
#print(classification_report(y_test,y_p))
#print(confusion_matrix(y_test,y_p))


# In[ ]:




