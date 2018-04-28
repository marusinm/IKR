import librosa
import os
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
import numpy as np
from tqdm import tqdm

#path to training data
TRAIN_DATA_PATH = "./my_data/"
#path to evaluation data
EVAL_DATA_PATH = "./eval/"
# Feature dimension
feature_dim_1 = 20
channel = 1
epochs = 50
batch_size = 100
verbose = 1
num_classes = 2
# Second dimension of the feature is dim2
feature_dim_2 = 11


# Input: Folder Path
# Output: Tuple (Label, Indices of the labels, one-hot encoded labels)
def get_labels(path=TRAIN_DATA_PATH):
    labels = os.listdir(path)
    label_indices = np.arange(0, len(labels))
    return labels, label_indices, to_categorical(label_indices)


# Handy function to convert wav2mfcc
def wav2mfcc(file_path, max_len=11):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    wave = wave[::3]
    mfcc = librosa.feature.mfcc(wave, sr=16000)

    # If maximum length exceeds mfcc lengths then pad the remaining ones
    if (max_len > mfcc.shape[1]):
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

    # Else cutoff the remaining parts
    else:
        mfcc = mfcc[:, :max_len]
    
    return mfcc


def save_data_to_array(path=TRAIN_DATA_PATH, max_len=11):
    labels, _, _ = get_labels(path)

    for label in labels:
        # Init mfcc vectors
        mfcc_vectors = []

        wavfiles = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
        for wavfile in tqdm(wavfiles, "Saving vectors of label - '{}'".format(label)):
            mfcc = wav2mfcc(wavfile, max_len=max_len)
            mfcc_vectors.append(mfcc)
        np.save(label + '.npy', mfcc_vectors)


def get_train_test(split_ratio=0.6, random_state=42):
    # Get available labels
    labels, indices, _ = get_labels(TRAIN_DATA_PATH)

    # Getting first arrays
    X = np.load(labels[0] + '.npy') #load firs npy file
    y = np.zeros(X.shape[0])        #create same array of shape X

    # Append all of the dataset into one single array, same goes for y
    for i, label in enumerate(labels[1:]): #go throught rest of npy files and use i as counter
        x = np.load(label + '.npy')        #load next npy file
        X = np.vstack((X, x))              #make 1D array from npyarray
        y = np.append(y, np.full(x.shape[0], fill_value= (i + 1))) #append in y, with x shape and fill with counter values

    assert X.shape[0] == len(y)

    return train_test_split(X, y, test_size= (1 - split_ratio), random_state=random_state, shuffle=True)

#Create model
def get_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(feature_dim_1, feature_dim_2, channel)))
    model.add(Conv2D(48, kernel_size=(2, 2), activation='relu'))
    model.add(Conv2D(120, kernel_size=(2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
#     model.add(Dense(num_classes, activation='softmax'))
    model.add(Dense(num_classes, activation='sigmoid'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model

# Predicts one sample
def predict(filepath, model):
    sample = wav2mfcc(filepath)
    sample_reshaped = sample.reshape(1, feature_dim_1, feature_dim_2, channel)
    return get_labels()[0][
            np.argmax(model.predict(sample_reshaped))
    ]

def main():
    # Save data to array file first
    save_data_to_array(max_len=feature_dim_2)

    # Loading train set and test set randomly splitted 
    X_train, X_test, y_train, y_test = get_train_test()

    # Reshaping to perform 2D convolution
    X_train = X_train.reshape(X_train.shape[0], feature_dim_1, feature_dim_2, channel)
    X_test = X_test.reshape(X_test.shape[0], feature_dim_1, feature_dim_2, channel)

    # One-hot encoding
    y_train_hot = to_categorical(y_train)
    y_test_hot = to_categorical(y_test)

    model = get_model()
    model.fit(X_train, y_train_hot, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=(X_test, y_test_hot))

    # Output prediction
    with open('output.txt', 'w') as f:
        print('wait for output ....')
        files = os.listdir(EVAL_DATA_PATH)
        files.sort()
        for file in files:
            if file.endswith(".wav"):
                # Getting the MFCC
                sample = wav2mfcc(os.path.join(EVAL_DATA_PATH, file))
                # We need to reshape it remember?
                sample_reshaped = sample.reshape(1, 20, 11, 1)

                # Perform forward pass
                predicted_class = np.argmax(model.predict(sample_reshaped))
                predicted_probability = np.argmax(model.predict_proba(sample_reshaped))
                # print(os.path.join(file[:-4]),
                #     predicted_class,
                #     predicted_probability,
                #     file=f)
                print(os.path.join(file[:-4]),
                    predicted_probability,
                    predicted_class,"(",get_labels()[0][predicted_class],")",
                    file=f)
        print('done!')
        f.close()




#Start program 
main()




