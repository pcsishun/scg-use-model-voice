"""
This file can be used to try a live prediction. 
"""

import keras
import numpy as np
import librosa
import recorder
import pvporcupine
import pyaudio
import struct
# from tensorflow import keras 

filename="test.wav"



class livePredictions:
    """
    Main class of the application.
    """

    def __init__(self, path, file):
        """
        Init method is used to initialize the main parameters.
        path="SER_model.h5", file="test.wav"
        """

        print("path ===> ", path)
        print("file ===>", file)

        self.path = "SER_model.h5" # put model here
        self.file = "test.wav" # put voice here

        print("path ===> ", path)
        print("file ===>", file)  
    
    def load_model(self):
        """
        Method to load the chosen model.
        :param path: path to your h5 model.
        :return: summary of the model with the .summary() function.
        """
        print("loading model ==>", self.path)
        self.loaded_model = keras.models.load_model(self.path)
        # print("loading model ", self.loaded_model)
        return self.loaded_model

    def makepredictions(self):
        """
        Method to process the files and create your features.
        """
        print("loading file ===>", self.file)
        data, sampling_rate = librosa.load(self.file)
        # print("loading file", self.file)
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
        x = np.expand_dims(mfccs, axis=1)
        x = np.expand_dims(x, axis=0)
        # predictions = self.loaded_model.predict_classes(x)
        # print("Prediction is", " ", self.convertclasstoemotion(predictions))

        predictions = self.loaded_model.predict_classes(x)
        self.convertclasstoemotion(predictions)

    @staticmethod
    def convertclasstoemotion(pred):
        """
        Method to convert the predictions (int) into human readable strings.
        """
        
        label_conversion = {'0': 'neutral',
                            '1': 'calm',
                            '2': 'happy',
                            '3': 'sad',
                            '4': 'angry',
                            '5': 'fearful',
                            '6': 'disgust',
                            '7': 'surprised'}

        for key, value in label_conversion.items():
            if int(key) == pred:
                label = value
                print("predict ===> ",label)
        return label

# Here you can replace path and file with the path of your model and of the file 
#from the RAVDESS dataset you want to use for the prediction,
# Below, I have used a neutral file: the prediction made is neutral.

try:
    print("start app...")
    accesskey = "4QUFReTvGjJJnCxTrw7JPgATaHClIGekXV/cuJzYYD6cO3K4fs3qMA=="
    porcupine = pvporcupine.create(access_key=accesskey,keywords=["computer", "alexa"])
    pa = pyaudio.PyAudio()
    audio_stream = pa.open(
        rate=porcupine.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=porcupine.frame_length)
    
    while True:
        pcm = audio_stream.read(porcupine.frame_length)
        pcms = struct.unpack_from("h" * porcupine.frame_length, pcm)
        keyword_index = porcupine.process(pcms)
        print(keyword_index)

        if keyword_index >= 0:
            print("startRecord")
            recorder.record(filename)
            pred = livePredictions(path='SER_model.h5',file='test.wav')

            pred.load_model()
            pred.makepredictions()
            break
            
except:
    print("error detect!")
    if porcupine is not None:
        porcupine.delete()
        print("End")
    if audio_stream is not None:
        audio_stream.close()
        print("End")
    if pa is not None:
        pa.terminate()
        print("End")

 

