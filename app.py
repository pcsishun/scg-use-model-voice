"""
This file can be used to try a live prediction. 
"""

# from http import client
import keras
import numpy as np
import librosa
import recorder
import pvporcupine
import pyaudio
import struct
from time import sleep

import os

# from tensorflow import keras 

filename="test.wav"
# connection_string = "mongodb://localhost:27017/" ## connect to local host ## 



class livePredictions:
    """
    Main class of the application.
    """

    def __init__(self, path, file):
        """
        Init method is used to initialize the main parameters.
        path="SER_model.h5", file="test.wav"
        """
 

        self.path = "SER_model.h5" # put model here
        self.file = "test.wav" # put voice here
 
    
    def load_model(self):
        """
        Method to load the chosen model.
        :param path: path to your h5 model.
        :return: summary of the model with the .summary() function.
        """
        print("loading model ==>", self.path)
        self.loaded_model = keras.models.load_model(self.path)
        return self.loaded_model
    
    ###  function ทำนายอารมณ์จากเสียง 
    def makepredictions(self):
        """
        Method to process the files and create your features.
        """
        print("loading file ===>", self.file)
        data, sampling_rate = librosa.load(self.file)
        
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
        x = np.expand_dims(mfccs, axis=1)
        x = np.expand_dims(x, axis=0)
        predictions = self.loaded_model.predict_classes(x)
        self.convertclasstoemotion(predictions)

    @staticmethod
    def convertclasstoemotion(pred):
        """
        Method to convert the predictions (int) into human readable strings.
        """
        print("pred ===> ",pred)
        label_conversion = {'0': 'Voice emotion: neutral',
                            '1': 'Voice emotion: calm',
                            '2': 'Voice emotion: happy',
                            '3': 'Voice emotion: sad',
                            '4': 'Voice emotion: angry',
                            '5': 'Voice emotion: fearful',
                            '6': 'Voice emotion: disgust',
                            '7': 'Voice emotion: surprised'}

        ## ระบบจะทำการหา index ของอารมณ์ จาก function makepredictions และ เขียนผลลงใน emotion_voice ตรงนี้ ##
        for key, value in label_conversion.items():
            if int(key) == pred:
                print("key ===> ",key)
                label = value
                print("predict ===> ",label)
                write_f = open("emotion_voice.txt", "w")
                write_f.write(label)
                write_f.close()
        # return label

def sleep_function(num):
    print(f'Waiting for start record in {num} seconds.')
    sleep(num)
    print("Start recording...")
    

def app_start():
    try:
        ## set การเปิดการทำงานด้วยเสียง
        print("start app...")
        accesskey = "PSMtKVzOysHScGy5g2mgc2ClrX1Xf/PYYafb4o7kQQMUsZmrbBAR1Q=="
        porcupine = pvporcupine.create(access_key=accesskey,keywords=["computer", "alexa"])
        pa = pyaudio.PyAudio()
        audio_stream = pa.open(
            # rate=porcupine.sample_rate,
            rate=porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            # frames_per_buffer=porcupine.frame_length
            frames_per_buffer=porcupine.frame_length
        )
        
        print("app listening voice...")
        while True:
            # สั่งใช้งาน function ปิดการทำงาน 
            ## คำสั่งเปิดการทำงานด้วยเสียง 

            pcm = audio_stream.read(porcupine.frame_length)
            pcms = struct.unpack_from("h" * porcupine.frame_length, pcm)
            keyword_index = porcupine.process(pcms)

            if keyword_index >= 0:
                print("is recording")
                write_f = open("emotion_voice.txt", "w")
                write_f.write("Start recording...")
                write_f.close()
                while True:
                    ## หน่วงเวลา 5 วิ
                    sleep_function(5)

                    ## สั่งบันทึกเสียง 10 วิ
                    recorder.record(filename)

                    ## สั่งทำนายเสียง
                    pred = livePredictions(path='SER_model.h5',file='test.wav')
                    pred.load_model() # โหลด model 
                    pred.makepredictions() # ทำนายผล 
                    os.remove("test.wav") # ลบไฟล์เสียงออก
                    print("finish predict")
        
                
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

app_start()

## เรียกใช้งาน app เเละทำการ loop inf. ##
action = True
while action == True:
    print("action..")
    app_start()
