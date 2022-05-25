import pyaudio
import wave


def record(filename):
    print("In recorder function!")
    chunk = 1024
    FORMAT = pyaudio.paInt16 
    # mono, change to 2 if you want stereo
    channels = 1 ## for using debug is 1 
    sample_rate = 44100
    record_seconds = 10 ## default from model is 4 sec
    p = pyaudio.PyAudio()
    print("setting all config stream")
    stream = p.open(format=FORMAT,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk)
    frames = []
    print("Recording...")
    for i in range(0, int(sample_rate / chunk * record_seconds)):
        data = stream.read(chunk)
        # print("data ==> ",data)
        frames.append(data)
    print("Finished recording.")
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(filename, "wb")
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(sample_rate)
    wf.writeframes(b"".join(frames))
    wf.close()
