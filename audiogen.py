import numpy as np
from scipy.io import wavfile
import csv

#function files
import audiogen_funcs

num_samples = 22050 #44100/2 as wavegen buffer can only hold a maximum of 32768 samples for the configuration

#Resample to match professional samplerate of recording equipment
resample_to = 48000

file_name = 'Hand Clap 02'
audio_wav = f'audio/{file_name}.wav'

samplerate, data = wavfile.read(filename=audio_wav)

#Only need one channel
data_ch1 = data[:,0]

#make audio 1 sec long
data_ch1 = audiogen_funcs.to_1sec(data_ch1, samplerate)

assert len(data_ch1)==samplerate, 'len(data_ch1)) != samplerate' 

#Resample, get new data and samplerate
data_ch1, samplerate = audiogen_funcs.resample(data_ch1, old_samplerate=samplerate, new_samplerate=resample_to, plot=True)

assert len(data_ch1)==resample_to, 'resampling failed' 

#meter distance between microphones
intermic_distance = 2
#Angle of incidence (aoi)
aoi = 0

num_aio = 12

for i in range(num_aio):
    aoi = i*(360/num_aio) 

    #estimate num samples tdoa, assuming speed of sound = 343m/s
    tdoa, samples_tdoa = audiogen_funcs.find_tdoa(aoi, intermic_distance, samplerate)

    # assert False
    if samples_tdoa>=0:
        #mic closest
        audio1 = np.pad(data_ch1, (0,samples_tdoa), constant_values=(0,0))
        #mic farthest
        audio2 = np.pad(data_ch1, (samples_tdoa,0), constant_values=(0,0))
    #negative tdoa means mic2 is closer to sound source -> change padding order  
    elif samples_tdoa<0:
        #mic closest
        audio2 = np.pad(data_ch1, (0,abs(samples_tdoa)), constant_values=(0,0))
        #mic farthest
        audio1 = np.pad(data_ch1, (abs(samples_tdoa),0), constant_values=(0,0))

    if audio1.shape != audio2.shape:
        print(f'OBS audio shapes are not equal: a1={audio1.shape} vs a2={audio2.shape}')

    #prepare for wavegen buffer
    audio1 = audio1[:24000]
    audio2 = audio2[:24000]
    assert len(audio1) == 24000
    assert len(audio2) == 24000

    print('## ### ### ##')
    print(f'AoI = {aoi}')
    print(f'TDOA samples = {samples_tdoa}')
    print('## ### ### ##')

    with open(f'generated_audio/{int(aoi)}deg_{int(samples_tdoa)}s_tdoa_1.csv', 'w') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(audio1)

    with open(f'generated_audio/{int(aoi)}deg_{int(samples_tdoa)}s_tdoa_2.csv', 'w') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(audio2)
    