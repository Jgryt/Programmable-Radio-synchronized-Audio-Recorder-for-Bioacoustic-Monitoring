import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

#Supporting function file
import audiogen_funcs

#low and high in fsk synch signal
synch_freqs = np.array([20000, 21500])

file_name_1 = 'fsk/fsk_time_0deg_280s_1'
file_name_2 = 'fsk/fsk_time_0deg_280s_2'

plot_title = 'Audio synchronization test'

audio_wav_1 = f'recorded_audio/{file_name_2}.wav'
audio_wav_2 = f'recorded_audio/{file_name_1}.wav'

samplerate_1, data_1 = wavfile.read(filename=audio_wav_1)
samplerate_2, data_2 = wavfile.read(filename=audio_wav_2)

assert samplerate_1 == samplerate_2, 'OBS! Samplerate not equal'
samplerate = samplerate_1

#meter distance between microphones
intermic_distance = 2
#maximum delay when angle of incidence is 0 deg
max_tdoa = intermic_distance / 343
samples_max_tdoa = int(samplerate * max_tdoa)

#sound of interest duration (sec)
soi_duration = 1

#Slice by maximum delay possible
slice_window = [samples_max_tdoa*4, samplerate*soi_duration]

#filter for audio
filtered_audio1 = audiogen_funcs.butter_filter(data_1, synch_freqs[0]-2000, filter_type='lowpass', fs=samplerate, order=20)
filtered_audio2 = audiogen_funcs.butter_filter(data_2, synch_freqs[0]-2000, filter_type='lowpass', fs=samplerate, order=20)

#The audio signals are inverted compared to each other, due to the xlr plugs having different wiring...
filtered_audio2 = filtered_audio2*-1

   ## ### ## ### ## #### ## ### ## ### ##
### For synching over slices of interest ###

num_peaks = 1
# # plots and returns estimated slices, returns [[slice1_low slice2_high], [slice2_low slice2_high], ...]
d1_slices, d2_slices = audiogen_funcs.plot_slices(filtered_audio1, filtered_audio2, title1=file_name_1, title2=file_name_2, window=slice_window, num_peaks=num_peaks)

# # #extract synch data
# try:
#     d1_fsk_low, d1_fsk_high = audiogen_funcs.demodulate_fsk(data_1[int(d1_slices[0][0]) : int(d1_slices[0][1])], freqs=synch_freqs, fs=samplerate)
#     d2_fsk_low, d2_fsk_high = audiogen_funcs.demodulate_fsk(data_2[int(d2_slices[0][0]) : int(d2_slices[0][1])], freqs=synch_freqs, fs=samplerate)
# except:
#     print('except:')
#     d1_fsk_low, d1_fsk_high = audiogen_funcs.demodulate_fsk(data_1[int(d1_slices[0]) : int(d1_slices[1])], freqs=synch_freqs, fs=samplerate)
#     d2_fsk_low, d2_fsk_high = audiogen_funcs.demodulate_fsk(data_2[int(d2_slices[0]) : int(d2_slices[1])], freqs=synch_freqs, fs=samplerate)

# # Synchronize audio signal according to 
# # a1_synched, a2_synched, synch_sample_shift = audiogen_funcs.synch_signals(d1_fsk_high, d2_fsk_high, filtered_audio1[int(d1_slices[0]) : int(d1_slices[1])], filtered_audio2[int(d2_slices[0]) : int(d2_slices[1])])
# a1_synched, a2_synched, synch_sample_shift = audiogen_funcs.synch_signals(d1_fsk_low, d2_fsk_low, filtered_audio1[int(d1_slices[0]) : int(d1_slices[1])], filtered_audio2[int(d2_slices[0]) : int(d2_slices[1])])

## ### ### ### ### ### ### ### ### ### ##
 ## ### ## ### ## #### ## ### ## ### ##

   ## ### ## ### ## ### ## ### ## ### ##
### For synching over whole recording ###

d1_fsk_low, d1_fsk_high = audiogen_funcs.demodulate_fsk(data_1, freqs=synch_freqs, fs=samplerate)
d2_fsk_low, d2_fsk_high = audiogen_funcs.demodulate_fsk(data_2, freqs=synch_freqs, fs=samplerate)
# a1_synched, a2_synched, synch_sample_shift = audiogen_funcs.synch_signals(d1_fsk_high, d2_fsk_high, filtered_audio1, filtered_audio2)
a1_synched, a2_synched, synch_sample_shift = audiogen_funcs.synch_signals(d1_fsk_low, d2_fsk_low, filtered_audio1, filtered_audio2)

## ### ### ### ### ### ### ### ### ### ##
  ## ### ## ### ## ### ## ### ## ### ##

#For plotting the synched timestamp signal
s1_synched, s2_synched, _ = audiogen_funcs.synch_signals(d1_fsk_high, d2_fsk_high, d1_fsk_high, d2_fsk_high)
# s1_synched, s2_synched, _ = audiogen_funcs.synch_signals(d1_fsk_low, d2_fsk_low, d1_fsk_low, d2_fsk_low)

audio_sample_shift = audiogen_funcs.find_sample_shift(a1_synched, a2_synched)#, plot=True)


audio_time_shift = audio_sample_shift / samplerate
print(f'Synchronized audio sample shift: {audio_sample_shift}') 
print(f'Synchronized audio time shift: {audio_time_shift:.6f}') 

fig, large_ax = plt.subplots()
large_ax.plot(a1_synched, label='audio 1', color='black')
large_ax.plot(a2_synched, label='audio 2', color='red')
large_ax.set_title(f'Synched Audio')
fig.text(0.05, 0.8, f'Sample Shift: {audio_sample_shift}', fontsize=12, weight='bold', color='black', bbox={'facecolor': 'white', 'alpha': 1, 'pad': 6, 'edgecolor': 'green'})
large_ax.yaxis.set_visible(False)
large_ax.legend()

#Slice inserted ax based on .plot_slices()
slice_low = min(int(d1_slices[0]), int(d2_slices[0]))
slice_high = max(int(d1_slices[1]), int(d2_slices[1]))
a1_sliced = filtered_audio1[slice_low : slice_high]
a2_sliced = filtered_audio2[slice_low : slice_high]
small_ax = fig.add_axes([.05, 0.18, .4, .2], facecolor='azure')
small_ax.plot(a1_sliced, color='black')
small_ax.plot(a2_sliced, color='red')
small_ax.set_title('Unsynched')
small_ax.yaxis.set_visible(False)
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(2)
axs[0].plot(d1_fsk_high, label='s1', color='black')
axs[0].plot(d2_fsk_high, label='s2', color='red')
axs[0].set_title(f'Unsynched timestamp')
fig.text(0.05, 0.8, f'Sample Shift: {synch_sample_shift}', fontsize=8, weight='bold', color='black',  bbox={'facecolor': 'white', 'alpha': 1, 'pad': 6, 'edgecolor': 'black'})
axs[0].yaxis.set_visible(False)
axs[0].legend()

axs[1].plot(s1_synched, label='s1', color='black')
axs[1].plot(s2_synched, label='s2', color='red')
axs[1].set_title('Synched Timestamp')
axs[1].yaxis.set_visible(False)
axs[1].legend()
fig.suptitle(plot_title)
plt.tight_layout()
plt.show()

# ref https://matplotlib.org/stable/gallery/subplots_axes_and_figures/axes_demo.html#sphx-glr-gallery-subplots-axes-and-figures-axes-demo-py
