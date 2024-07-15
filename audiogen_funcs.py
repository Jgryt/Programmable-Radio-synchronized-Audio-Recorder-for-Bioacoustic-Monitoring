import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, ifft

#Ensures data is 1 sec long
def to_1sec(data, samplerate):
    samples_diff = samplerate-len(data)
    if samples_diff > 0:
        data = np.pad(data, (0,samples_diff), constant_values=(0,0))
    elif samples_diff < 0:
        data = data[:samplerate]
    return data

def resample(data, old_samplerate, new_samplerate, plot=True):
    data_resampled = signal.resample(data, new_samplerate)
    
    if plot:
        _, axs = plt.subplots(2)
        axs[0].plot(data)
        axs[0].set_title(f'Old samplerate of {old_samplerate}')
        axs[1].plot(data_resampled)
        axs[1].set_title(f'New samplerate of {new_samplerate}')
        plt.tight_layout()
        plt.show()
    return data_resampled.astype(np.int32), new_samplerate

def find_tdoa(aoi, intermic_distance=2, samplerate=48000):
    #tdoa assuming speed of sound = 343m/s
    effective_distance = intermic_distance * np.cos(aoi*(np.pi/180))
    tdoa = effective_distance / 343
    #num samples
    return tdoa, round(samplerate * tdoa)

def to_linelevel(sig):
    line_level = 1.736
    # sig = np.int64(sig)
    return sig/max(abs(sig)) * line_level

def bit_reduction(sig, res):
    return np.right_shift(sig, (32-res))

def butter_filter(sig, freq, filter_type, fs, order=20):
    filter = butter_coeff(order, cutoff_freq=freq, type=filter_type,fs=fs)

    return apply_bfilter(filter, sig)

def butter_coeff(order, cutoff_freq, type, fs, analog=False, output='sos'):
    #It’s recommended to use second-order sections format when filtering, to avoid numerical error with transfer function (ba) format
    sos = signal.butter(N=order, Wn=cutoff_freq, btype=type, analog=analog, fs=fs, output=output)
    return sos

#Filter a data sequence, x, using a digital IIR filter defined by sos
def apply_bfilter(sos, sig_arra):
    return signal.sosfilt(sos, sig_arra) #sos - Array of second-order filter coefficients 

def plot_fft(data, Fs, color='black', title='FFT plot', show=True):
    fft_spect = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(fft_spect), d=1/(Fs))
    plt.figure()
    plt.plot(freqs[1:len(freqs)//2], 20*np.log10(abs(fft_spect[1:len(fft_spect)//2])), color=color)
    plt.title(title)
    plt.ylabel('Magnitude (dB)')

    if show:
        plt.show()

def plot_phat(phat_cc, sig1, shift_flag, shift):
    shift_axis = np.arange(start=-len(sig1)/2, stop=len(sig1)/2)
    if shift_flag:
        phat_cc = np.concatenate((phat_cc[int((len(phat_cc))/2):], phat_cc[:int((len(phat_cc))/2)]))
    plt.plot(shift_axis, phat_cc, color='blue')
    plt.text(0.05, 0.8, f'Shift: {shift}', fontsize=12, weight='bold', color='black', bbox={'facecolor': 'white', 'alpha': 1, 'pad': 6, 'edgecolor': 'black'})
    plt.title('Phat Cross Correlation')
    plt.show()

#Estiamtes slices for axtracting audio of interest
def plot_slices(sig1, sig2, title1, title2, window, num_peaks):
    fig, axs = plt.subplots(2)
    axs[0].plot(sig1)
    axs[0].set_title(title1)
    
    axs[1].plot(sig2)
    axs[1].set_title(title2)
    
    sig1_sort = np.argsort(sig1)
    sig1_peaks_idx = sig1_sort[-num_peaks:]
    
    sig2_sort = np.argsort(sig2)
    sig2_peaks_idx = sig2_sort[-num_peaks:]
    colors = ['red', 'green', 'orange', 'black','magenta', 'blue', 'teal', 'olive']
    for peak in range(len(sig1_peaks_idx)):
         axs[0].axvline(x=(sig1_peaks_idx[peak]-window[0]), label=f'{sig1[sig1_peaks_idx[peak]]-window[0]}', color=colors[peak])
         axs[0].axvline(x=(sig1_peaks_idx[peak]+window[1]), label=f'{sig1[sig1_peaks_idx[peak]]+window[1]}', color=colors[peak])
         axs[1].axvline(x=(sig2_peaks_idx[peak]-window[0]), label=f'{sig2[sig2_peaks_idx[peak]]-window[0]}', color=colors[peak])
         axs[1].axvline(x=(sig2_peaks_idx[peak]+window[1]), label=f'{sig2[sig2_peaks_idx[peak]]+window[1]}', color=colors[peak])
    # plt.legend()
    plt.tight_layout()
    plt.show()
    return np.dstack((sig1_peaks_idx-window[0], sig1_peaks_idx+window[1])).squeeze(), np.dstack((sig2_peaks_idx-window[0], sig2_peaks_idx+window[1])).squeeze()

#constructs square wave synch signal around transition of thresshold of envolope of filtered signal
def envelope_detector(f_low, f_high, threshhold = 0.5):
    #Find envelope
    f_low = np.abs(signal.hilbert(f_low))
    f_high = np.abs(signal.hilbert(f_high))

    # #Find values where the transitions are stable
    f_low_middle = np.max(f_low) * threshhold
    f_high_middle = np.max(f_high) * threshhold

    #Consruct square wave for stable values
    f_low_square_wave = []
    f_high_square_wave = []

    assert len(f_low) == len(f_high)
    
    for sample in range(len(f_low)):
        if f_low[sample] <= f_low_middle:
            f_low_square_wave.append(0)
        elif f_low[sample] > f_low_middle:
            f_low_square_wave.append(1)
        
        if f_high[sample] <= f_high_middle:
            f_high_square_wave.append(0)
        elif f_high[sample] > f_high_middle:
            f_high_square_wave.append(1)

    return f_low_square_wave, f_high_square_wave

def demodulate_fsk(sig, freqs, fs):
    #Find middle frequency between high and low frequency component in the fsk signal
    middle_freq = abs((freqs[1]-freqs[0])/2)

    f_low_bp_filter = butter_coeff(20, cutoff_freq=[(freqs[0] - middle_freq), (freqs[0] + middle_freq)], type='bandpass',fs=fs)
    f_high_bp_filter = butter_coeff(20, cutoff_freq=[(freqs[1] - middle_freq), (freqs[1] + middle_freq)], type='bandpass',fs=fs)

    f_low_filtered = apply_bfilter(f_low_bp_filter, sig)
    f_high_filtered = apply_bfilter(f_high_bp_filter, sig)

    f_low_envelope, f_high_envelope = envelope_detector(f_low_filtered, f_high_filtered)

    return f_low_envelope, f_high_envelope

#Finds relative samples shift with cross correlation in fourier domain 
def find_sample_shift(sig1, sig2, synch_sig=False, plot=False):
    #pad end of array if lenght is not equal. (Padding the end should not cause shift)
    if len(sig1) != len(sig2):
        print('len(sig1) != leng(sig2), signals are padded to equal lenght')
        longest_rec = max(len(sig1), len(sig2))
        sig1 = np.pad(sig1, (0, longest_rec - len(sig1)), 'constant')
        sig2 = np.pad(sig2, (0, longest_rec - len(sig2)), 'constant')

    print(f'sig1 length: {len(sig1)}')
    print(f'sig2 length: {len(sig2)}')

    s1_fft = fft(sig1)
    s2_fft = fft(sig2)

    if synch_sig:
        #Synchsignal does not need normalizing
        phat_cc = ifft(s1_fft * np.conj(s2_fft))
    else:
        #mitigate NaN values for phat_cc when denomiator is approx 0
        epsilon = 1e-10 
        #normalize to phat spectrum before finding cross correlation
        phat_cc = ifft((s1_fft * np.conj(s2_fft)) / abs(s1_fft * np.conj(s2_fft) + epsilon))

    shift = np.argmax(phat_cc)

    #Resulting phat_cc ranges from 0 to (len(sig1==sig2)-1), meaning shift=0 is at index len(sig1)/2
    shift_flag = 0
    if shift > len(sig1) / 2:
        shift -= len(sig1)
        shift_flag = 1
    
    if plot:
        plot_phat(phat_cc, sig1, shift_flag, shift)

    return shift

#synchronizes two shifted arrays by padding array with sample_shift
def synchronize(sig1, sig2, sample_shift):
    #pad with zeros correspondign to sample_shift
    if sample_shift > 0:
        sig1_aligned = np.pad(sig1, (0, abs(sample_shift)), constant_values=(0,0))
        sig2_aligned = np.pad(sig2, (abs(sample_shift), 0), constant_values=(0,0))
        return sig1_aligned, sig2_aligned
    else:
        sig1_aligned = np.pad(sig1, (abs(sample_shift), 0), constant_values=(0,0))
        sig2_aligned = np.pad(sig2, (0, abs(sample_shift)), constant_values=(0,0))
        return sig1_aligned, sig2_aligned
    
#encapsulating function for ensuring correct passing of paramters... 
def synch_signals(synch_sig1, synch_sig2, audio1, audio2):
    sample_shift = find_sample_shift(synch_sig1, synch_sig2,synch_sig=True)# plot)
    print(f'synch signal shift: {sample_shift}')
    a1_synched, a2_synched = synchronize(audio1, audio2, sample_shift)
    return a1_synched, a2_synched, sample_shift
