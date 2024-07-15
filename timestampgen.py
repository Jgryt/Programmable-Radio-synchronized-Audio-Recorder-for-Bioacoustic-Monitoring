import numpy as np

def int_to_bit_array(number, bit_width):

    # Convert number to a binary string, removing the '0b' prefix
    binary_str = bin((number + (1 << bit_width)) % (1 << bit_width))[2:]
    
    # Pad the binary string to ensure it has a length of 'bit_width'
    padded_binary = binary_str.zfill(bit_width)
    
    # Convert the binary string to a NumPy array of integers
    bit_array = np.array([int(bit) for bit in padded_binary], dtype=int)
    
    return bit_array

def map_to_second(binary_arr, samples_per_sec):
    dcf77_repr = []

    for arr in binary_arr:
        for bit in arr:
            sec_vals = np.ones(samples_per_sec, dtype=int)
            #set the number of logic low to represent twice the lenght of logic high
            if bit==1:
                sec_vals[-2:] = 0
            elif bit==0:
                sec_vals[-1] = 0
            dcf77_repr.append(sec_vals)
    return np.array(dcf77_repr)

#slice to max samples
def prepare_for_buffer(data, num_buff=2, samples_buff=1000):
    buff = []
    for b in range(num_buff):
        buff.append(data[int(b*samples_buff): int((b+1)*samples_buff)])
    return np.array(buff).transpose()

#number of seconds
seconds = 30

#starting time
h = 12
m = 30
s = 15

filename = f'h{h}m{m}m{s}_1024_samples'

time_array = []
for sec in range(seconds):
    h_store = int_to_bit_array(h, 5)
    m_store = int_to_bit_array(m, 6)
    s_store = int_to_bit_array(s, 6)
    time = np.concatenate([h_store, m_store, s_store])
    time_array.append(time)
    
    s += 1
    if s >= 60:
        s = 0
        m += 1
        if m >= 60:
            m = 0
            h = (h + 1) % 24

time_array = np.array(time_array)

print(time_array.shape)
print(len(time_array[0]))

#map to 
amplitude_mod = map_to_second(time_array, samples_per_sec=10)
amplitude_mod = amplitude_mod.flatten()
print(amplitude_mod.shape)

amplitude_mod = prepare_for_buffer(amplitude_mod, num_buff=1, samples_buff=1024)
print(amplitude_mod)
print(amplitude_mod.shape)

np.savetxt(f'generated_synchsig/{filename}.csv', amplitude_mod, delimiter=',', fmt='%d')
