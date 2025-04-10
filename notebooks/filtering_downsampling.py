%%time
import os
import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi
# from scipy.signal import butter, sosfilt, sosfilt_zi
from scipy.signal import butter, sosfiltfilt

def get_all_chunk_bounds(total_length, chunk_size):
    chunks = []
    for i in range(0, total_length, chunk_size):
        start = i
        end = min(i + chunk_size, total_length)
        chunks.append((start, end))
    return chunks
# Design filter (Butterworth bandpass)
def design_filter(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    b = butter(order, highcut / nyq, btype='low', output='sos')#butter(order, [lowcut / nyq, highcut / nyq], btype='band', output='sos')
    return b
# file_name = '/mnt/server_epl/SCN1A/R1648H-export/20-3-25/slice2/2025-03-20T12-08-00McsRecording.h5'
file_name = '/mnt/server_epl/SCN1A/R1648H-export/20-3-25/slice5/2025-03-20T15-02-51McsRecording.h5' #'/mnt/server_epl/SCN1A/R1648H-export/20-3-25/slice4/2025-03-20T14-07-28McsRecording.h5'

# source_file = '/mnt/server_epl/SCN1A/R1648H-export/20-3-25/slice2/2025-03-20T12-08-00McsRecording.h5'
# get file info
channel_raw_data = McsPy.McsData.RawData(file_name)    
analog_stream_0 = channel_raw_data.recordings[0].analog_streams[0]
adc_step = analog_stream_0.channel_infos[2].adc_step.magnitude 
fs = na(analog_stream_0.channel_infos[2].sampling_frequency) # get frequency
date_time =channel_raw_data.date
n_channels=252
b = design_filter(.1, 200, fs)
subsampling_rate = 800# Hz
subsampling_step = int(fs/subsampling_rate) # 


slice_ =file_name.split('/')[-2]
name = file_name.split('/')[-1]#'2025-03-20T14-07-28McsRecording'
date = formatted = channel_raw_data.date.strftime('%d-%m-%y')
target_folder = '/home/ovinogradov/Projects/MEA-analysis/data/R1648H_filt/'
name_filtered = 'R1648H_{date}_{slice_}_{name}_filtered.h5'.format(date=date, slice_=slice_,name = name)
target_file= os.path.join(target_folder,name_filtered)
n_samples_in = analog_stream_0.timestamp_index[-1][-1]+1
#os.join '/home/ovinogradov/Projects/MEA-analysis/data'#'/mnt/server_epl/SCN1A/R1648H-export/20-3-25/slice2/2025-03-20T12-08-00McsRecording_filt.h5'


chunck_size = 2000000
n_samples_out = n_samples_in // subsampling_step
n_samples_out = int(n_samples_in // subsampling_step)

# with open('/Users/ovinogradov/Documents/Projects/SCN1A/MEA-analysis/data/Shaima_example/filt.bin', 'wb') as f_out:

# if chunck_size>subsampling_step:
    
with h5py.File(target_file, 'w') as write_file:
    dset = write_file.create_dataset(
        'data',
        shape=(n_samples_out, n_channels),             # start with 0 rows
        maxshape=(n_samples_out, n_channels),       # allow unlimited rows
        # chunks=(chunck_size, n_channels),   # efficient I/O chunks
        dtype='float32',
        compression='lzf'                 # optional: for smaller files
    )
    
with h5py.File(target_file, 'a') as write_file:
    dset = write_file['data']
    # Read chunk (adapt depending on file format)
    with h5py.File(file_name, "r") as f:
        data= f['Data']['Recording_0']['AnalogStream']['Stream_0']['ChannelData']
        chunnks = get_all_chunk_bounds(data.shape[1],chunck_size)        #  # Choose an optimal chunk size
        current_rows = 0
        for k in chunnks[:]:
            data_chunk = data[:,k[0]:k[1]]  # Process in chunks
            filtered_chunk = sosfiltfilt(b, data_chunk[:,:].T,axis=0)
            filtered_chunk = filtered_chunk[::subsampling_step,:]
            new_rows = current_rows + filtered_chunk.shape[0]
            dset[current_rows:new_rows, :] = filtered_chunk.astype(np.float32)
            current_rows = new_rows
            
    write_file.attrs['resampled_rate'] = subsampling_rate
    write_file.attrs['original_rate'] = fs
    write_file.attrs['filter'] = 'lowpass 200Hz, order=5, zero-phase, sos'
    write_file.attrs['scale'] = adc_step # votage step
    write_file.attrs['n_channels'] = n_channels
    write_file.attrs['date_time'] =  date_time.strftime('%Y-%m-%d %H:%M:%S.%f')