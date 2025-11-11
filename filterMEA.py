# Filtergin and subsampling MEA data
import sys, importlib, os
import McsPy.McsData
import McsPy.McsCMOS
from McsPy import ureg, Q_
import numpy as np
from scipy import signal
import pandas as pd
import numpy as np
import scipy.signal
na = np.array
import h5py
import os
import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi
from scipy.signal import butter, sosfiltfilt
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt
from McsPy import ureg, Q_
from tqdm import tqdm
import json

# Define filters
def get_all_chunk_bounds(total_length, chunk_size):
    chunks = []
    for i in range(0, total_length, chunk_size):
        start = i
        end = min(i + chunk_size, total_length)
        #Skip the last chunk if that's smaller than chunk_size
        if (end-start)==chunk_size:
            chunks.append((start, end))
    return chunks

# Define filters
def get_filter_chunk_bounds(total_length,
                            chunk_size,overlap=500):
    """ get chunks with overlap for filtering"""
    chunks = []
    for i in range(0, total_length, chunk_size):
        if i<overlap:
            start = i
        else:
            start = i-overlap
        end = min(i + chunk_size+overlap, total_length)
        #Skip the last chunk if that's smaller than chunk_size
        # if (end-start)==chunk_size+overlap:
        chunks.append((start, end))
    return chunks


def design_filter(highcut, fs, order=5):
    nyq = 0.5 * fs
    b = butter(order, highcut / nyq, btype='low', output='sos')
    return b

def design_notch_filter(freq, fs, quality=30):
    b, a = iirnotch(freq, quality, fs)
    return b, a


def filter_subsample(file_name,
                     subsampling_rate=2000,
                     overwrite = True,
                     low_pass_freq=200,
                     target_folder = '/home/ovinogradov/Projects/MEA-analysis/data/test'
                     
                     ):
    #/Volumes/Epileptology01/Oleg/KCNQ/MEAdata/filt
    """ 
    Low-pass filter MEA data 
    <400Hz
    """
    channel_raw_data = McsPy.McsData.RawData(file_name)    
    analog_stream_0 = channel_raw_data.recordings[0].analog_streams[0]
    adc_step = analog_stream_0.channel_infos[2].adc_step.magnitude 
    fs = na(analog_stream_0.channel_infos[2].sampling_frequency) # get frequency
    signal = channel_raw_data.recordings[0].analog_streams[0].get_channel_in_range(5, 1, 2)
    scale_factor_for_uV = Q_(1,signal[1]).to(ureg.uV).magnitude
    date_time =channel_raw_data.date
    n_channels=252
    b = design_filter(low_pass_freq, fs)
    print('sampling rate',fs)
    subsampling_step = int(fs/subsampling_rate) # 

    slice_ =file_name.split('/')[-2]
    name = file_name.split('/')[-1]#'2025-03-20T14-07-28McsRecording'
    date = formatted = channel_raw_data.date.strftime('%d-%m-%y')
    name_filtered = 'R1648H_{date}_{slice_}_{name}_filtered_orig1.h5'.format(date=date, slice_=slice_,name = name)
    target_file= os.path.join(target_folder,name_filtered)
    
    with h5py.File(file_name, "r") as f:
        data= f['Data']['Recording_0']['AnalogStream']['Stream_0']['ChannelData']
        n_samples_in = np.shape(data)[1]
    n_samples_out = len(np.arange(n_samples_in)[::subsampling_step])#int(n_samples_in // subsampling_step)

    print('subsampling_step',subsampling_step)
    # Global subsamling schema 
    # predifine chunks that have qual boundaries
    chunk_size = 100_000#1000000
    
    if os.path.isfile(target_file) and overwrite==False:
        print("File exists.")
    else:
        
        with h5py.File(target_file, 'w') as write_file:
            dset = write_file.create_dataset(
                'data',
                shape=(n_samples_out, n_channels),             # start with 0 rows
                maxshape=(None, n_channels),       # allow unlimited rows
                # chunks=(chunck_size, n_channels),   # efficient I/O chunks
                dtype='float32',
                compression='lzf'                 # optional: for smaller files
            )
        with h5py.File(target_file, 'a') as write_file:
            
            dset = write_file['data']
            # Read chunk (adapt depending on file format)
            with h5py.File(file_name, "r") as f:
                data= f['Data']['Recording_0']['AnalogStream']['Stream_0']['ChannelData']
                overlap = 500
                filter_chunks = get_filter_chunk_bounds(data.shape[1],chunk_size,overlap=overlap)
                # chunks = get_all_chunk_bounds(data.shape[1],chunck_size) #Choose an optimal chunk size
                current_rows = 0
                # print(filter_chunks)
                # print('total number',len(chunnks))
                for k_i,k in enumerate(filter_chunks[:]):
                    data_chunk = data[:,k[0]:k[1]]  # Process in chunks
                    filtered_chunk = sosfiltfilt(b, data_chunk[:,:].T,axis=0)
                    if k[0]<overlap:
                        overlap_ = 0
                    else:
                        overlap_ = overlap
                    filtered_chunk = filtered_chunk[overlap_:-overlap,:]
                    filtered_chunk = filtered_chunk[::subsampling_step,:]
                    # TODO MAKE SURE THAT THE ENEVEN CHUNKS ARE MEANINGFULLY PROCESSED SOMEHOW
                    # TEST THE OUTPUT FOR CORRUPTION ETC
                    
                    # print(len(filtered_chunk))
                    # if len(filtered_chunk)<chunk_size:
                    #     print(k_i,k,len(filter_chunks))
                    # assert(len(filtered_chunk)==chunk_size)
                    new_rows = current_rows + filtered_chunk.shape[0]
                    if new_rows> dset.shape[0]:
                        # Resize if needed — only once for final chunk
                        dset.resize((new_rows,n_channels))
                    dset[current_rows:new_rows, :] = filtered_chunk.astype(np.float32)
                    current_rows = new_rows
        #seperately add paramters
        with h5py.File(target_file, 'a') as write_file:
            write_file['resampled_rate'] = subsampling_rate
            write_file['original_rate'] = fs
            write_file['filter'] = 'lowpass %sHz, order=5, zero-phase, sos'%low_pass_freq
            write_file['scale'] = adc_step # votage step
            write_file['n_channels'] = n_channels
            write_file['date_time'] =  date_time.strftime('%Y-%m-%d %H:%M:%S.%f')
            write_file['scale_factor_for_uV'] = scale_factor_for_uV
            # # save channel dict
            # channels = []
            channels_dict ={}
            for i in list(channel_raw_data.recordings[0].analog_streams[0].channel_infos.keys()):
                channels_dict[int(i)]= channel_raw_data.recordings[0].analog_streams[0].channel_infos[i].label
        
            dset_channels = write_file.create_dataset(
                'channels',
                data=json.dumps(channels_dict)
            )

        
        return None

def process_file(file_path):
    print(f"\nProcessing: {file_path}")
    if not os.path.exists(file_path):
        print("  ❌ File not found!")
        return
    with open(file_path, 'r') as f:
        # lines = f.readlines()
        lines = [line.strip() for line in f.readlines()]
    num_lines = len(lines)
    print(f"  📄 Number of files to process: {num_lines}")
    return lines

def main():
    if len(sys.argv) < 2:
        print("Usage: python process_txt.py file1.txt file2.txt ...")
        sys.exit(1)
    txt_files = sys.argv[1:]
    for file_path in txt_files:
        lines= process_file(file_path)
    # target_folder = '/home/ovinogradov/Projects/MEA-analysis/data/R1648H_filt/'#'/mnt/server_epl/KCNQ/MEAdata/filt/'
    target_folder = '/home/ovinogradov/Projects/MEA-analysis/data/test'
    # target_folder = '/home/ovinogradov/Projects/MEA-analysis/data/test/'#'/mnt/server_epl/KCNQ/MEAdata/filt/'
    for i, file_name in enumerate(tqdm(lines, total=len(lines))):
        # try:
        filter_subsample(file_name, subsampling_rate=2000,
                         overwrite = True,
                         target_folder = target_folder)
        # except:
        #     print('%s could '%file_name)

if __name__ == "__main__":
    main()