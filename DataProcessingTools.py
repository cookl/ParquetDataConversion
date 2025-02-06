import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib.colors import LogNorm
import time
import pyarrow.parquet as pq
import gc
import os
import glob

class BeamSpillLEDIntervals:
    #this class calculates the readout times to figure out when a readout is due to beam spill or LED event
    #it has  function load_specific_spill to load waveforms for the run from a specific beam spill 

    def __init__(self, directory, runNumber):
        self.directory = directory
        self.runNumber = runNumber
        # self.nSubruns = nSubruns
        
        # file_pattern_led = self.directory+"/periodic_readout_"+str(self.runNumber)+"_*_led.parquet"  # assuming the files have .json extension
        # file_pattern_waveforms = self.directory+"/periodic_readout_"+str(self.runNumber)+"_*_waveforms.parquet"  # assuming the files have .json extension
        
    
    def loadCoarseCountsData(self, fileType):
        #fileType either waveforms or led
        if fileType not in ["waveforms", "led"]:
            raise ValueError(f"Invalid fileType: {fileType}. Must be 'waveforms' or 'led'.")
        
        allWindows = []
        all_coarse_data = []
        file_pattern = self.directory+"/periodic_readout_"+str(self.runNumber)+"_*_"+fileType+".parquet"
        files = glob.glob(file_pattern)

        for file_path in files:
            # file_stats={}
            # file_stats['subRunNo'] = subNo
            # file_path = self.directory+"/periodic_readout_"+str(self.runNumber)+"_"+str(subNo)+"_"+fileType+".parquet"
            if not os.path.exists(file_path):
                print(f"Error: File {file_path} does not exist. Skipping this file.")
                continue  # Skip this iteration and move to the next one
                
            # Load only the 'coarse' column
            try:
                table = pq.read_table(file_path, columns=['coarse'])
            except:
                print("Skipping failed file",file_path )
                continue
            
            # # Convert to a NumPy array for fast processing
            coarse_data = table.column('coarse').to_numpy()
            all_coarse_data.append(coarse_data)
            
        waveform_coarse_data = np.concatenate(all_coarse_data)
        return waveform_coarse_data
        
    def find_readout_times(self,coarse_counts):
        #this function takes a long vector of coarse counts where readout happened 
        #it hitograms these readout times and groups them if they are within 3s of each other
        #the purpose is to get the times of LED or Beamspill readout during a run and enumerate the spills
        
        #first histogram to find the rough bins
        xmin = 0
        xmax = coarse_counts.max()
        step = 1.0/(8e-9) #1s bins
        bins = np.arange(xmin,xmax+step,step)
        counts, bin_edges = np.histogram(coarse_counts, bins=bins)
        bin_centers = (bin_edges[1:]+bin_edges[:-1])/2
        all_coarse_peaks = bin_centers[counts > 0]

        # determine the readout times allowing for the fact that some readouts may be between multiple bins
        threshold_to_group_peaks = 3/(8e-9) #if peaks are within 3s of each other make them the same peak
                                            #this is done to solve the issue when histogramming and the peak falls between two bins 
        coarse_peaks = []
        current_batch = []
        
        for peak in all_coarse_peaks:
            
            if len(current_batch)==0:
                current_batch.append(peak)
            else:
                #already a peak or peaks in the current batch 

                batch_average = (sum(current_batch) / len(current_batch))
                if abs(batch_average-peak) < threshold_to_group_peaks:
                    #current peak is within the threshold of the previous peaks and should be added to them 
                    current_batch.append(peak)
                
                else:
                    #current peak is outside the threshold of the previous peaks and should be added to them
                    coarse_peaks.append(batch_average)
                    current_batch = []
                    current_batch.append(peak)
        
        #append the last peak 
        batch_average = (sum(current_batch) / len(current_batch))
        coarse_peaks.append(batch_average)

        return coarse_peaks

    def make_readout_intervals(self,readout_times):
        #this function takes as input the readout times from find_readout_times and creates a 
        # range of times from which it can be determined which spill a single coarse count corresponds to 
        #from a list of times determine the range for each readout period
        readout_intervals = np.zeros((len(readout_times),2))
        
        lower_limit =0
        for iReadout, time in enumerate(readout_times):
            readout_intervals[iReadout,0] = lower_limit
            
            if(iReadout != len(readout_times)-1):
                next_readout = readout_times[iReadout+1]
            else:
                #at the last entry make the next readout 20s after the current readout
                next_readout = time+(20/8e-9)
            
            upper_limit = (next_readout+time)/2 # halfway between next and current readout
            
            readout_intervals[iReadout,1] = upper_limit
            #next iteration make the lower limit the upper limit of this one
            lower_limit = upper_limit

        return readout_intervals
    
    def determineBeamLEDIntervals(self):
        print("Determining times of LED and beam spill")
        waveform_coarse_data = self.loadCoarseCountsData("waveforms")
        led_coarse_data = self.loadCoarseCountsData("led")
        
        readout_times = self.find_readout_times(waveform_coarse_data)
        readout_intervals = self.make_readout_intervals(readout_times)
        led_readout_times = self.find_readout_times(led_coarse_data)
        
        #separate readout intervals into intervals of readout for beam and LED
        led_intervals =[]
        beam_spill_intervals =[]

        for interval in readout_intervals:

            #loop through LED times and see if it is coincident
            led_in_interval =-1
            for iled, led_readout in enumerate(led_readout_times):
                if(led_readout>interval[0] and led_readout<interval[1]):
                    #led readout is in the interval so this is an LED event 
                    led_in_interval=iled
            
            if(led_in_interval>=0):
                led_intervals.append(interval)
            else:
                beam_spill_intervals.append(interval)

        beam_spill_intervals = np.array(beam_spill_intervals)    
        led_intervals = np.array(led_intervals)
        
        return beam_spill_intervals, led_intervals
        

    #This function will load all waveforms between specific coarse counters, NB due to large size this is intented to be used for a single spill 
    #if the range is very large it is likely to crash 
    def load_specific_spill(self,coarse_min,coarse_max):
        
        #first just load the coarse column to check which files need to be loaded 
        file_list = []
        
        file_pattern = self.directory+"/periodic_readout_"+str(self.runNumber)+"_*_waveforms.parquet"
        files = glob.glob(file_pattern)

        for file_path in files:
            
        # for subNo in range(nSubruns):
        #     file_path = directory+"/periodic_readout_"+str(runNumber)+"_"+str(subNo)+"_waveforms.parquet"
            
            if not os.path.exists(file_path):
                print(f"Error: File {file_path} does not exist. Skipping this file.")
                continue  # Skip this iteration and move to the next one
        
            
            # Load only the 'coarse' column
            try:
                table = pq.read_table(file_path, columns=['coarse'])
            except:
                print("Skipping failed file",file_path )
                continue
                
            # # Convert to a NumPy array for fast processing
            coarse_data = table.column('coarse').to_numpy()
            count_in_range = np.sum((coarse_data >= coarse_min) & (coarse_data <= coarse_max))
            
            if(count_in_range>0):
                print("File",file_path,"relevant")
                file_list.append(file_path)
        
        #now load in only those waveforms 
        df_list = []
        chunk_size = 10000
        
        for file_path in file_list:
            parquet_file = pq.ParquetFile(file_path)
            print("opened file")
            
            total_rows = parquet_file.metadata.num_rows
            batches = (total_rows // chunk_size) + 1
            
            for i, batch in enumerate(parquet_file.iter_batches(batch_size=chunk_size)):
            # Process the chunk (batch)
                if(i%100==0):
                    print("Load", i,"/",batches)
                        
                df = batch.to_pandas(split_blocks=True, self_destruct=True)
                filtered_df = df[(df['coarse'] >= coarse_min) & (df['coarse'] <= coarse_max)]
                
                # Append the filtered data to the list
                df_list.append(filtered_df)
                
                # Free up memory for this batch
                del df
                batch = None  # Release the batch object
            
        # Combine any remaining DataFrames
        interval_waveforms_df = pd.concat(df_list, ignore_index=True)

        return interval_waveforms_df    

#make the beam trigger by looking for coincidences
#look for coincidences across the 4 T1 PMTs 131 15-18
def find_beam_triggers(interval_waveforms_df):
    
    def get_peak_timebins(waveform, threshold):
        # use the most frequent waveform value as the baseline
        values, counts = np.unique(waveform, return_counts=True)
        baseline = values[np.argmax(counts)]
        # baseline - waveform is positive going signal typically around 0 when there is no signal
        # threshold is the minimum positive signal above baseline

        above = (waveform[0]-baseline) <= threshold
        peak_timebins = []
        max_val = 0
        max_timebin = -1
        for i in range(len(waveform)):
            if above and (waveform[i]-baseline) > threshold:
                above = False
                max_val = 0
                max_timebin = -1
            if not above:
                if (waveform[i]-baseline) > max_val:
                    max_val = waveform[i] -baseline
                    max_timebin = i
                if (waveform[i]-baseline) <= threshold:
                    above = True
                    peak_timebins.append(max_timebin)
        return peak_timebins

    #debug counter
    window_with_4_pulses =0
    #returns an array of dictonaries where coincidences exist between the 
    trigger_peak_threshold = 20
    trigger_peak_coincidence = 10
    
    card_131_df = interval_waveforms_df[(interval_waveforms_df['card_id'] == 131) & 
                                    (interval_waveforms_df['chan'].isin([15, 16, 17, 18]))]
    
    #find coarse times at when we readout from board 131
    unique_coarse_values = card_131_df['coarse'].unique()
    unique_coarse_values_sorted = sorted(unique_coarse_values)
    
    trigger = []
    
    for window_coarse in unique_coarse_values_sorted:
        
        window_df= card_131_df[card_131_df["coarse"]==window_coarse]
        
        if(sorted(window_df['chan'].tolist())!=[15, 16, 17, 18]):
            print("Bad: bad windows",window_df['chan'].tolist())
            continue
    
        chan_lis = [15, 16, 17, 18]
        window_dict = {}
        waveforms_arr =[]
        peaks_arr = []
        n_peaks_arr = []
        
        is_peak_in_each_channel = True
        
        #look for peaks in each of the T1 PMT channels
        for chan in chan_lis:
            waveform = window_df[window_df['chan']==chan]['samples'].iloc[0]
            waveforms_arr.append(waveform)
            
            peaks = get_peak_timebins(waveform,30)
            peaks_arr.append(peaks)
            
            n_peaks = len(peaks)

            if(n_peaks==0): 
                is_peak_in_each_channel = False
                
            n_peaks_arr.append(n_peaks)
        
        if(is_peak_in_each_channel):
            window_with_4_pulses+=1
            #check whether the peak times are coincident
            is_coincident_time = False
            
            for time_ch_15 in peaks_arr[0]:
                #loop over each peak in the first channel (15)
                is_coincident_with_other_channels = True
                #check with other chanels 
                for chan_peak in peaks_arr[1:]:
                    if np.sum(np.abs(time_ch_15-np.array(chan_peak))<trigger_peak_coincidence)==0:
                        #no coincidence found between time_ch_15 and any of the peaks in other channel 
                        is_coincident_with_other_channels = False
                        break
                if(is_coincident_with_other_channels==True):
                    #we found a coincidence in this time by checking all 
                    #other channels and in every channel there was a coincidence
                    is_coincident_time = True
                    break
                
            if(is_coincident_time):
                #we found a coincidence between the 4 T1 PMTs
                window_dict['waveforms']= waveforms_arr
                window_dict['peaks']= peaks_arr
                window_dict['n_peaks']= n_peaks_arr
                window_dict['coarse']=window_coarse
                trigger.append(window_dict)
            # else:
            #     for waveform in waveforms_arr:
            #         plt.plot(waveform)
    print(window_with_4_pulses," windows with at least one pulse in each PMT")
    print(len(trigger)," windows with coincident pulses in each PMT")
          
    return trigger

class MPMTMapping:
    def __init__(self,file_path):
        # file_path = "../mappings/WCTE_PMT_Mapping/PMT_Mapping.json"
        # Load the JSON file
        with open(file_path, 'r') as file:
            self.pmt_mapping = json.load(file)["mapping"]
    
    def get_positions_from_entry(self, long_form_id):
        #longform id is just the mpmt_id*100 + pmt_id 
        return long_form_id//100, long_form_id%100
    
    def getMPMTPos(self, card_id, chan):
        long_form_value = (100 *  card_id) + chan
        slot_id = -1
        pmt_pos_id = -1
        if str(long_form_value) in self.pmt_mapping:
            slot_id, pmt_pos_id = self.get_positions_from_entry(self.pmt_mapping[str(long_form_value)])
        
        return slot_id, pmt_pos_id
    
    def get_key_from_value(self, input_value):
        for key, value in self.pmt_mapping.items():
            if value == input_value:
                return key
        return None  # Return None if no match is found  
        
            