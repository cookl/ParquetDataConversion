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
import importlib
from array import array
from DataProcessingTools import BeamSpillLEDIntervals, find_beam_triggers
import argparse
from makeLEDDictFormat import MakeLEDDict
import pickle
from ROOTOutputTools import ROOTOutput 

parser = argparse.ArgumentParser(description="Process some arguments.")
parser.add_argument("-d", "--directory", required=True, help="Specify the directory")
parser.add_argument("-r", "--runno", type=int, required=True, help="Specify the run number (integer)")
parser.add_argument("-o", "--outDir", required=True, help="Specify the out directory")
args = parser.parse_args()

process_beam = True
process_led = True

directory = args.directory
runNumber = args.runno

#get the coarse counter intervals over which beam events or LED events were written out 
beamSpillLEDIntervals = BeamSpillLEDIntervals(directory,runNumber)

#produce some debugging histograms showing the total coarse counts and the the card ids
waveform_coarse_data, waveform_card_id_data = beamSpillLEDIntervals.loadCoarseCountsData("waveforms",loadCardID=True)
runTime = (waveform_coarse_data.max()-waveform_coarse_data.min())*8e-9
print("Difference between smallest and largest coarse counts",runTime,"seconds")
maxCard = waveform_card_id_data.max()
print("Maximum Card ID",maxCard)

if(runTime>60*60):
    raise Exception("Run time is longer than 1 hr - probably an error",runTime )
if(maxCard>135):
    raise Exception("Max card ID is - likely corrupt data",maxCard)

plt.hist(waveform_coarse_data, bins=100)
plt.xlabel("Waveform coarse count")
plt.ylabel("frequency")
plt.savefig(args.outDir+"/"+str(runNumber)+"_CoarseCountDist.png")
plt.close()

plt.hist(waveform_card_id_data, bins=200)
plt.xlabel("Card ID Distribution")
plt.ylabel("frequency")
plt.savefig(args.outDir+"/"+str(runNumber)+"_CardIDDist.png")
plt.close()

#determine the beamspill and led intervals from the file         
beam_spill_intervals, led_intervals, readout_times = beamSpillLEDIntervals.determineBeamLEDIntervals()

#more debugging and making plots of the timing of beam spill readout
print("Run",runNumber,"Found",len(beam_spill_intervals),"Beam spills",len(led_intervals),"LED flashing periods")
if(len(beam_spill_intervals)>100):
    raise Exception("Too many beam spills found")

beam_spill_readout = [
    readout_time
    for readout_time in readout_times
    for interval in beam_spill_intervals
    if interval[0] <= readout_time < interval[1]
]
beam_spill_readout= np.array(beam_spill_readout)
plt.plot(beam_spill_readout*8e-9, marker='x', linestyle='none', label="Total "+str(len(beam_spill_intervals))+" Beam Spills")
plt.xlabel("n Beam Spill Readout")
plt.ylabel("Readout time [s]")
plt.legend()
plt.savefig(args.outDir+"/"+str(runNumber)+"_beamspillReadout.png")
plt.close()

#setup some diagnostic plots
spill_no =[]
trigger_count = []
n_chan_readout = []
mean_n_chan_readout = []
chan_bins = np.arange(0, 2000, 20)

#coarse counter threshold to group events with the same event
event_window_threshold = 500

if process_beam:
    rootOutput = ROOTOutput()
    for readout_no in range(len(beam_spill_intervals)):
        print("*******Processing beamSpill ",readout_no,"of",len(beam_spill_intervals))

        coarse_min = beam_spill_intervals[readout_no,0]  
        coarse_max = beam_spill_intervals[readout_no,1]
        #load a specific beam spill basesd on the coarse count intervals above
        interval_waveforms_df = beamSpillLEDIntervals.load_specific_spill(coarse_min, coarse_max)

        #find beam triggers in that interval
        triggers = find_beam_triggers(interval_waveforms_df)
        print(len(triggers),"triggers found")
        
        spill_no.append(readout_no)
        trigger_count.append(len(triggers))
        event_n_chan_readout = []
        #go through the triggers and build events which are within event_window_threshold of the event 
        for itrigger, trigger in enumerate(triggers):
            
            trigger_time=trigger['coarse']
            readout_window_df = interval_waveforms_df[(interval_waveforms_df['coarse']<(trigger_time+event_window_threshold))&(interval_waveforms_df['coarse']>(trigger_time-event_window_threshold))]

            unique_pmt_ids = (100*readout_window_df['card_id'])+readout_window_df['chan']
            # print("Channels reading out",len(unique_pmt_ids))
            if len(np.unique(unique_pmt_ids))!= len(unique_pmt_ids):
                #one channel reporting more than once
                print("One channel reports multiple times in readout window")
            event_n_chan_readout.append(len(unique_pmt_ids))
            
            #write info out to rootOutput
            rootOutput.addTriggerDetails(trigger_time*8.0,runNumber,readout_no,itrigger)
            rootOutput.addHitsFromReadoutWindow(trigger_time,readout_window_df)
            rootOutput.fillTTree()
        
        #histogram showing the number of channels reading out in each event    
        hist, bin_edges = np.histogram(event_n_chan_readout, bins=chan_bins)
        n_chan_readout.append(hist)
        mean_channels_readout = 0
        if(len(event_n_chan_readout)>0):
            mean_channels_readout = np.mean(event_n_chan_readout)
        mean_n_chan_readout.append(mean_channels_readout)

    #write out root file
    rootOutput.writeTTreeFile(args.outDir+"/"+str(runNumber)+"_beamEvents.root")        
 
    #make some diagnostic plots for the runs
    plt.plot(spill_no, trigger_count, marker="x", linestyle = 'none')
    plt.xlabel("Spill No.")
    plt.ylabel("Trigger Count")
    plt.savefig(args.outDir+"/"+str(runNumber)+"_TriggerCount.png")
    plt.close()
    print(np.array(n_chan_readout).shape)
    plt.plot(spill_no,mean_n_chan_readout, label = "Mean")
    plt.pcolor(spill_no, (chan_bins[1:]+chan_bins[:-1])/2.0, np.array(n_chan_readout).T, cmap="YlOrBr")
    plt.xlabel("Spill No.")
    plt.ylabel("n Channel Readout")
    plt.colorbar()
    plt.savefig(args.outDir+"/"+str(runNumber)+"_ChannelReadout.png")
    plt.close()


if process_led:
    #now process the LED data
    led_df = beamSpillLEDIntervals.load_led_data()

    for readout_no in range(len(led_intervals)):
        print("Process first LED flashing")
        coarse_min = led_intervals[readout_no,0]  
        coarse_max = led_intervals[readout_no,1]
        #load a specific beam spill basesd on the coarse count intervals above
        interval_waveforms_df = beamSpillLEDIntervals.load_specific_spill(coarse_min, coarse_max)
        interval_led = led_df[(led_df['coarse'] >= coarse_min) & (led_df['coarse'] <= coarse_max)]
        
        makeLEDDict = MakeLEDDict(interval_waveforms_df,interval_led)
        interspill_led_dict = makeLEDDict.buildLEDDict()
        
        
        outfile = args.outDir+"/"+str(runNumber)+"_interspill_LED_"+str(readout_no)+".dict"
        with open(outfile, 'wb') as f:
            pickle.dump(interspill_led_dict, f, protocol=4)
        
    