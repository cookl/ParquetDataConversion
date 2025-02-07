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
# import DataProcessingTools
import ROOT 
from array import array
from DataProcessingTools import BeamSpillLEDIntervals, find_beam_triggers, MPMTMapping
import argparse


parser = argparse.ArgumentParser(description="Process some arguments.")
# Add arguments
parser.add_argument("-d", "--directory", required=True, help="Specify the directory")
parser.add_argument("-r", "--runno", type=int, required=True, help="Specify the run number (integer)")
# parser.add_argument("-o", "--outDir", required=True, help="Specify the out directory")
# Parse arguments
args = parser.parse_args()

#details on the run
# directory = "/eos/experiment/wcte/wcte_tests/mPMT_periodic_readout/2024-11-26"
# runNumber = 20241126165657
directory = args.directory
runNumber = args.runno


#load the mPMT mapping
file_path = "../mappings/WCTE_PMT_Mapping/PMT_Mapping.json"
mPMTMapping = MPMTMapping(file_path)


# Create a ROOT file
file = ROOT.TFile(str(runNumber)+"_beamEvents.root", "RECREATE")

#initialise ttree for saving
tree = ROOT.TTree("WCTEReadoutWindows", "Readout of WCTE data")
#event details 
window_time = array('d',[0])
run_id = array('L',[0])
spill_counter = array('i',[0])
event_number = array('i',[0])
tree.Branch("window_time", window_time, "window_time/D")
tree.Branch("run_id", run_id, "run_id/L")
tree.Branch("spill_counter", spill_counter, "spill_counter/I")
tree.Branch("event_number", event_number, "event_number/I")

#pmt_waveforms one entry per waveform
pmt_waveform_mpmt_card_ids = ROOT.std.vector('int')()
pmt_waveform_pmt_channel_ids = ROOT.std.vector('int')()
pmt_waveform_mpmt_slot_ids = ROOT.std.vector('int')()
pmt_waveform_pmt_position_ids = ROOT.std.vector('int')()
pmt_waveform_times = ROOT.std.vector('double')()
pmt_waveforms = ROOT.std.vector('std::vector<int>')()  # This creates a C++ std::vector<std::vector<double>>
tree.Branch("pmt_waveform_mpmt_card_ids", pmt_waveform_mpmt_card_ids)
tree.Branch("pmt_waveform_pmt_channel_ids", pmt_waveform_pmt_channel_ids)
tree.Branch("pmt_waveform_mpmt_slot_ids", pmt_waveform_mpmt_slot_ids)
tree.Branch("pmt_waveform_pmt_position_ids", pmt_waveform_pmt_position_ids)
tree.Branch("pmt_waveform_times", pmt_waveform_times)
tree.Branch("pmt_waveforms", pmt_waveforms)


#get the coarse counter intervals over which beam events or LED events were written out 
beamSpillLEDIntervals = BeamSpillLEDIntervals(directory,runNumber)
beam_spill_intervals, led_intervals = beamSpillLEDIntervals.determineBeamLEDIntervals()

print("Run",runNumber,"Found",len(beam_spill_intervals),"Beam spills",len(led_intervals),"LED flashing periods")
if(len(beam_spill_intervals)>100):
    raise Exception("Too many beam spills found")
#make some diagnostic plots
spill_no =[]
trigger_count = []
n_chan_readout = []
chan_bins = np.arange(0, 2000, 20)
#coarse counter threshold to group events with the same event
event_window_threshold = 500


for readout_no in range(len(beam_spill_intervals)):
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
        print("n mpmt boards reporting", len(np.unique(readout_window_df['card_id'])))
        event_n_chan_readout.append(len(unique_pmt_ids))
        print("trigger n chan ",len(unique_pmt_ids) )
        #write info out to ttree the [0] is necessary for writing to ttree in pyroot
        window_time[0] = trigger_time*8.0 #convert to ns
        run_id[0] = runNumber
        spill_counter[0] = readout_no
        event_number[0] = itrigger

        #clear the waveform vectors
        pmt_waveform_mpmt_card_ids.clear()
        pmt_waveform_pmt_channel_ids.clear()
        pmt_waveform_mpmt_slot_ids.clear()
        pmt_waveform_pmt_position_ids.clear()
        pmt_waveform_times.clear()
        pmt_waveforms.clear()

        #loop over all the waveforms 
        for wf, wf_coarse, wf_card, wf_chan in zip(readout_window_df['samples'],readout_window_df['coarse'],readout_window_df['card_id'],readout_window_df['chan']):
            
            pmt_waveform_mpmt_card_ids.push_back(wf_card)
            pmt_waveform_pmt_channel_ids.push_back(wf_chan)

            slot_id, pmt_pos_id = mPMTMapping.getMPMTPos(wf_card,wf_chan)
            pmt_waveform_mpmt_slot_ids.push_back(slot_id)    
            pmt_waveform_pmt_position_ids.push_back(pmt_pos_id)    
            
            waveform_time = (wf_coarse-trigger_time)*8.0  #convert to ns
            # print("waveform_time",waveform_time,type(waveform_time))

            pmt_waveform_times.push_back(waveform_time)

            #process the waveform 
            single_wavform_vector = ROOT.std.vector('int')()  # Create vector to store single waveform in
            for sample in wf:  # Inner loop, fill with increasing integers
                # print("sample",sample,type(sample))
                single_wavform_vector.push_back(int(sample))
            pmt_waveforms.push_back(single_wavform_vector)
    
        tree.Fill()
    hist, bin_edges = np.histogram(event_n_chan_readout, bins=chan_bins)
    print("hist.shape",hist.shape)
    n_chan_readout.append(hist)
        
tree.Write()
file.Close()
#make some diagnostic plots for the runs
plt.plot(spill_no, trigger_count, marker="x", linestyle = 'none')
plt.xlabel("Spill No.")
plt.ylabel("Trigger Count")
plt.savefig(str(runNumber)+"_TriggerCount.png")
plt.close()
print(np.array(n_chan_readout).shape)
# plt.plot(event_n_chan_readout)
plt.pcolor(spill_no, (chan_bins[1:]+chan_bins[:-1])/2.0, np.array(n_chan_readout).T, cmap="YlOrBr")
plt.xlabel("Spill No.")
plt.ylabel("n Channel Readout")
plt.colorbar()
plt.savefig(str(runNumber)+"_ChannelReadout.png")
plt.close()