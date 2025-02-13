import ROOT 
from array import array
from DataProcessingTools import MPMTMapping


class ROOTOutput:
    
    def __init__(self):
        self.initTTree()

        #mPMT mapping needed for slot and position IDs
        file_path = "../mappings/WCTE_PMT_Mapping/PMT_Mapping.json"
        self.mPMTMapping = MPMTMapping(file_path)
    
    def initTTree(self):
        #initialise ttree for saving
        self.tree = ROOT.TTree("WCTEReadoutWindows", "Readout of WCTE data")
        #event details 
        self.window_time = array('d',[0])
        self.run_id = array('L',[0])
        self.spill_counter = array('i',[0])
        self.event_number = array('i',[0])
        self.tree.Branch("window_time", self.window_time, "window_time/D")
        self.tree.Branch("run_id", self.run_id, "run_id/L")
        self.tree.Branch("spill_counter", self.spill_counter, "spill_counter/I")
        self.tree.Branch("event_number", self.event_number, "event_number/I")

        #pmt_waveforms one entry per waveform
        self.pmt_waveform_mpmt_card_ids = ROOT.std.vector('int')()
        self.pmt_waveform_pmt_channel_ids = ROOT.std.vector('int')()
        self.pmt_waveform_mpmt_slot_ids = ROOT.std.vector('int')()
        self.pmt_waveform_pmt_position_ids = ROOT.std.vector('int')()
        self.pmt_waveform_times = ROOT.std.vector('double')()
        self.pmt_waveforms = ROOT.std.vector('std::vector<int>')()  # This creates a C++ std::vector<std::vector<double>>
        self.tree.Branch("pmt_waveform_mpmt_card_ids", self.pmt_waveform_mpmt_card_ids)
        self.tree.Branch("pmt_waveform_pmt_channel_ids", self.pmt_waveform_pmt_channel_ids)
        self.tree.Branch("pmt_waveform_mpmt_slot_ids", self.pmt_waveform_mpmt_slot_ids)
        self.tree.Branch("pmt_waveform_pmt_position_ids", self.pmt_waveform_pmt_position_ids)
        self.tree.Branch("pmt_waveform_times", self.pmt_waveform_times)
        self.tree.Branch("pmt_waveforms", self.pmt_waveforms)
        
    def addTriggerDetails(self,trigger_time_ns,runNumber,readout_no,event_number):
        #write info out to ttree the [0] is necessary for writing to ttree in pyroot
        self.window_time[0] = trigger_time_ns
        self.run_id[0] = runNumber
        self.spill_counter[0] = readout_no
        self.event_number[0] = event_number
    
    def addHitsFromReadoutWindow(self,trigger_time,readout_window_df):
        self.pmt_waveform_mpmt_card_ids.clear()
        self.pmt_waveform_pmt_channel_ids.clear()
        self.pmt_waveform_mpmt_slot_ids.clear()
        self.pmt_waveform_pmt_position_ids.clear()
        self.pmt_waveform_times.clear()
        self.pmt_waveforms.clear()

        #loop over all the waveforms 
        for wf, wf_coarse, wf_card, wf_chan in zip(readout_window_df['samples'],readout_window_df['coarse'],readout_window_df['card_id'],readout_window_df['chan']):
            
            self.pmt_waveform_mpmt_card_ids.push_back(wf_card)
            self.pmt_waveform_pmt_channel_ids.push_back(wf_chan)

            #get the slot and position ids from the mapping
            slot_id, pmt_pos_id = self.mPMTMapping.getMPMTPos(wf_card,wf_chan)
            self.pmt_waveform_mpmt_slot_ids.push_back(slot_id)    
            self.pmt_waveform_pmt_position_ids.push_back(pmt_pos_id)    
            
            #care since coarse counter is unsigned int and was causing issues
            waveform_time = 0
            if(wf_coarse>trigger_time):
                waveform_time = (wf_coarse-trigger_time)*8.0
            else:
                waveform_time = (trigger_time-wf_coarse)*-8.0
                
            self.pmt_waveform_times.push_back(waveform_time)

            #process the waveform 
            single_wavform_vector = ROOT.std.vector('int')()  # Create vector to store single waveform in
            for sample in wf: 
                single_wavform_vector.push_back(int(sample))
            self.pmt_waveforms.push_back(single_wavform_vector)
    
    def fillTTree(self):
        self.tree.Fill()
        
    def writeTTreeFile(self,fileName):
        file = ROOT.TFile(fileName, "RECREATE")
        self.tree.Write()
        file.Close()
        
        