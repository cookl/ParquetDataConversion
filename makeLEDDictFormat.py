import pyarrow.parquet as pq
import numpy as np
import pickle
from pathlib import Path
import bisect
import time


class MakeLEDDict:
    #class to process the LED dict interval by interval
    def __init__(self, interval_waveforms_df, interval_led_flashes):
        self.interval_waveforms_df = interval_waveforms_df
        self.interval_led_flashes = interval_led_flashes

    def buildLEDDict(self):
        analysis = 'led'
        
        #copied and pasted from Dean's process_parquet_v3.py
        #changes are to make the lists specific to each spill 
        # pql_card_id,pql_coarse,pql_led_no,pql_seq_no = pq.read_table(folder+'/'+prefix+run+'_led.parquet',columns=['card_id','coarse','led_no','seq_no'])
        self.interval_led_flashes = self.interval_led_flashes.sort_values(by='coarse', ascending=True)
        # led_card_id = self.interval_led_flashes['card_id'].tolist()
        # led_coarse = self.interval_led_flashes['coarse'].tolist()
        # led_no = self.interval_led_flashes['led_no'].tolist()
        # led_seq_no = self.interval_led_flashes['seq_no'].tolist()
        
        led_card_id = self.interval_led_flashes['card_id'].to_numpy()
        led_coarse = self.interval_led_flashes['coarse'].to_numpy()
        led_no = self.interval_led_flashes['led_no'].to_numpy()
        led_seq_no = self.interval_led_flashes['seq_no'].to_numpy()

        # Read the waveform data directly from the interval dataframe
        # card_id = self.interval_waveforms_df['card_id'].tolist()
        # chan = self.interval_waveforms_df['chan'].tolist()
        # samples = self.interval_waveforms_df['samples'].tolist()
        # coarse = self.interval_waveforms_df['coarse'].tolist()
        
        card_id = self.interval_waveforms_df['card_id'].to_numpy()
        chan = self.interval_waveforms_df['chan'].to_numpy()
        samples = self.interval_waveforms_df['samples'].to_numpy()
        coarse = self.interval_waveforms_df['coarse'].to_numpy()

        expected_dts = [10000,3125,5000,20000,25000,62500,2500000]

        sample_cards = {}

        if analysis == 'led':
            led_cards = {}
            led_nos = {}

            for dict,data in zip([sample_cards,led_cards,led_nos],[card_id,led_card_id,led_no]):
                for i in range(len(data)):
                    if data[i] not in dict:
                        dict[data[i]] = 1
                    else:
                        dict[data[i]] += 1
        
        if analysis == 'led':
            led_times_by_card = {}
            last_time_by_card = {}
            for i in range(len(led_card_id)):
                card = led_card_id[i]
                if card not in led_times_by_card:
                    led_times_by_card[card] = []
                    last_time_by_card[card] = 0
                dt = led_coarse[i] - last_time_by_card[card]
                if dt not in expected_dts:
                    led_times_by_card[card].append(led_coarse[i])
                last_time_by_card[card] = led_coarse[i]

            delta_led_times_by_card = {}
            for card in led_times_by_card:
                delta_led_times_by_card[card] = [led_times_by_card[card][i+1] - led_times_by_card[card][i] for i in range(len(led_times_by_card[card])-1)]


        sample_times_by_card ={}
        last_time_by_card = {}
        for i in range(len(coarse)):
            card = card_id[i]
            if card not in sample_times_by_card:
                sample_times_by_card[card] = []
                last_time_by_card[card] = 0
            dt = coarse[i] - last_time_by_card[card]
            if analysis != 'coincidences' and dt !=0 and dt not in expected_dts:
                sample_times_by_card[card].append(coarse[i])
            last_time_by_card[card] = coarse[i]

        delta_sample_times_by_card = {}
        for card in sample_times_by_card:
            delta_sample_times_by_card[card] = [sample_times_by_card[card][i+1] - sample_times_by_card[card][i] for i in range(len(sample_times_by_card[card])-1)]

        # Print summary of runs and delta times

        if analysis != 'no_print':
            LED_buff = 'none'
            if analysis == 'led':
                LED_buff = ",".join([str(card) for card in led_cards])
            SAMPLE_buff = ",".join([str(card) for card in sample_cards])
            # print('RUN:',run,' LED cards:',LED_buff,' SAMPLE cards:',SAMPLE_buff)
            print('LED cards:',LED_buff,' SAMPLE cards:',SAMPLE_buff)
            if analysis == 'led':
                print('*** LED summary ***')
                for card in led_cards:
                    print(f'Card {card} has {led_cards[card]} LED flashes')
                    if card in delta_led_times_by_card:
                        if len(delta_led_times_by_card[card])>0:
                            print(f'  Delta times: {delta_led_times_by_card[card]}')
                    else:
                        print(f'  No delta times')
            print('*** Sample summary ***')
            for card in sample_cards:
                print(f'Card {card} has {sample_cards[card]} samples')
                if card in delta_sample_times_by_card:
                    if len(delta_sample_times_by_card[card])>0:
                        print(f'  Delta times: {delta_sample_times_by_card[card]}')
                else:
                    print(f'  No delta times')

        # Analyze the waveform data with CFD

        cfd_raw_t = [0.16323452713658082,
                    0.20385733509493395,
                    0.24339187740767365,
                    0.2822514122310461,
                    0.3208335490313887,
                    0.35953379168152044,
                    0.3987592183841288,
                    0.4389432980060811,
                    0.4805630068163285,
                    0.5241597383052767,
                    0.5703660640730557,
                    0.6199413381955754,
                    0.6738206794685682,
                    0.7331844507933303,
                    0.7995598000823612,
                    0.874973724581176,
                    0.9621917102137131,
                    1.0301530251726216,
                    1.0769047405430523,
                    1.1210801763323819,
                    1.1632345271365807]

        # correction for the amplitude derived by summing the largest three adcs
        amp_raw_t = [2.0413475167493225, 2.0642014124776784, 2.0847238089021274, 2.1028869067818117, 2.118667914530039,
                    2.1320484585033723, 2.1430140317025583, 2.151553497195665, 2.1576586607668613, 2.1613239251470255,
                    2.162546035746829, 2.1613239251470255, 2.1576586607668617, 2.1515534971956654, 2.143014031702558,
                    2.1320484585033723, 2.118667914530039, 2.1028869067818117, 2.0847238089021274, 2.0642014124776784,
                    2.0413475167493225]

        cfd_true_t = [-0.5 + 0.05*i for i in range(21)]

        # the following is from process_root
        # note that the waveforms in the parquet file are already baseline subtracted and inverted
        # So reverse the sign of baseline and adcs in the get_cfd function

        def get_peak_timebins(waveform, threshold, card_id):
            # use the most frequent waveform value as the baseline
            values, counts = np.unique(waveform, return_counts=True)
            baseline = values[np.argmax(counts)]
            # baseline - waveform is positive going signal typically around 0 when there is no signal
            # threshold is the minimum positive signal above baseline

            below = (waveform[0] - baseline) <= threshold
            peak_timebins = []
            max_val = 0
            max_timebin = -1
            for i in range(len(waveform)):
                if below and (waveform[i] - baseline) > threshold:
                    below = False
                    max_val = 0
                    max_timebin = -1
                if not below:
                    if (waveform[i] - baseline) > max_val:
                        max_val = waveform[i] - baseline
                        max_timebin = i
                    if (waveform[i] - baseline) <= threshold:
                        below = True
                        #if brb != 3 or len(peak_timebins) == 0 or max_timebin > peak_timebins[-1] + 4:  # eliminate peaks from ringing... (for brb03)
                        if 1 == 1:
                            peak_timebins.append(max_timebin)
            return peak_timebins

        def get_cfd(adcs):
            # Use a cfd like algorithm with delay d = 2, multiplier c = -2
            c = -2.
            d = 2
            # for cfd just use the average of the first 3 adcs as the baseline
            baseline = (adcs[0] + adcs[1] + adcs[2]) / 3.
            # the amplitude is found by adding the highest 3 adcs and subtracting the baseline
            #amp = (baseline - np.min(adcs)) / 100.
            n_largest_vals = sorted(np.array(adcs)-baseline, reverse=True)[:3]
            amp = sum(n_largest_vals)
            # converting to positive going pulses
            data = [(adcs[i]-baseline) + c * (adcs[i - d]-baseline) for i in range(d, len(adcs))]
            # find largest swing zero crossing
            max_diff = 0
            i_md = -1
            for iv in range(1, len(data)):
                if data[iv - 1] > 0. and data[iv] < 0.:
                    if data[iv - 1] - data[iv] > max_diff:
                        max_diff = data[iv - 1] - data[iv]
                        i_md = iv

            if i_md > -1:
                x0 = i_md - 1
                y0 = data[i_md - 1]
                x1 = i_md
                y1 = data[i_md]

                # using a linear interpolation, find the value of x for which y = 0
                x = x0 - (x1 - x0) / (y1 - y0) * y0
                # apply offset assuming sigma = 0.96 (see try_cfd.ipynb)
                #x -= 0.5703
                # apply a correction
                apply_correction = True
                offset = 5.
                delta = x - offset
                t = None
                if apply_correction:
                    if cfd_raw_t[0] < delta < cfd_raw_t[-1]:
                        correct_t = np.interp(delta,cfd_raw_t,cfd_true_t)
                        t = offset + correct_t
                    elif delta < cfd_raw_t[0]:
                        delta += 1
                        if cfd_raw_t[0] < delta < cfd_raw_t[-1]:
                            correct_t = np.interp(delta, cfd_raw_t, cfd_true_t)
                            t = offset - 1 + correct_t
                    elif delta > cfd_raw_t[-1]:
                        delta -= 1
                        if cfd_raw_t[0] < delta < cfd_raw_t[-1]:
                            correct_t = np.interp(delta, cfd_raw_t, cfd_true_t)
                            t = offset + 1 + correct_t
                if t is None:
                    t = x - 0.5703
                    amp = amp/2.118 # average correction
                else:
                    correct_amp = np.interp(correct_t, cfd_true_t, amp_raw_t)
                    amp /= correct_amp

            else:
                t = -999
                amp = -999

            return t, amp, baseline

        pq_data = []

        if analysis == 'led':
            # find the samples that are close to the LED firings
            # The sample data are in blocks for each mPMT card (~200 flashes)
            sample_blocks = {}
            for j in range(len(card_id)):
                card = card_id[j]
                if card not in sample_blocks:
                    sample_blocks[card] = []
                sample_blocks[card].append(j)

            led_samples = {}
            start_time = time.time()
            end_time = time.time()
            
            for card in sample_blocks:
                sample_indices = np.array(sample_blocks[card])
                print("Starting Card",card,"last operation took",end_time-start_time)
                start_time = time.time()
                # k = 0
                last_led_card = -1
                for i in range(len(led_card_id)):
                    #if led_card_id[i] != last_led_card: for debugging
                    #    last_led_card = led_card_id[i]
                    #    xxx=1
                    #difference between the waveform coarse counters from this card and the led flast at i 
                    abs_diff = np.abs(coarse[sample_indices] - led_coarse[i])
                    samples_near_led = sample_indices[abs_diff < 400]
                    # samples_near_led = []
                    
                    # while k < len(sample_indices) and abs(coarse[sample_indices[k]] - led_coarse[i]) < 400:
                    #     samples_near_led.append(sample_indices[k])
                    #     k += 1
                    
                    # for index in sample_indices:
                    #     if abs(coarse[index] - led_coarse[i]) < 400:
                    #         samples_near_led.append(index)
                            
                    if len(samples_near_led) > 0:
                        if i not in led_samples:
                            led_samples[i] = []
                        
                        #change this part since samples_near_led is now a numpy array
                        led_samples[i] += samples_near_led.tolist()
                        # led_samples[i] += samples_near_led
                end_time = time.time()
            
            for i in led_samples:
            #for i in range(35):
                if(i%1000==0): 
                    print("Processing flash on mPMT",led_card_id[i],"#",i,"of",len(led_samples)," # mPMT receivers",len(led_samples[i]))
                    print("time",end_time-start_time)
                start_time = time.time()
                if led_card_id[i] > 129:
                    continue
                led = {}
                led['card_id'] = led_card_id[i]
                led['led_no'] = led_no[i]
                led['coarse'] = led_coarse[i]
                mpmts = {}
                for j in led_samples[i]:
                    mpmt_card_id = card_id[j]
                    if mpmt_card_id > 129:
                        continue
                    if mpmt_card_id not in mpmts:
                        mpmts[mpmt_card_id] = []

                    offset = min(50,max(0,np.argmax(samples[j]) - 6)) # where the pulse starts in the sample
                    t, amp, baseline = get_cfd(samples[j][offset:])
                    t+=offset

                    pmt_cfd = {'chan': chan[j],
                            't': t,
                            'amp': amp,
                            'baseline': baseline,
                            'coarse': coarse[j],
                            #'fine_time': fine_time[i],    data not filled
                            #'charge': charge[i],
                            #'coarse_hits':coarse_hits[i]
                            }
                    mpmts[mpmt_card_id].append(pmt_cfd)
                led['mpmts'] = mpmts
                pq_data.append(led)
                end_time = time.time()
            
            unique_card_ids = set(led['card_id'] for led in pq_data)
            print("LEDs flashing in dict are:",unique_card_ids)
            
            total_mPMT_waveforms = 0
            for led in pq_data:
                total_mPMT_waveforms += len(led['mpmts'])  # Count keys in mpmts dictionary

            mean_mPMT_waveforms = total_mPMT_waveforms / len(pq_data) if pq_data else 0  # Avoid division by zero if leds is empty
            print("Mean number of mPMTs across all leds:", mean_mPMT_waveforms)
            
        return pq_data

    def buildLEDDict_ben(self):
        analysis = 'led'
        
        #copied and pasted from Dean's process_parquet_v3.py
        #changes are to make the lists specific to each spill 
        # pql_card_id,pql_coarse,pql_led_no,pql_seq_no = pq.read_table(folder+'/'+prefix+run+'_led.parquet',columns=['card_id','coarse','led_no','seq_no'])
        self.interval_led_flashes = self.interval_led_flashes.sort_values(by='coarse', ascending=True)
        # led_card_id = self.interval_led_flashes['card_id'].tolist()
        # led_coarse = self.interval_led_flashes['coarse'].tolist()
        # led_no = self.interval_led_flashes['led_no'].tolist()
        # led_seq_no = self.interval_led_flashes['seq_no'].tolist()
        
        led_card_id = self.interval_led_flashes['card_id'].to_numpy()
        led_coarse = self.interval_led_flashes['coarse'].to_numpy()
        led_no = self.interval_led_flashes['led_no'].to_numpy()
        led_seq_no = self.interval_led_flashes['seq_no'].to_numpy()

        # Read the waveform data directly from the interval dataframe
        # card_id = self.interval_waveforms_df['card_id'].tolist()
        # chan = self.interval_waveforms_df['chan'].tolist()
        # samples = self.interval_waveforms_df['samples'].tolist()
        # coarse = self.interval_waveforms_df['coarse'].tolist()
        
        card_id = self.interval_waveforms_df['card_id'].to_numpy()
        chan = self.interval_waveforms_df['chan'].to_numpy()
        samples = self.interval_waveforms_df['samples'].to_numpy()
        coarse = self.interval_waveforms_df['coarse'].to_numpy()

        expected_dts = [10000,3125,5000,20000,25000,62500,2500000]

        sample_cards = {}

        if analysis == 'led':
            led_cards = {}
            led_nos = {}

            for dict,data in zip([sample_cards,led_cards,led_nos],[card_id,led_card_id,led_no]):
                for i in range(len(data)):
                    if data[i] not in dict:
                        dict[data[i]] = 1
                    else:
                        dict[data[i]] += 1
        
        if analysis == 'led':
            led_times_by_card = {}
            last_time_by_card = {}
            for i in range(len(led_card_id)):
                card = led_card_id[i]
                if card not in led_times_by_card:
                    led_times_by_card[card] = []
                    last_time_by_card[card] = 0
                dt = led_coarse[i] - last_time_by_card[card]
                if dt not in expected_dts:
                    led_times_by_card[card].append(led_coarse[i])
                last_time_by_card[card] = led_coarse[i]

            delta_led_times_by_card = {}
            for card in led_times_by_card:
                delta_led_times_by_card[card] = [led_times_by_card[card][i+1] - led_times_by_card[card][i] for i in range(len(led_times_by_card[card])-1)]


        sample_times_by_card ={}
        last_time_by_card = {}
        for i in range(len(coarse)):
            card = card_id[i]
            if card not in sample_times_by_card:
                sample_times_by_card[card] = []
                last_time_by_card[card] = 0
            dt = coarse[i] - last_time_by_card[card]
            if analysis != 'coincidences' and dt !=0 and dt not in expected_dts:
                sample_times_by_card[card].append(coarse[i])
            last_time_by_card[card] = coarse[i]

        delta_sample_times_by_card = {}
        for card in sample_times_by_card:
            delta_sample_times_by_card[card] = [sample_times_by_card[card][i+1] - sample_times_by_card[card][i] for i in range(len(sample_times_by_card[card])-1)]

        # Print summary of runs and delta times

        if analysis != 'no_print':
            LED_buff = 'none'
            if analysis == 'led':
                LED_buff = ",".join([str(card) for card in led_cards])
            SAMPLE_buff = ",".join([str(card) for card in sample_cards])
            # print('RUN:',run,' LED cards:',LED_buff,' SAMPLE cards:',SAMPLE_buff)
            print('LED cards:',LED_buff,' SAMPLE cards:',SAMPLE_buff)
            if analysis == 'led':
                print('*** LED summary ***')
                for card in led_cards:
                    print(f'Card {card} has {led_cards[card]} LED flashes')
                    if card in delta_led_times_by_card:
                        if len(delta_led_times_by_card[card])>0:
                            print(f'  Delta times: {delta_led_times_by_card[card]}')
                    else:
                        print(f'  No delta times')
            print('*** Sample summary ***')
            for card in sample_cards:
                print(f'Card {card} has {sample_cards[card]} samples')
                if card in delta_sample_times_by_card:
                    if len(delta_sample_times_by_card[card])>0:
                        print(f'  Delta times: {delta_sample_times_by_card[card]}')
                else:
                    print(f'  No delta times')

        # Analyze the waveform data with CFD

        TIME_MAGIC = 0.5703
        AMP_MAGIC = 2.118 
        DEBUG = False

        cfd_raw_t = np.array([0.16323452713658082,
                        0.20385733509493395,
                        0.24339187740767365,
                        0.2822514122310461,
                        0.3208335490313887,
                        0.35953379168152044,
                        0.3987592183841288,
                        0.4389432980060811,
                        0.4805630068163285,
                        0.5241597383052767,
                        0.5703660640730557,
                        0.6199413381955754,
                        0.6738206794685682,
                        0.7331844507933303,
                        0.7995598000823612,
                        0.874973724581176,
                        0.9621917102137131,
                        1.0301530251726216,
                        1.0769047405430523,
                        1.1210801763323819,
                        1.1632345271365807])

        # correction for the amplitude derived by summing the largest three adcs
        amp_raw_t = np.array([2.0413475167493225, 2.0642014124776784, 2.0847238089021274, 2.1028869067818117, 2.118667914530039,
                                2.1320484585033723, 2.1430140317025583, 2.151553497195665, 2.1576586607668613, 2.1613239251470255,
                                2.162546035746829, 2.1613239251470255, 2.1576586607668617, 2.1515534971956654, 2.143014031702558,
                                2.1320484585033723, 2.118667914530039, 2.1028869067818117, 2.0847238089021274, 2.0642014124776784,
                                2.0413475167493225])

        cfd_true_t =np.linspace(-0.5, 0.5, 21)

        # the following is from process_root
        # note that the waveforms in the parquet file are already baseline subtracted and inverted
        # So reverse the sign of baseline and adcs in the get_cfd function

        # def get_peak_timebins(waveform, threshold, card_id):
        #     # use the most frequent waveform value as the baseline
        #     values, counts = np.unique(waveform, return_counts=True)
        #     baseline = values[np.argmax(counts)]
        #     # baseline - waveform is positive going signal typically around 0 when there is no signal
        #     # threshold is the minimum positive signal above baseline

        #     below = (waveform[0] - baseline) <= threshold
        #     peak_timebins = []
        #     max_val = 0
        #     max_timebin = -1
        #     for i in range(len(waveform)):
        #         if below and (waveform[i] - baseline) > threshold:
        #             below = False
        #             max_val = 0
        #             max_timebin = -1
        #         if not below:
        #             if (waveform[i] - baseline) > max_val:
        #                 max_val = waveform[i] - baseline
        #                 max_timebin = i
        #             if (waveform[i] - baseline) <= threshold:
        #                 below = True
        #                 #if brb != 3 or len(peak_timebins) == 0 or max_timebin > peak_timebins[-1] + 4:  # eliminate peaks from ringing... (for brb03)
        #                 if 1 == 1:
        #                     peak_timebins.append(max_timebin)
        #     return peak_timebins

        def do_cfd(waveforms):
            #print("Running CFD")
            assert len(waveforms[0])==32, "Double-check the format of the waveforms file"
            
            nbins = 4
            nsample = len(waveforms[0])
            delay = 2
            multiplier = -2
            apply_correction = True

            baseline = np.mean(waveforms[:,:nbins], axis=1)

            # baseline is number of waves 
            baseline = np.tile(baseline,nsample).reshape(nsample, len(baseline)).T

            # SLOW
            sort_vals = np.sort(waveforms, axis=1)


            amplitudes = sort_vals[:, -1] + sort_vals[:, -2] + sort_vals[:, -3]
            amplitudes = amplitudes.astype(float)

            # converted to positive-going pulses
            trimmed =(waveforms[:, delay:] -baseline[:, delay:] ) + multiplier*(waveforms[:, :-delay] - baseline[:, delay:])
            trimmed = trimmed.astype(float)

            # now, find the index with the largest step that goes from + to -
            offset = 5
            crossings = []
            # SLOW 
            for i in range(len(trimmed)):
                indx = np.argwhere(np.diff(np.sign(trimmed[i][offset:])))
                if len(indx)==0:
                    crossings.append(1)
                    #crossings.append(np.argwhere(np.diff(np.sign(trimmed[i])))[0] + 1 )
                else:
                    crossings.append(int(indx[0] + offset + 1))
            
            crossings = np.array(crossings).flatten()
            

            times = np.zeros(len(crossings))

            # okay now apply the CFD 
            xmin = crossings-1
            xmax = crossings

            ymin = trimmed[np.arange(len(trimmed)), crossings-1]
            ymax =  trimmed[np.arange(len(trimmed)), crossings]

            slope = (xmax-xmin) / (ymax-ymin) 

            good_mask = slope>0
            x_interp = xmin -  slope * ymin 
            
            delta = x_interp - (offset +1)

            in_bounds = np.logical_and(cfd_raw_t[0] < delta, delta < cfd_raw_t[-1])

            shift_up = delta +1 
            shift_down = delta-1 

            shift_up_good = np.logical_and( np.logical_not(in_bounds), np.logical_and(cfd_raw_t[0] < shift_up, shift_up < cfd_raw_t[-1]))
            shift_down_good = np.logical_and( np.logical_not(in_bounds), np.logical_and(cfd_raw_t[0] < shift_down, shift_down < cfd_raw_t[-1]))
            all_bad = np.logical_not(np.logical_or(in_bounds, np.logical_or(shift_up_good, shift_down_good)))

            
            #apply_correction
            if apply_correction:


                times[in_bounds] = offset + np.interp(delta[in_bounds], cfd_raw_t, cfd_true_t)
                times[shift_up_good] = offset + 1 + np.interp(shift_up[shift_up_good], cfd_raw_t, cfd_true_t)
                times[shift_down_good] = offset -1 + np.interp(shift_down[shift_down_good], cfd_raw_t, cfd_true_t)
            else:
                times = offset  + delta

            times[all_bad] = x_interp[all_bad] - TIME_MAGIC
            amplitudes[all_bad] = amplitudes[all_bad]/AMP_MAGIC

            amplitudes[np.logical_not(all_bad)] /= np.interp(times[np.logical_not(all_bad)], cfd_true_t, amp_raw_t )

            times -= offset

            good_mask = np.logical_and(good_mask, np.abs(times)<0.6)
            #good_mask = np.logical_and(good_mask, np.logical_not(in_bounds))
            #good_mask = np.logical_and(good_mask, np.logical_not(bad_sign))

            if DEBUG:
                i = 0
                plotted = 0
                while True:
                    if True: 
                        good_mask[i] = False
                        #print(x_interp[i], delta[i])
                        plt.clf()
                        
                        #trim_cros =np.argwhere(np.diff(np.sign(trimmed[i])))
                        #plt.plot(crossings[i]+delay, trimmed[i][crossings[i]], 'rd', label="Crossings")

                        plt.plot(range(len(waveforms[i])), waveforms[i], label="Waveform")
                        plt.plot(np.array(range(len(trimmed[i]))) + delay, trimmed[i], label="Processed")

                        plt.plot([xmin[i]+delay, xmax[i]+delay], [ymin[i], ymax[i]], label="Crossing", color="green")
                        plt.grid(which='major', alpha=0.5)
                        plt.xlabel("Coarse Counter", size=14)
                        plt.ylabel("ADC", size=14)
                        plt.legend()
                        plt.tight_layout()
                        plt.show()
                        plotted+=1
                    i+=1 
                    
                    if plotted>10:
                        break


                return times[good_mask], amplitudes[good_mask], baseline[good_mask], good_mask
                
        pq_data = []

        if analysis == 'led':
            # find the samples that are close to the LED firings
            # The sample data are in blocks for each mPMT card (~200 flashes)
            sample_blocks = {}
            for j in range(len(card_id)):
                card = card_id[j]
                if card not in sample_blocks:
                    sample_blocks[card] = []
                sample_blocks[card].append(j)

            led_samples = {}
            start_time = time.time()
            end_time = time.time()
            
            for card in sample_blocks:
                sample_indices = np.array(sample_blocks[card])
                print("Starting Card",card,"last operation took",end_time-start_time)
                start_time = time.time()
                # k = 0
                last_led_card = -1
                for i in range(len(led_card_id)):
                    #if led_card_id[i] != last_led_card: for debugging
                    #    last_led_card = led_card_id[i]
                    #    xxx=1
                    #difference between the waveform coarse counters from this card and the led flast at i 
                    abs_diff = np.abs(coarse[sample_indices] - led_coarse[i])
                    samples_near_led = sample_indices[abs_diff < 400]
                    # samples_near_led = []
                    
                    # while k < len(sample_indices) and abs(coarse[sample_indices[k]] - led_coarse[i]) < 400:
                    #     samples_near_led.append(sample_indices[k])
                    #     k += 1
                    
                    # for index in sample_indices:
                    #     if abs(coarse[index] - led_coarse[i]) < 400:
                    #         samples_near_led.append(index)
                            
                    if len(samples_near_led) > 0:
                        if i not in led_samples:
                            led_samples[i] = []
                        
                        #change this part since samples_near_led is now a numpy array
                        led_samples[i] += samples_near_led.tolist()
                        # led_samples[i] += samples_near_led
                end_time = time.time()
            total_cdf_time =0
            for i in led_samples:
            #for i in range(35):
                
                if(i%100==0): 
                    print("Processing flash on mPMT",led_card_id[i],"#",i,"of",len(led_samples)," # mPMT receivers",len(led_samples[i]))
                    print("time",end_time-start_time, "CFD time",total_cdf_time)
                total_cdf_time =0
                start_time = time.time()
                if led_card_id[i] > 129:
                    continue
                led = {}
                led['card_id'] = led_card_id[i]
                led['led_no'] = led_no[i]
                led['coarse'] = led_coarse[i]
                mpmts = {}
                for j in led_samples[i]:
                    mpmt_card_id = card_id[j]
                    if mpmt_card_id > 129:
                        continue
                    if mpmt_card_id not in mpmts:
                        mpmts[mpmt_card_id] = []

                    offset = min(50,max(0,np.argmax(samples[j]) - 6)) # where the pulse starts in the sample
                    start_time_cfd = time.time()
                    t, amp, baseline = get_cfd_ben(samples[j][offset:])
                    end_time_cfd = time.time()
                    total_cdf_time += end_time_cfd-start_time_cfd
                    t+=offset

                    pmt_cfd = {'chan': chan[j],
                            't': t,
                            'amp': amp,
                            'baseline': baseline,
                            'coarse': coarse[j],
                            #'fine_time': fine_time[i],    data not filled
                            #'charge': charge[i],
                            #'coarse_hits':coarse_hits[i]
                            }
                    mpmts[mpmt_card_id].append(pmt_cfd)
                led['mpmts'] = mpmts
                pq_data.append(led)
                end_time = time.time()
            
            unique_card_ids = set(led['card_id'] for led in pq_data)
            print("LEDs flashing in dict are:",unique_card_ids)
            
            total_mPMT_waveforms = 0
            for led in pq_data:
                total_mPMT_waveforms += len(led['mpmts'])  # Count keys in mpmts dictionary

            mean_mPMT_waveforms = total_mPMT_waveforms / len(pq_data) if pq_data else 0  # Avoid division by zero if leds is empty
            print("Mean number of mPMTs across all leds:", mean_mPMT_waveforms)
            
        return pq_data