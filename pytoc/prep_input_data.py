import yaml
import numpy as np
import scipy.io
from tqdm import tqdm
from argparse import ArgumentParser



class PrepInput(object):
    '''
    Class to prepare input data for IC neuron simulations and generating Poisson spikes based on STRF data.
    Parameters:
        args : argparse.Namespace
            Command-line arguments containing parameters like number of channels, trials, etc.
        config : dict
            Configuration dictionary loaded from a YAML file containing parameters for spike generation.
    '''
    def __init__(self, args, config):
        
        self.chans = args.chans
        self.sigma = args.sigma
        self.trials = args.trials
        self.padToTime = args.padToTime
        self.dt = args.dt
        
        
        self.dt = config['dt']
        self.t_ref = config['t_ref']
        self.t_ref_rel = config['t_ref_rel']
        self.rec = config['rec']
        self.refrac = config['refrac']
        self.scale_factor = config['scale_factor']
        self.offset_val = config['offset_val']
        
        self.azi = np.flip(np.linspace(-90,90, self.chans))
        
        self.spatialCurves = self.genSpatiallyTunedChans()

    def genSpatiallyTunedChans(self):
        '''
        Generate spatial tuning curves for each channel based on Gaussian profiles.
        '''
        spatialCurves = np.zeros((self.chans, self.chans))
        for idx in range(self.chans): 
            spatialCurves[idx,:]= np.exp(((-1/2)*(self.azi-self.azi[idx])**2)*(1/self.sigma)) 
            
        return spatialCurves

    def make_grid_target_masker_locs(self):
        '''
        Create grid (coordinates) of masker and target locations based on number of channels.
        '''
        grid_x, grid_y = np.meshgrid(np.arange(len(self.azi)), np.arange(len(self.azi))) # masker, target
        masker_locs = grid_x.flatten() # masker
        target_locs = grid_y.flatten() # target
        return masker_locs, target_locs  

    def gen_IC_spks(self, tmax, locs, fr_targets, fr_masker, newStrfGain, strfGain):
        '''
        Generate adjusted IC based on STRF data for given masker and target locations.
        Parameters:
            tmax : int
                Maximum time for spike train (number of time steps).
            locs : tuple
                (masker_location, target_location)
            fr_targets : np.ndarray
                Firing rates for target stimulus.
            fr_masker : np.ndarray
                Firing rates for masker stimulus.
            newStrfGain : float
                New STRF gain to adjust firing rates.
            strfGain : float    
                Original STRF gain.
        Returns:    
            spks : np.ndarray 
                Generated spike train matrix (time x channels x trials).
        '''

        m_loc, t_loc = locs # masker, target
        singleConfigSpks = np.zeros((self.trials,self.spatialCurves.shape[0], tmax))  
        
        for t in range(self.trials):
            for ch in range(self.spatialCurves.shape[0]):
                t_wt = self.spatialCurves[ch,self.azi==self.azi[t_loc]] if t_loc is not None else 0.
                m_wt = self.spatialCurves[ch,self.azi==self.azi[m_loc]] if m_loc is not None else 0.

                if t_wt + m_wt == 0: raise Exception(f'No contribution from target or masker at channel {ch}, trial {t+1}. Both cannot be None.')
                    
                singleConfigSpks[t,ch,:] = t_wt*fr_targets.squeeze() + m_wt*fr_masker[t].squeeze()      

                if t_wt + m_wt >= 1: singleConfigSpks[t,ch,:] = singleConfigSpks[t,ch,:] / (t_wt + m_wt)
        
        if singleConfigSpks.shape[2] < self.padToTime/self.dt:
            padSize = int(self.padToTime/self.dt)-singleConfigSpks.shape[2]
            singleConfigSpks = np.concatenate([singleConfigSpks, np.zeros((self.trials,self.spatialCurves.shape[0],padSize))], axis=2)
        
        spks =  singleConfigSpks.transpose(2,1,0) * newStrfGain / strfGain  # Shape to (time, chans, trials)

        return spks


    def process_input(self, strf_path, list_locs, on_neuron=True, off_neuron=True):
        '''
        Process input STRF data to generate spike trains for specified masker and target locations.
        Parameters:
            strf_path : str
                Path to the .mat file containing STRF data.
            list_locs : list of tuples
                List of (masker_location, target_location) pairs.
            on_neuron : bool
                Whether to generate spikes for ON neurons.
            off_neuron : bool
                Whether to generate spikes for OFF neurons.
            
            Returns:    
                spks_dict : dict
                    Dictionary containing generated spike trains (adjusted IC and poisson spikes ) for each location and stimulus type.
                
                    '''
        data = scipy.io.loadmat(strf_path)
        fr_target_on = np.array([np.array(dta) for dta in data['fr_target_on'].squeeze()])
        fr_target_off = np.array([np.array(dta) for dta in data['fr_target_off'].squeeze()])
        fr_masker = np.array([np.array(dta) for dta in data['fr_masker'].squeeze()])
        strfGain = float(data['strfGain'].squeeze())
        tmax = fr_target_on.shape[1]
        newStrfGain = strfGain
        
        progress_bar = tqdm(list_locs)

        spks_dict = {}
        for locs in progress_bar:
            progress_bar.set_description(f'Generating spikes for Masker Loc: {locs[0]}, Target Loc: {locs[1]}')
            if on_neuron:
                spks_dict[f'locs_masker_{locs[0]}_target_{locs[1]}_on'] = {}
                for stimulus in range(fr_target_on.shape[0]):
                    on_spks = self.gen_IC_spks(
                                        tmax=tmax, 
                                        locs=locs, 
                                        fr_targets=fr_target_on[stimulus], 
                                        fr_masker=fr_masker, 
                                        newStrfGain=newStrfGain, 
                                        strfGain=strfGain)
                    
                    on_poisson_spks = self.gen_poisson_inputs(on_spks)
                    
                    spks_dict[f'locs_masker_{locs[0]}_target_{locs[1]}_on'][f'stimulus_{stimulus}_IC_spks'] = on_spks
                    spks_dict[f'locs_masker_{locs[0]}_target_{locs[1]}_on'][f'stimulus_{stimulus}_poisson_spks'] = on_poisson_spks
                
            if off_neuron:
                spks_dict[f'locs_masker_{locs[0]}_target_{locs[1]}_off'] = {}
                for stimulus in range(fr_target_off.shape[0]):
                    off_spks = self.gen_IC_spks(
                                        tmax=tmax, 
                                        locs=locs, 
                                        fr_targets=fr_target_off[stimulus], 
                                        fr_masker=fr_masker, 
                                        newStrfGain=newStrfGain, 
                                        strfGain=strfGain)
                    off_poisson_spks = self.gen_poisson_inputs(off_spks)
                    spks_dict[f'locs_masker_{locs[0]}_target_{locs[1]}_off'][f'stimulus_{stimulus}_IC_spks'] = off_spks
                    spks_dict[f'locs_masker_{locs[0]}_target_{locs[1]}_off'][f'stimulus_{stimulus}_poisson_spks'] = off_poisson_spks
                    
        return spks_dict
    

    def spike_generator(self, rate):
        """
        Generate a Poisson spike train with an absolute and relative refractory period.
        parameters:
            rate : np.ndarray
        """
        dt_sec = self.dt / 1000  # ms to seconds

        n = len(rate)
        spike_train = np.zeros(n)
        spike_times = []

        #9/17 refractory preiod seems to low compared to real data. Perhaps extend this?
        # n_refab = int(0 / 1000 / dt_sec)  # number of samples for ref. period window
        n_refab = int(15 / 1000 / dt_sec)  # number of samples for ref. period window
        #n_refab = int(30 / 1000 / dt_sec)  # number of samples for ref. period window
        tw = np.arange(n_refab + 1)

        t_ref_samp = int(self.t_ref / 1000 / dt_sec)
        t_rel_samp = int(self.t_ref_rel / 1000 / dt_sec)


        # Recovery function based on Schaette et al. 2005
        with np.errstate(divide='ignore', invalid='ignore'):
            w = np.power(tw - t_ref_samp, self.rec) / (
                np.power(tw - t_ref_samp, self.rec) + np.power(t_rel_samp, self.rec)
            )
            w[tw < t_ref_samp] = 0
            w = np.nan_to_num(w)

        x = np.random.rand(n)
        

        for i in range(n):
            if spike_times and i - spike_times[-1] < n_refab:

                rate[i] *= w[i - spike_times[-1]]
            if x[i] < dt_sec * rate[i]:
                spike_train[i] = 1
                spike_times.append(i)

        return spike_train
    
    
    def gen_poisson_times(self, chans, FR, std, simlen, trials):
        """
        Generate Poisson spike trains with refractory period.

        Parameters:
            chans : int
                Number of neurons in population.
            FR : float
                Firing rate in Hz.
            std : float
                Standard deviation of the firing rate.
            simlen : int
                Number of time steps (default = 35000)

        Returns:
            token : np.array
                Binary spike train matrix of shape (simlen, chans)
        """
        chans = int(chans)
        simlen = int(simlen)
        std = int(std) #set to 0.0 right now
        #FR = int(FR) # set to  8.0 right now

        # Generate Poisson spikes with added noise

        #Convert things from tensors so we can work with them
        rand_gauss = FR + std * np.random.randn(simlen, chans, trials)
        rand_bin = np.random.rand(simlen, chans, trials) < (rand_gauss * self.dt / 1000)

        temp = rand_bin.astype(np.uint8)

        for chan in range(chans):
            for trial in range(trials):
                spk_inds = np.where(temp[:, chan, trial])[0]
                if len(spk_inds) > 1:
                    ISIs = np.diff(spk_inds) * self.dt
                    violate_inds = np.where(ISIs < self.refrac)[0] + 1
                    temp[spk_inds[violate_inds], chan, trial] = 0

        
        return np.array(temp)


    def gen_poisson_inputs(self, spks):
        """
        Generate Poisson spike inputs from spike rates.

        Parameters:
            spks : np.ndarray
                Spike rate matrix (time x neurons x trials)

        Returns:
            s : np.ndarray
                Binary spike train matrix (time x neurons)
        """
        
        trial_rate = spks  # shape: (time, neurons, trials)

        rate = trial_rate[int(self.offset_val):int(self.offset_val+len(trial_rate)*self.scale_factor)] if self.scale_factor != 1 else trial_rate

        s = np.zeros_like(rate)
        for chan in range(rate.shape[1]):
            for trial_num in range(rate.shape[2]):
                s[:, chan, trial_num] = self.spike_generator(rate[:, chan, trial_num])
            
        return s
    
    
    def process_input_from_raw_stim(self, fr_target_on, fr_target_off, fr_masker, strfGain, list_locs, on_neuron=True, off_neuron=True):
        '''
        Process input STRF data to generate spike trains for specified masker and target locations.
        Parameters:
            fr
            strf_path : str
                Path to the .mat file containing STRF data.
            list_locs : list of tuples
                List of (masker_location, target_location) pairs.
            on_neuron : bool
                Whether to generate spikes for ON neurons.
            off_neuron : bool
                Whether to generate spikes for OFF neurons.
            
            Returns:    
                spks_dict : dict
                    Dictionary containing generated spike trains (adjusted IC and poisson spikes ) for each location and stimulus type.
                
                    '''
        tmax = fr_target_on.shape[1]
        newStrfGain = strfGain
        
        progress_bar = tqdm(list_locs)
        spks_dict = {}
        for locs in progress_bar:
            progress_bar.set_description(f'Generating spikes for Masker Loc: {locs[0]}, Target Loc: {locs[1]}')
            if on_neuron:
                spks_dict[f'locs_masker_{locs[0]}_target_{locs[1]}_on'] = {}
                for stimulus in range(fr_target_on.shape[0]):
                    on_spks = self.gen_IC_spks(
                                        tmax=tmax, 
                                        locs=locs, 
                                        fr_targets=fr_target_on[stimulus], 
                                        fr_masker=fr_masker, 
                                        newStrfGain=newStrfGain, 
                                        strfGain=strfGain)
                    
                    on_poisson_spks = self.gen_poisson_inputs(on_spks)
                    
                    spks_dict[f'locs_masker_{locs[0]}_target_{locs[1]}_on'][f'stimulus_{stimulus}_IC_spks'] = on_spks
                    spks_dict[f'locs_masker_{locs[0]}_target_{locs[1]}_on'][f'stimulus_{stimulus}_poisson_spks'] = on_poisson_spks
                
            if off_neuron:
                spks_dict[f'locs_masker_{locs[0]}_target_{locs[1]}_off'] = {}
                for stimulus in range(fr_target_off.shape[0]):
                    off_spks = self.gen_IC_spks(
                                        tmax=tmax, 
                                        locs=locs, 
                                        fr_targets=fr_target_off[stimulus], 
                                        fr_masker=fr_masker, 
                                        newStrfGain=newStrfGain, 
                                        strfGain=strfGain)
                    off_poisson_spks = self.gen_poisson_inputs(off_spks)
                    spks_dict[f'locs_masker_{locs[0]}_target_{locs[1]}_off'][f'stimulus_{stimulus}_IC_spks'] = off_spks
                    spks_dict[f'locs_masker_{locs[0]}_target_{locs[1]}_off'][f'stimulus_{stimulus}_poisson_spks'] = off_poisson_spks
                    
        return spks_dict
    
if __name__ == "__main__":
    '''
    Example usage of the PrepInput class to generate spike trains based on STRF data.
    1. Load configuration from YAML file.
    2. Initialize PrepInput with command-line arguments and configuration.
    3. Generate masker and target locations.
    4. Process input STRF data to generate spike trains.
    5. Print the keys of the generated spike train dictionary to visualize.
    6. Adjust parameters as needed for different scenarios.
    '''
    
    path = r"D:\School_Stuff\Rotation_1_Sep_Nov_Kamal_Sen\Code\MouseSpatialGrid-19-Chan\ICSimStim\default_STRF_with_offset_200k.mat"
    
    args = ArgumentParser()
    args.add_argument('--chans', type=int, default=4, help='Number of channels')
    args.add_argument('--trials', type=int, default=10, help='Number of trials')
    args.add_argument('--padToTime', type=int, default=3500, help='Time to pad spike trains to (ms)')
    args.add_argument('--sigma', type=int, default=300, help='Standard deviation for Gaussian tuning curves')    
    args.add_argument('--dt', type=float, default=0.1, help='Time step in ms')
    args.add_argument('--FR', type=float, default=8.0, help='Firing rate of neurons in Hz')
    args.add_argument('--std', type=float, default=0.0, help='Standard deviation of the firing rate.')
    args.add_argument('--simlen', type=int, default=35000, help='Number of time steps (default = 35000).')
    args.add_argument('--target_dir', type=str, default='resampled-stimuli/target', help='directory containing target stimuli')
    args.add_argument('--masker_dir', type=str, default='resampled-stimuli/masker', help='directory containing masker stimuli')
        
    parsed_args = args.parse_args()
    

    # the yaml cofig file is constant and does not need to be changed for different runs
    yaml_path = 'config/config.yaml'
    config = yaml.safe_load(open(yaml_path, 'r'))
    sub_config = config['input_spike_train']
    
    prep_input = PrepInput(parsed_args, sub_config)
    
    masker_locs, target_locs = prep_input.make_grid_target_masker_locs()
    list_locs = list(zip(masker_locs, target_locs))
    
    spks = prep_input.process_input(
                strf_path=path, 
                list_locs=list_locs, 
                on_neuron=True, 
                off_neuron=True,)
    
    print("Generated spike train keys:", spks.keys())
    
    
    