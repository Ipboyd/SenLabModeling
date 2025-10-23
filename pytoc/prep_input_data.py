import filecmp
import numpy as np
import scipy.io
from tqdm import tqdm
from argparse import ArgumentParser

class PrepInput(object):
    '''
    Class to prepare input data for IC neuron simulations based on STRF data.
    Attributes:
        chans : int
            Number of channels.
        azi : np.ndarray
            Azimuth locations for channels. 
        sigma : float
            Standard deviation for Gaussian tuning curves.
        trials : int
            Number of trials.   
        padToTime : int
            Time to pad spike trains to (ms).
        dt : float      
            Time step in ms.
        
    '''
    def __init__(self, args):
        
        self.chans = args.chans
        self.sigma = args.sigma
        self.trials = args.trials
        self.padToTime = args.padToTime
        self.dt = args.dt
        
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
                    spks_dict[f'locs_masker_{locs[0]}_target_{locs[1]}_on'][f'stimulus_{stimulus}'] = on_spks
                
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
                    spks_dict[f'locs_masker_{locs[0]}_target_{locs[1]}_off'][f'stimulus_{stimulus}'] = off_spks
                
        return spks_dict


if __name__ == "__main__":
    
    path = r"D:\School_Stuff\Rotation_1_Sep_Nov_Kamal_Sen\Code\MouseSpatialGrid-19-Chan\ICSimStim\default_STRF_with_offset_200k.mat"
    
    args = ArgumentParser()
    args.add_argument('--chans', type=int, default=4, help='Number of channels')
    args.add_argument('--trials', type=int, default=10, help='Number of trials')
    args.add_argument('--padToTime', type=int, default=3500, help='Time to pad spike trains to (ms)')
    args.add_argument('--sigma', type=int, default=300, help='Standard deviation for Gaussian tuning curves')    
    args.add_argument('--dt', type=float, default=0.1, help='Time step in ms')
    parsed_args = args.parse_args()
    
    
    prep_input = PrepInput(parsed_args)
    masker_locs, target_locs = prep_input.make_grid_target_masker_locs()
    list_locs = list(zip(masker_locs, target_locs))
    
    spks = prep_input.process_input(
                strf_path=path, 
                list_locs=list_locs, 
                on_neuron=True, 
                off_neuron=True,)
    
    print("Generated spike train keys:", spks.keys())
    
