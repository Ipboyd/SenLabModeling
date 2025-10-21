import numpy as np
from tqdm import tqdm

def genSpatiallyTunedChans(azi, chans=19, sigma=300):
    
    spatialCurves = np.zeros((chans,chans))
    for idx in range(chans): 
        spatialCurves[idx,:]= np.exp(((-1/2)*(azi-azi[idx])**2)*(1/sigma)) 
        
    return spatialCurves


def make_grid_target_masker_locs(azi):
    grid_x, grid_y = np.meshgrid(np.arange(len(azi)), np.arange(len(azi))) # masker, target
    masker_locs = grid_x.flatten() # masker
    target_locs = grid_y.flatten() # target
    return masker_locs, target_locs  

def gen_IC_spks(spatialCurves, azi, tmax, locs, fr_targets, fr_masker, newStrfGain, strfGain, trials=1, padToTime = 3500, dt=0.1):
    
    m_loc, t_loc = locs
    singleConfigSpks = np.zeros((trials,spatialCurves.shape[0],tmax))  
    
    for t in range(trials):
        for ch in range(spatialCurves.shape[0]):
            t_wt = spatialCurves[ch,azi==azi[t_loc]] if t_loc is not None else 0.
            m_wt = spatialCurves[ch,azi==azi[m_loc]] if m_loc is not None else 0.

            if t_wt + m_wt == 0: raise Exception(f'No contribution from target or masker at channel {ch}, trial {t+1}. Both cannot be None.')
                
            singleConfigSpks[t,ch,:] = t_wt*fr_targets.squeeze() + m_wt*fr_masker[t].squeeze()      

            if t_wt + m_wt >= 1: singleConfigSpks[t,ch,:] = singleConfigSpks[t,ch,:] / (t_wt + m_wt)
    
    if singleConfigSpks.shape[2] < padToTime/dt:
        padSize = int(padToTime/dt)-singleConfigSpks.shape[2]
        singleConfigSpks = np.concatenate([singleConfigSpks, np.zeros((trials,spatialCurves.shape[0],padSize))], axis=2)
    
    spks =  singleConfigSpks.transpose(2,1,0) * newStrfGain / strfGain  # Shape to (time, chans, trials)

    return spks


if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    import scipy.io
    
    chans = 19
    azi = np.flip(np.linspace(-90, 90, chans))
    spatialCurves = genSpatiallyTunedChans(azi, chans=chans, sigma=300)
    
    # plt.plot(azi, spatialCurves, label=[f'Chan {i}' for i in range(spatialCurves.shape[0])])
    # plt.xlabel('Azimuth (degrees)')
    # plt.show()    
    
    
    path = r"D:\School_Stuff\Rotation_1_Sep_Nov_Kamal_Sen\Code\MouseSpatialGrid-19-Chan\ICSimStim\default_STRF_with_offset_200k.mat"
    data = scipy.io.loadmat(path)
    
    fr_target_on = np.array([np.array(dta) for dta in data['fr_target_on'].squeeze()])
    fr_target_off = np.array([np.array(dta) for dta in data['fr_target_off'].squeeze()])
    fr_masker = np.array([np.array(dta) for dta in data['fr_masker'].squeeze()])
    strfGain = float(data['strfGain'].squeeze())
    tmax = fr_target_on.shape[1]
    newStrfGain = strfGain
    subz = np.linspace(5,int(chans*5),chans, dtype=int)
    
    
    masker_locs, target_locs = make_grid_target_masker_locs(azi)
    
    progress_bar = tqdm(zip(masker_locs, target_locs))
    for m_loc, t_loc in progress_bar:
        
        progress_bar.set_description(f'Generating spikes for Masker Loc: {m_loc}, Target Loc: {t_loc}')
        on_spks_stimulus_1 = gen_IC_spks(spatialCurves=spatialCurves, 
                            azi=azi, 
                            tmax=tmax, 
                            locs=(m_loc, t_loc), 
                            fr_targets=fr_target_on[0], 
                            fr_masker=fr_masker, 
                            newStrfGain=newStrfGain, 
                            strfGain=strfGain, 
                            trials=10, 
                            padToTime = 3500, 
                            dt=0.1)