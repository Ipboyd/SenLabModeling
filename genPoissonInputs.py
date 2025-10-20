import numpy as np
import scipy.io

def spike_generator(rate, dt, t_ref, t_ref_rel, rec):
    """
    Generate a Poisson spike train with an absolute and relative refractory period.
    """
    dt_sec = dt / 1000  # ms to seconds

    n = len(rate)
    spike_train = np.zeros(n)
    spike_times = []

    #9/17 refractory preiod seems to low compared to real data. Perhaps extend this?
    n_refab = int(0 / 1000 / dt_sec)  # number of samples for ref. period window
    #n_refab = int(15 / 1000 / dt_sec)  # number of samples for ref. period window
    #n_refab = int(30 / 1000 / dt_sec)  # number of samples for ref. period window
    tw = np.arange(n_refab + 1)

    t_ref_samp = int(t_ref / 1000 / dt_sec)
    t_rel_samp = int(t_ref_rel / 1000 / dt_sec)


    # Recovery function based on Schaette et al. 2005
    with np.errstate(divide='ignore', invalid='ignore'):
        w = np.power(tw - t_ref_samp, rec) / (
            np.power(tw - t_ref_samp, rec) + np.power(t_rel_samp, rec)
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

def gen_poisson_inputs(t_ref, t_ref_rel, rec, scale_factor, spk_file_name, dt = 0.1, offset_val = 3153):
    """
    Generate Poisson spike inputs from a .mat file of spike rates.

    Parameters:
        t_ref, t_ref_rel : float
            Absolute and relative refractory periods (ms).
        rec : float
            Sharpness of relative refractory function.
        scale_factor : float
        dt : float
            Time step (ms).

    Returns:
        s : np.ndarray
            Binary spike train matrix (time x neurons)
    """
    
    data = scipy.io.loadmat(spk_file_name)
    trial_rate = np.array(data['spks'])  # shape: (time * locations, neurons, trials)

    rate = trial_rate[int(offset_val):int(offset_val+len(trial_rate)*scale_factor)] if scale_factor != 1 else trial_rate


    s = np.zeros_like(rate)
    for chan in range(rate.shape[1]):
        for trial_num in range(rate.shape[2]):
            s[:, chan, trial_num] = spike_generator(rate[:, chan, trial_num], dt, t_ref, t_ref_rel, rec)
        
    return s


if __name__ == "__main__":
    dt = 0.1  # time step in ms
    offset_val = 3153  # offset value to start
    t_ref = 2.0  
    t_ref_rel = 10.0 
    rec = 2.0
    scale_factor = 1.0
    spk_file_name = r"D:\School_Stuff\Rotation_1_Sep_Nov_Kamal_Sen\Code\MouseSpatialGrid-19-Chan\run\4-channel-PV-inputs\solve\IC_spks_on.mat"  # Replace with your .mat file path
    
    spikes = gen_poisson_inputs(t_ref, t_ref_rel, rec, scale_factor, spk_file_name,dt = dt, offset_val = offset_val)
    print("Generated spike train shape:", spikes.shape)
    print("Unique: ", np.unique(spikes, return_counts=True))