import numpy as np
import scipy.io
import matplotlib.pyplot as plt

def spike_generator(rate, dt, t_ref, t_ref_rel, rec):
    """
    Generate a Poisson spike train with an absolute and relative refractory period.
    """
    dt_sec = dt / 1000  # ms to seconds

    #print(dt_sec)

    n = len(rate)
    spike_train = np.zeros(n)
    spike_times = []

    #9/17 refractory preiod seems to low compared to real data. Perhaps extend this?
    #n_refab = int(0 / 1000 / dt_sec)  # number of samples for ref. period window
    n_refab = int(15 / 1000 / dt_sec)  # number of samples for ref. period window
    #n_refab = int(30 / 1000 / dt_sec)  # number of samples for ref. period window
    tw = np.arange(n_refab + 1)

    t_ref_samp = int(t_ref / 1000 / dt_sec)
    t_rel_samp = int(t_ref_rel / 1000 / dt_sec)

    
    #print(t_ref_samp)
    #print('here2')

    # Recovery function based on Schaette et al. 2005
    with np.errstate(divide='ignore', invalid='ignore'):
        w = np.power(tw - t_ref_samp, rec) / (
            np.power(tw - t_ref_samp, rec) + np.power(t_rel_samp, rec)
        )
        w[tw < t_ref_samp] = 0
        w = np.nan_to_num(w)

    x = np.random.rand(n)

    #print('here3')

    

    for i in range(n):
        if spike_times and i - spike_times[-1] < n_refab:

            rate[i] *= w[i - spike_times[-1]]
            #print('here4')
        if x[i] < dt_sec * rate[i]:
            spike_train[i] = 1
            spike_times.append(i)
            #print('here5')


    #print(spike_times)

    return spike_train

def gen_poisson_inputs(trial, loc_num, label, t_ref, t_ref_rel, rec, scale_factor, layer_name = "1-channel-paper"):
    """
    Generate Poisson spike inputs from a .mat file of spike rates.

    Parameters:
        trial : int
            Trial index (1-based in MATLAB, 0-based in Python).
        loc_num : int or None
            Location number, or None to use all.
        label : str
            Label for the stimulus (e.g. 'on' or 'off').
        t_ref, t_ref_rel : float
            Absolute and relative refractory periods (ms).
        rec : float
            Sharpness of relative refractory function.


    Returns:
        s : np.ndarray
            Binary spike train matrix (time x neurons)
    """
    dt = 0.1  # ms
    label_clean = label.strip("'")  # remove literal apostrophes

    #This needs to be made more dynamic and put in a spot where it will actually be updated (perhaps contingetnt on the model being run)
    matfile_path = f"C:/Users/ipboy/Documents/GitHub/ModelingEffort/Single-Channel/Model/Model-Core/Model-Main/run/{layer_name}/solve"
    filename = f"{matfile_path}/IC_spks_{label_clean}.mat"
    
    data = scipy.io.loadmat(filename)
    temp = np.array(data['spks'])  # shape: (time * locations, neurons, trials)

    loc_size = temp.shape[0] // 24

    #Issue with this line below 5/20

    

    #print(int(trial))

    trial_rate = temp[:, :, int(trial)]  # select trial

    trial_rate = np.reshape(trial_rate, (-1, trial_rate.shape[1]))
    #print(len(trial_rate))

    #print(np.shape(np.array(trial_rate)))
    
    #trial_rate = np.atleast_2d(temp[:, :, trial])

    loc_num = int(loc_num)

    #print(loc_num)
    #Okay this will definitely need to be updated for the multichannel model
    #It looks like the indexing was wrong here for the single channel model
    #and it was causing errors.
    #For the single channel model the rate should just be the fr

    #if loc_num is not None:
        #print('here')
    #    rate = trial_rate[loc_size * (loc_num - 1):loc_size * loc_num, :]
    #else:

    #Scale down length according to scale factor

    if scale_factor != 1:
        #offset_val = np.argmax(trial_rate != 0) #If we are using a scale factor that is small, we do not want the signal to be all zeros if we are taking form the start of the signal. The signal starts with about 3000-5000 samples of silence.
        offset_val = 3153

        rate = trial_rate[int(offset_val):int(offset_val+len(trial_rate)*scale_factor)]

    else:
        rate = trial_rate



    #plt.plot(rate)
    #print(rate)
    #print(np.argmax(trial_rate != 0))

    #rate = np.atleast_2d(rate)

    #print('here')

    #print('shape')
    #print(np.shape(rate))

    s = np.zeros_like(rate)

    for n in range(rate.shape[1]):
        s[:, n] = spike_generator(rate[:, n], dt, t_ref, t_ref_rel, rec)
        #print('here6')

    #print(len(s))

    #print(s)

    #filename = "C:/Users/ipboy/Documents/GitHub/ModelingEffort/Single-Channel/Model/Model-Core/Model-Main/test_spike_train.mat"
    
    #s = scipy.io.loadmat(filename)
    #print('here')
    #print(s['s'])

    #s = s['s']

    return s



'''import torch
import scipy
import numpy as np

def spike_generator(rate, dt, t_ref, t_ref_rel, rec, device):
    """
    Generate a Poisson spike train with an absolute and relative refractory period.
    """
    dt_sec = dt / 1000  # ms to seconds
    n = rate.size(0)  # length of the rate tensor (size 0 corresponds to the first dimension)
    spike_train = torch.zeros(n, device=device)
    spike_times = []

    n_refab = int(15 / 1000 / dt_sec)  # number of samples for ref. period window
    tw = torch.arange(n_refab + 1, device=device)

    t_ref_samp = (t_ref / 1000 / dt_sec).to(device).int()
    t_rel_samp = (t_ref_rel / 1000 / dt_sec).to(device).int()

    # Recovery function based on Schaette et al. 2005
    with torch.no_grad():  # No need to track gradients here
        w = torch.pow(tw - t_ref_samp, rec) / (
            torch.pow(tw - t_ref_samp, rec) + torch.pow(t_rel_samp, rec)
        )
        w[tw < t_ref_samp] = 0
        w = torch.nan_to_num(w)

    x = torch.rand(n, device=device)

    for i in range(n):
        if spike_times and i - spike_times[-1] < n_refab:
            rate[i] *= w[i - spike_times[-1]]

        if x[i] < dt_sec * rate[i]:
            spike_train[i] = 1
            spike_times.append(i)

    return spike_train


def gen_poisson_inputs(trial, loc_num, label, t_ref, t_ref_rel, rec, device):
    dt = 0.1  # ms
    label_clean = label.strip("'")
    matfile_path = "C:/Users/ipboy/Documents/GitHub/ModelingEffort/Single-Channel/Model/Model-Core/Model-Main/run/1-channel-paper/solve"
    filename = f"{matfile_path}/IC_spks_{label_clean}.mat"

    data = scipy.io.loadmat(filename)
    temp = np.array(data['spks'])

    loc_size = temp.shape[0] // 24
    print('trial')
    print(trial)
    trial_rate = temp[:, :, int(trial)]
    trial_rate = np.reshape(trial_rate, (-1, trial_rate.shape[1]))

    rate = trial_rate
    s = torch.zeros_like(torch.tensor(rate, device=device))

    for n in range(rate.shape[1]):
        s[:, n] = spike_generator(torch.tensor(rate[:, n], device=device), dt, t_ref, t_ref_rel, rec, device)

    return s'''

