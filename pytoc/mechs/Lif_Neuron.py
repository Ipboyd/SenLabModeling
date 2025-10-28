#The following script acts as a "constructor" that tells the euler_compiler how to put things together.
#The function takes in all the variabeles for our LIF model and returns them as a dictionary. 
#This allows for the user to flexibly say WHICH variables they want to change or just keep them as default.
#It also names the variables per-neuron with the user specified "name"

import numpy as np
def Build_Vars(C = 0.1, g_L = 1/200, E_L = -65, noise = 0, t_ref = 1, E_k = -80, tau_ad = 5, g_inc = 0, Itonic = 0, Imask='', N_pop=1 , V_thresh = -47, V_reset = -54, name='', is_output = 0, is_noise = 0, is_input = 0, g_postIC = 0.17, E_exc = 0, netcon = '', N_chans = 1, nSYN = 0, noise_E_exc = 0, tauR_N = 0.7, tauD_N = 1.5, response = ''):

    #All basic variables that are made from the declarations
    R = 1/g_L
    tau = C*R
    noise_scale = (tauD_N/tauR_N)**(tauR_N/(tauD_N-tauR_N))

    #Variable Explainations
    #C             % membrane capacitance [nF]
    #g_L           % leak resistance [uS]
    #R             % membrane resistance [Mohm]
    #tau           % membrane time constant [ms]
    #E_L           % equilibrium potential/resting potential [mV]
    #noise         % noise [nA]
    #t_ref         % refractory period [ms]
    #E_k           % spike-rate adaptation potential [mV]
    #tau_ad        % spike-rate adaptation time constant [ms]
    #g_inc         % spike-rate adaptation increment [uS]
    #Itonic        % Injected current [nA]
    #V_thresh      % spike threshold [mV]
    #V_reset       % reset voltage [mV]

    #Error out if name is not given
    if len(name) < 1:
        raise ValueError("A name must be given to your Neuron. Set -> name = 'name of your choice' in declarations")
    
    #If no netcon is given just keep everything within channel
    if len(netcon) < 1:
        netcon = f'np.eye({N_chans})[None,:,:]' 
        Imask = f'np.ones((1,{N_chans}))[None,:,:]'

    #Build dictionary
    return {f'{name}_C' : C,f'{name}_g_L' : g_L,f'{name}_E_L' : E_L,f'{name}_noise' : noise,f'{name}_t_ref' : t_ref,f'{name}_E_k' : E_k,f'{name}_tau_ad' : tau_ad,f'{name}_g_inc' : g_inc,f'{name}_Itonic' : Itonic, f'{name}_Imask' : Imask, f'{name}_R' : R,f'{name}_tau' : tau, f'{name}_V_thresh' : V_thresh, f'{name}_V_reset' : V_reset, 'name' : name, 'is_output' : is_output, 'is_noise' : is_noise, 'is_input' : is_input, f'{name}_g_postIC' : g_postIC, f'{name}_E_exc' : E_exc, f'{name}_netcon' : netcon, f'{name}_nSYN' : nSYN, f'{name}_noise_E_exc' : noise_E_exc, f'{name}_tauR_N' : tauR_N, f'{name}_tauD_N' : tauD_N, f'{name}_noise_scale' : noise_scale, 'response' : response}


