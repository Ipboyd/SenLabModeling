#The following script acts as a "constructor" that tells the euler_compiler how to put things together.
#The function takes in all the variabeles for our LIF model and returns them as a dictionary. 
#This allows for the user to flexibly say WHICH variables they want to change or just keep them as default.
#It also names the variables per-neuron with the user specified "name"

import numpy as np
def Build_Vars(C = 0.1, g_L = 1/200, E_L = -65, noise = 0, t_ref = 1, E_k = -80, tau_ad = 5, g_inc = 0, Itonic = 0, N_pop=1 , V_thresh = -47, V_reset = -54, name=''):

    #All basic variables that are made from the declarations
    R = 1/g_L
    tau = C*R

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

    #Build dictionary
    return {f'{name}_C' : C,f'{name}_g_L' : g_L,f'{name}_E_L' : E_L,f'{name}_noise' : noise,f'{name}_t_ref' : t_ref,f'{name}_E_k' : E_k,f'{name}_tau_ad' : tau_ad,f'{name}_g_inc' : g_inc,f'{name}_Itonic' : Itonic,f'{name}_R' : R,f'{name}_tau' : tau, f'{name}_V_thresh' : V_thresh, f'{name}_V_reset' : V_reset, 'name' : name}


