#The following script acts as a "constructor" that tells the euler_compiler how to put things together.
#The function takes in all the variabeles for our LIF model and returns them as a dictionary. 
#This allows for the user to flexibly say WHICH variables they want to change or just keep them as default.
#It also names the variables per-synapse with the user specified "name"

import numpy as np
def Build_Vars(ESYN = 0, tauD = 1.5, tauR = 0.3, delay = 0, gSYN = 1, fF = 0, fP = 0, tauF = 180, tauP = 60, maxF=4 , name='', netcon='', is_convergent = 0, N_chans = 1):

    #All basic variables that are made from the declarations
    scale = (tauD/tauR)**(tauR/(tauD-tauR))

    #Varialbe Explainations
    #ESYN            % reversal potential [mV]
    #tauD            % decay time [ms]
    #tauR            % rise time [ms]
    #delay           % synaptic delay [ms]
    #gSYN            % synaptic connectivity [uS]
    #fF              % degree of synaptic facilitation ([0 1])
    #fP              % degree of synaptic depression ([0 1])
    #tauF            % facilitation decay time [ms]
    #tauP            % depression decay time [ms]
    #maxF            % maximum facilitation strength

    #Error out if name is not given
    if len(name) < 1:
        raise ValueError("A name must be given to your Neuron. Set -> name = 'name of your choice' in declarations")

    #If no netcon is given just keep everything within channel
    if len(netcon) < 1:
        if is_convergent == 0:
            netcon = f'np.eye({N_chans})[None,:,:]' 
        if is_convergent == 1:
            netcon = f'np.ones((1,{N_chans}))[None,:,:]' 


    #Build dictionary
    return {f'{name}_ESYN' : ESYN,f'{name}_tauD' : tauD,f'{name}_tauR' : tauR,f'{name}_PSC_delay' : delay,f'{name}_gSYN' : gSYN,f'{name}_PSC_fF' : fF,f'{name}_PSC_fP' : fP,f'{name}_tauF' : tauF,f'{name}_tauP' : tauP,f'{name}_PSC_maxF' : maxF , 'name' : name, f'{name}_netcon' : netcon, f'{name}_scale' : scale}



