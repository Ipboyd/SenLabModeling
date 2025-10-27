def options():
   
    dt = 0.1                     #Simulation timestep size
    sim_len = 29800              #Length of simulation
    N_trials = 10                #Number of trials
    N_channels = 1               #Number of spatial channels
    N_batch = 1                  #Batch size

    return {'dt' : dt, 'sim_len' : sim_len, 'N_batch' : N_batch, 'N_channels' : N_channels, 'N_trials' : N_trials}
