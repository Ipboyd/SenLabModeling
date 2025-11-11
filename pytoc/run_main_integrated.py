import set_options
import declarations
from BuildFile import Forwards_Method, Compile_Solve
import numpy as np
import time
import Update_params
import InitParams
from scipy.io import loadmat, savemat
import yaml
import os
from strf_handler import call_strfs
from input_handler import call_inputs

class runSimulation(object):

    #Little control pannel for now (Eventually move this to a yaml file or whatever makes the most sense)

    gen_strfs_toggle = 0  #Toggle generating the STRFs
    gradients_toggle = 0  #Toggle generating the graidnets in the forwards process *Also toggles running epochs

    #Run STRF
    if gen_strfs_toggle == 1:
        call_strfs()
    #PreProcessesing  #Note! Recheck everything once you start running multichannel inputs -- check where the gain control for the tuning curves is and make sure we arn't doing extra steps
    #Will also need two worry about how exactly you are going to parse spks once you have multiple data streams
    spks = call_inputs()

    #Current single channel test parsing
    on_spks = np.transpose(spks[f'locs_masker_None_target_0_on'][f'stimulus_0_IC_spks'],(2,0,1))
    off_spks = np.transpose(spks[f'locs_masker_None_target_0_off'][f'stimulus_0_IC_spks'],(2,0,1))
    noise = np.transpose(spks['noise_masker_None_target_0'],(2,0,1))

    #batch,trials,channels,timecourse

    #Set options
    opts = set_options.options()
    #Declare architecture
    arch = declarations.Declare_Architecture(opts)
    #Build the forwards euler loop
    file_body_forwards = Forwards_Method.Euler_Compiler(arch[0],arch[1],arch[2],opts)
    #Compile a solve file (python or c++)
    solve_file = Compile_Solve.solve_file_generator(solve_file_body = file_body_forwards, cpp_gen = 1)
    from BuildFile import generated_solve_file


    ############
    #- Move the data loading to a seperate file and make it toggleable

    # -- Load in data
    filename = f"C:/Users/ipboy/Documents/Github/ModelingEffort/Multi-Channel/Plotting/OliverDataPlotting/PicturesToFit/picture_fit{7}contra.mat"
    data = loadmat(filename)['picture'].astype(np.float32)  #trials,timecourse
    data = data[:,:,None]


    num_params = 4
    batch_size = opts['N_batch']
    p = InitParams.pinit(batch_size,num_params)


    #Grad Params
    beta1, beta2 = 0.99, 0.9995 
    eps = 1e-6
    lr = 5e-3

    #Init mvt
    m = np.zeros((num_params,batch_size))
    v = np.zeros((num_params,batch_size))
    t = 0
    
    losses = []
    param_tracker = []

    best_loss = 1e32

    best_output = []

    start = time.perf_counter()
    for epoch in range(20):

        #Make it so that you don't have to supply data if you are not running gradients
        output, grads = generated_solve_file.solve_run(on_spks,off_spks,noise,data,p) #Python Verison to build
    
        #Use adam optimizer on grads
        m, v, p, t = Update_params.adam_update(m, v, p, t, beta1, beta2, lr, eps, grads)


    elapsed = time.perf_counter() - start
    print(f"{elapsed*1000:.2f} ms")

    savemat("output_compressed.mat", {"output": output}, do_compression=True)

if __name__ == "__main__":


    run_sim = runSimulation()