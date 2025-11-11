import set_options
import declarations
from BuildFile import Forwards_Method, Compile_Solve
from argparse import ArgumentParser
import numpy as np
import prep_input_data_for_prototypeing
import prep_input_times_for_prototypeing
import genPoissonTimes
import genPoissonInputs
import time
import Update_params
import InitParams
from scipy.io import loadmat, savemat




#Set your runtime options
opts = set_options.options()
#Declare the network architecture
arch = declarations.Declare_Architecture(opts)
#Build the forwards euler loop
file_body_forwards = Forwards_Method.Euler_Compiler(arch[0],arch[1],arch[2],opts)
#Compile a solve file (python or c++)
solve_file = Compile_Solve.solve_file_generator(solve_file_body = file_body_forwards, cpp_gen = 1)
from BuildFile import generated_solve_file

#Prep the input data    
all_on_spks = []
all_off_spks = []
noise = []

for k in range(10):
    spks = genPoissonTimes.gen_poisson_times(1,0.1,8,0)
    noise.append(spks)

print(np.shape(noise))

for k in range(10):
    on_spks = genPoissonInputs.gen_poisson_inputs(1,5,'on',1,1,2,1)
    all_on_spks.append(on_spks)

for k in range(10):
    off_spks = genPoissonInputs.gen_poisson_inputs(1,5,'off',1,1,2,1)
    all_off_spks.append(off_spks)

#print(np.shape(all_on_spks))

noise = np.array(noise)
all_on_spks = np.array(all_on_spks)
all_off_spks = np.array(all_off_spks)


def f64c(x):
    # cast to float64 and ensure C-contiguous layout
    return np.ascontiguousarray(x, dtype=np.float64)



#print(np.shape(noise))

#import matplotlib.pyplot as plt

#savemat("input_tracker.mat", {"on_spks": all_on_spks}, do_compression=True)
#savemat("input_tracker_off.mat", {"off_spks": all_off_spks}, do_compression=True)
#savemat("noise.mat", {"noise": noise}, do_compression=True)

#noise_token = prep_input_times_for_prototypeing.gen_poisson_times_parallel(
#    num_trials=10, N_pop=1, dt=0.1, FR=8.0, std=0.0, simlen=29801, refrac_ms=1.0, seed=0
#)  # shape: (10, 35000, 50) 

start = time.perf_counter()   # high-res timer

# -- Load in data
filename = f"C:/Users/ipboy/Documents/Github/ModelingEffort/Multi-Channel/Plotting/OliverDataPlotting/PicturesToFit/picture_fit{7}contra.mat"
data = loadmat(filename)['picture'].astype(np.float32)  #trials,timecourse
data = data[:,:,None]
#data = np.transpose(data,(2,0,1)) #Transpose things to be Batch,trials,channels,timecouse


num_params = 4
batch_size = opts['N_batch']
p = InitParams.pinit(batch_size,num_params)


on  = f64c(all_on_spks)
off = f64c(all_off_spks)
noi = f64c(noise)
p =  f64c(p)
data = f64c(data)

#Grad Params
beta1, beta2 = 0.99, 0.9995 
eps = 1e-6
lr = 1e-3

#Init mvt
m = np.zeros((num_params,batch_size))
v = np.zeros((num_params,batch_size))
t = 0
    
losses = []
param_tracker = []

best_loss = 1e32

best_output = []

t0 = time.perf_counter()
for epoch in range(1):

    output, grads = generated_solve_file.solve_run(all_on_spks,all_off_spks,noise,data,p) #Python Verison to build
    
    #CPP Imports
    #from BuildFile import generated_solve_file_test as gs   
    #output, grads = gs.solve_run(on,off,noi,data,p) #CPP verison

    #Use adam optimizer on grads
    m, v, p, t = Update_params.adam_update(m, v, p, t, beta1, beta2, lr, eps, grads)


elapsed = time.perf_counter() - start
print(f"{elapsed*1000:.2f} ms")



savemat("output_compressed.mat", {"output": output}, do_compression=True)

#Prep the noise tokens
#noise_tokens = 


#print(on_spks)
#print(np.shape(on_spks))

#print(sum(sum(off_spks)))


