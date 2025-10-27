import set_options
import declarations
from BuildFile import Forwards_Method, Compile_Solve
from argparse import ArgumentParser
import numpy as np
import prep_input_data_for_prototypeing
import prep_input_times_for_prototypeing
import time



#Set your runtime options
opts = set_options.options()
#Declare the network architecture
arch = declarations.Declare_Architecture(opts)
#Build the forwards euler loop
file_body_forwards = Forwards_Method.Euler_Compiler(arch[0],arch[1],arch[2],opts)
#Compile a solve file (python or c++)
solve_file = Compile_Solve.solve_file_generator(solve_file_body = file_body_forwards, cpp_gen = 0)
from BuildFile import generated_solve_file

#Prep the input data    
on_spks = prep_input_data_for_prototypeing.gen_poisson_inputs_parallel(10,label="on")
off_spks = prep_input_data_for_prototypeing.gen_poisson_inputs_parallel(10,label="off")


noise_token = prep_input_times_for_prototypeing.gen_poisson_times_parallel(
    num_trials=10, N_pop=1, dt=0.1, FR=8.0, std=0.0, simlen=29801, refrac_ms=1.0, seed=0
)  # shape: (10, 35000, 50) 

start = time.perf_counter()   # high-res timer

for tt in range(50):
    ouput = generated_solve_file.solve_run([on_spks,off_spks],noise_token)

elapsed = time.perf_counter() - start
print(f"{elapsed*1000:.2f} ms")

#Prep the noise tokens
#noise_tokens = 


#print(on_spks)
#print(np.shape(on_spks))

#print(sum(sum(off_spks)))


