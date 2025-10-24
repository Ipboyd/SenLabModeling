import set_options
import declarations
from BuildFile import Forwards_Method, Compile_Solve
from argparse import ArgumentParser
import numpy as np
import prep_input_data_for_prototypeing

#Set your runtime options
opts = set_options.options()
#Declare the network architecture
arch = declarations.Declare_Architecture(opts)
#Build the forwards euler loop
file_body_forwards = Forwards_Method.Euler_Compiler(arch[0],arch[1],arch[2],opts)
#Compile a solve file (python or c++)
solve_file = Compile_Solve.solve_file_generator(solve_file_body = file_body_forwards, cpp_gen = 0)

#Prep the input data    
on_spks = prep_input_data_for_prototypeing.gen_poisson_inputs_parallel(10,label="on")
off_spks = prep_input_data_for_prototypeing.gen_poisson_inputs_parallel(10,label="off")

#Prep the noise tokens
noise_tokens = 


#print(on_spks)
#print(np.shape(on_spks))

#print(sum(sum(off_spks)))


