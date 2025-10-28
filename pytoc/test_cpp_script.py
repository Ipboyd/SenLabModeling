#C++ Test


from BuildFile import generated_solve_file_test as gs

print(gs.__file__)



import set_options
import declarations
from BuildFile import Forwards_Method, Compile_Solve
from argparse import ArgumentParser
import numpy as np
import prep_input_data_for_prototypeing
import prep_input_times_for_prototypeing
import time


#Prep the input data    
on_spks = prep_input_data_for_prototypeing.gen_poisson_inputs_parallel(10,label="on")
off_spks = prep_input_data_for_prototypeing.gen_poisson_inputs_parallel(10,label="off")


noise_token = prep_input_times_for_prototypeing.gen_poisson_times_parallel(
    num_trials=10, N_pop=1, dt=0.1, FR=8.0, std=0.0, simlen=29801, refrac_ms=1.0, seed=0
)  # shape: (10, 35000, 50) 


def f64c(x):
    # cast to float64 and ensure C-contiguous layout
    return np.ascontiguousarray(x, dtype=np.float64)

on  = f64c(on_spks)
off = f64c(off_spks)
noi = f64c(noise_token)

start = time.perf_counter()   # high-res timer

ouput = gs.solve_run(on,off,noi)

elapsed = time.perf_counter() - start
print(f"{elapsed*1000:.2f} ms")