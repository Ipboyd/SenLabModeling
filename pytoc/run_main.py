import set_options
import declarations
from BuildFile import Forwards_Method, Compile_Solve
import prep_input_data2

#Set your runtime options
opts = set_options.options()
#Declare the network architecture
arch = declarations.Declare_Architecture(opts)
#Build the forwards euler loop
file_body_forwards = Forwards_Method.Euler_Compiler(arch[0],arch[1],arch[2],opts)
#Compile a solve file (python or c++)
solve_file = Compile_Solve.solve_file_generator(solve_file_body = file_body_forwards, cpp_gen = 0)
#Generate external model inputs
inputs, noise_tokens = prepare_inputs


