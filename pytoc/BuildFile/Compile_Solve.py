import os

def solve_file_generator(solve_file_body = '', cpp_gen = 0):
    
    #Add necessary imports
    imports_string = bring_in_imports()

    #Generate py header
    header = gen_py_header()

    #Gernerate cpp header
    if cpp_gen == 1:
        header = gen_cpp_header() + header

    solve_file_complete = imports_string + header + solve_file_body

    #Compile file
    compile_file(solve_file_complete,cpp_gen)


def bring_in_imports():

    import_string = 'import numpy as np\n'

    return import_string


def gen_py_header():
    
    header = 'def solve_run(inputs,noise_token):\n'
    
    return header

def gen_cpp_header():


    return

def compile_file(solve_file_complete,cpp_gen):

    #Write to a python file
    out_filename = f"generated_solve_file.py"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    outpath = os.path.join(script_dir, out_filename)
    with open(outpath, "w") as f:
        f.write(solve_file_complete)

    print(f"generated_solve_file.py has been created.")
    
     

    
