import sys, os, time, subprocess
from brian2 import *

# --------- CLI flags ---------
USE_CPP_STANDALONE = ("--standalone" in sys.argv)
NO_BUILD = ("--nobuild" in sys.argv)
USE_PARALLEL = ("--parallel" in sys.argv)

# Building appropriate build directory
BUILD_DIR = os.path.join(os.getcwd(), "b2_two_neuron_build")
if USE_PARALLEL:
    BUILD_DIR = os.path.join(os.getcwd(), "b2_two_neuron_build_parallel")

print(f"Build dir: {BUILD_DIR}")

defaultclock.dt = 0.1*ms  # explicit Euler step size

if USE_CPP_STANDALONE:
    set_device('cpp_standalone', directory=BUILD_DIR, build_on_run=(not NO_BUILD))
    
    # Enabling OpenMP parallelization if requested
    if USE_PARALLEL:
        # Setting OpenMP threads
        prefs.devices.cpp_standalone.openmp_threads = 0  # 0 = auto-detect cores and use all
        
        omp_include = '/opt/homebrew/opt/libomp/include'
        omp_lib = '/opt/homebrew/opt/libomp/lib'
        
        prefs.codegen.cpp.extra_compile_args_gcc = [
            '-Xpreprocessor', '-fopenmp',
            f'-I{omp_include}'
        ]
        prefs.codegen.cpp.extra_link_args = [
            f'-L{omp_lib}',
            '-lomp'
        ]
        prefs.codegen.cpp.libraries = []  # Clearing this to avoid duplicate -lomp
        
        print(f"OpenMP parallelization enabled (auto-detect cores)")
        print(f"macOS OpenMP paths configured:")
        print(f"  Include: {omp_include}")
        print(f"  Library: {omp_lib}")
        print(f"openmp_threads preference: {prefs.devices.cpp_standalone.openmp_threads}")

prefs.core.default_float_dtype = float64

# --------- Equations ---------
eqs = Equations("""
dV/dt = ((E_L - V)
         - R*g_ad*(V - E_k)
         - R*(g_on*s_on*(V - E_on) + g_off*s_off*(V - E_off))
         + R*Itonic*Imask) / tau : 1 (unless refractory)
dg_ad/dt = -g_ad/tau_ad : 1
ds_on/dt = -s_on/tau_s_on : 1
ds_off/dt = -s_off/tau_s_off: 1
E_L     : 1
E_k     : 1
E_on    : 1
E_off   : 1
R       : 1
g_on    : 1
g_off   : 1
Itonic  : 1
Imask   : 1
tau        : second
tau_ad     : second
tau_s_on   : second
tau_s_off  : second
V_thresh   : 1
V_reset    : 1
g_inc      : 1
""")

# --------- Neuron group (N=2) ---------
N = 2
G = NeuronGroup(
    N, model=eqs,
    threshold="V > V_thresh",
    reset="V = V_reset; g_ad += g_inc",
    refractory="t_ref",
    method="euler"
)

# Parameters
G.tau = 20*ms
G.tau_ad = 200*ms
G.tau_s_on = 5*ms
G.tau_s_off = 10*ms
G.E_L, G.E_k, G.E_on, G.E_off = 0.0, -1.0, 1.0, -1.0
G.R, G.g_on, G.g_off = 1.0, 0.04, 0.00
G.Itonic, G.Imask = 0.30, 1.0
G.V_thresh, G.V_reset, G.g_inc = 0.5, -0.5, 0.10
t_ref = 2*ms
G.namespace['t_ref'] = t_ref

G.V = [-0.2, -0.25]
G.g_ad = 0.0
G.s_on = 0.0
G.s_off = 0.0

# Synapses
S = Synapses(G, G, model="w : 1", on_pre="s_on_post += w", method="euler")
S.connect(i=[0, 1], j=[1, 0])
S.w = 0.30

M = StateMonitor(G, ["V"], record=True)
Sp = SpikeMonitor(G)

# --------- Run and timing ---------
runtime = 30_000 * defaultclock.dt  # 30k steps
num_trials = 10

def run_trials_python():
    """Run 10 trials in Python (non-standalone) mode"""
    for i in range(num_trials):
        start = time.time()
        run(runtime, report=None)
        dur = time.time() - start
        print(f"Trial {i+1}: {dur:.3f} s")

if not USE_CPP_STANDALONE:
    print("=" * 60)
    print("MODE 1: Running in Python runtime mode...")
    print("=" * 60)
    start_total = time.time()
    run_trials_python()
    total_time = time.time() - start_total
    avg_time = total_time / num_trials
    print(f"\nTotal runtime: {total_time:.3f} s")
    print(f"Average per trial: {avg_time:.3f} s")
    print("=" * 60)
else:
    mode_name = "MODE 3: C++ Standalone + OpenMP Parallelization" if USE_PARALLEL else "MODE 2: C++ Standalone (no parallelization)"
    print("=" * 60)
    print(f"{mode_name}")
    print("=" * 60)
    print("Generating C++ standalone project...")
    gen_start = time.time()
    run(runtime)  # triggers code generation (and build if not NO_BUILD)
    gen_end = time.time()
    print(f"Generation + build time: {gen_end - gen_start:.3f} s")
    print(f"Standalone project under: {BUILD_DIR}")
    
    # Verify OpenMP was actually enabled in the build
    if USE_PARALLEL:
        makefile_path = os.path.join(BUILD_DIR, "makefile")
        if os.path.exists(makefile_path):
            with open(makefile_path, 'r') as f:
                makefile_content = f.read()
                if '-fopenmp' in makefile_content or 'omp' in makefile_content.lower():
                    print("OpenMP flags detected in makefile")
                else:
                    print("WARNING: No OpenMP flags found in makefile")
        
        # Check for pragma directives in generated code
        code_objects_dir = os.path.join(BUILD_DIR, "code_objects")
        if os.path.exists(code_objects_dir):
            pragma_found = False
            for root, dirs, files in os.walk(code_objects_dir):
                for file in files:
                    if file.endswith('.cpp'):
                        filepath = os.path.join(root, file)
                        with open(filepath, 'r') as f:
                            if '#pragma omp' in f.read():
                                pragma_found = True
                                break
                if pragma_found:
                    break
            if pragma_found:
                print("OpenMP pragmas detected in generated code")
            else:
                print("WARNING: No OpenMP pragmas found in generated code")
    
    if not NO_BUILD:
        bin_path = os.path.join(BUILD_DIR, "main")
        if os.path.exists(bin_path):
            print(f"\nRunning compiled binary {num_trials} times for timing...")
            trial_times = []
            for i in range(num_trials):
                run_start = time.time()
                subprocess.run([bin_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                run_end = time.time()
                trial_time = run_end - run_start
                trial_times.append(trial_time)
                print(f"Trial {i+1}: {trial_time:.3f} s")
            
            avg_time = sum(trial_times) / len(trial_times)
            print(f"\nAverage per trial: {avg_time:.3f} s")
            print("=" * 60)
        else:
            print("Compiled binary not found. Try rebuilding without --nobuild.")