import os
import time
import subprocess
from brian2 import *

# ============================================================
#  COMPILER & DEVICE SETUP
# ============================================================

# Use Homebrew GCC
os.environ['CXX'] = '/opt/homebrew/bin/g++-15'

# Enable OpenMP via compiler/link flags
prefs['codegen.cpp.extra_compile_args'] = ['-fopenmp', '-O3']
prefs['codegen.cpp.extra_link_args'] = ['-fopenmp']

# Use 64-bit floats
prefs.core.default_float_dtype = float64

# Set integration timestep
defaultclock.dt = 0.1*ms

# Set device
set_device('cpp_standalone')
device.reinit()
device.activate()

# Specify build directory
build_dir = os.path.join(os.getcwd(), "b2_two_neuron_build_parallel")
device.build_directory = build_dir

print(f"Using compiler (from CXX env): {os.environ.get('CXX', 'default system compiler')}")
print(f"Build directory: {build_dir}")

# ============================================================
#  MODEL EQUATIONS
# ============================================================

eqs = Equations("""
dV/dt = ((E_L - V)
         - R*g_ad*(V - E_k)
         - R*(g_on*s_on*(V - E_on) + g_off*s_off*(V - E_off))
         + R*Itonic*Imask) / tau : 1 (unless refractory)
dg_ad/dt = -g_ad/tau_ad : 1
ds_on/dt = -s_on/tau_s_on : 1
ds_off/dt = -s_off/tau_s_off : 1
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

# ============================================================
#  NEURON GROUP
# ============================================================

N = 10000
t_ref = 2*ms

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
G.namespace['t_ref'] = t_ref

# Initial conditions
G.V = -0.2
G.g_ad = 0.0
G.s_on = 0.0
G.s_off = 0.0

# ============================================================
#  SYNAPSES
# ============================================================

S = Synapses(G, G, model="w : 1", on_pre="s_on_post += w", method="euler")
S.connect(i=[0,1], j=[1,0])
S.w = 0.30

# ============================================================
#  MONITORS
# ============================================================

M = StateMonitor(G, "V", record=True)
Sp = SpikeMonitor(G)

# ============================================================
#  CHECK OPENMP PRAGMAS
# ============================================================

pragma_count = 0
for root, _, files in os.walk(build_dir):
    for file in files:
        if file.endswith(".cpp"):
            with open(os.path.join(root, file)) as f:
                for line in f:
                    if "#pragma omp" in line:
                        pragma_count += 1

if pragma_count > 0:
    print(f"Found {pragma_count} OpenMP pragma(s) in generated code!")
else:
    print("No OpenMP pragmas found — Brian2 may have skipped parallelization for this model.")

# ============================================================
#  RUN EXECUTABLE
# ============================================================

bin_path = os.path.join(build_dir, "main")
num_trials = 5

if not os.path.exists(bin_path):
    raise FileNotFoundError(f"Binary not found at {bin_path}")

print(f"\nRunning compiled binary {num_trials} times for timing...")
times = []
for i in range(num_trials):
    t0 = time.time()
    subprocess.run([bin_path], check=True)
    t1 = time.time()
    trial_time = t1 - t0
    times.append(trial_time)
    print(f"Trial {i+1}: {trial_time:.3f} s")

avg_time = sum(times)/len(times)
print(f"\nAverage per trial: {avg_time:.3f} s")
print("="*60)

print(f"Total spikes recorded: {Sp.num_spikes}")
print("Simulation finished successfully ")
