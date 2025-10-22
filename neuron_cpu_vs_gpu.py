# Importing necessary libraries

import sys, os, time, subprocess
import numpy as np
from brian2 import *

# Establishing CLI Flags

USE_CPP_STANDALONE = ("--standalone" in sys.argv)
USE_GPU = ("--gpu" in sys.argv)
NO_BUILD = ("--nobuild" in sys.argv)

# Setting default clock and preferences

defaultclock.dt = 0.1*ms  
prefs.core.default_float_dtype = float64

# Equations 

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

# Network Build Function

def build_network(SCALE):
    """Build a scalable E-I network with Poisson input
    
    Returns:
        Network: Brian2 Network object containing all simulation components
    """
    
    # Population sizes (maintain 4:1 excitatory:inhibitory ratio)
    N_poisson = SCALE
    N_exc = int(SCALE * 0.8)  # 80% excitatory
    N_inh = int(SCALE * 0.2)  # 20% inhibitory
    
    print(f"\n  Network scale: {SCALE}")
    print(f"    Poisson input: {N_poisson} neurons")
    print(f"    Excitatory: {N_exc} neurons")
    print(f"    Inhibitory: {N_inh} neurons")
    
    # Poisson Input (shared by both populations) 
    poisson_input = PoissonGroup(N_poisson, rates=15*Hz)
    
    # Excitatory Population 
    exc_neurons = NeuronGroup(
        N_exc, model=eqs,
        threshold="V > V_thresh",
        reset="V = V_reset; g_ad += g_inc",
        refractory="t_ref",
        method="euler"
    )
    
    exc_neurons.tau = 20*ms
    exc_neurons.tau_ad = 200*ms
    exc_neurons.tau_s_on = 5*ms
    exc_neurons.tau_s_off = 10*ms
    exc_neurons.E_L, exc_neurons.E_k, exc_neurons.E_on, exc_neurons.E_off = 0.0, -1.0, 1.0, -1.0
    exc_neurons.R, exc_neurons.g_on, exc_neurons.g_off = 1.0, 0.04, 0.02
    exc_neurons.Itonic, exc_neurons.Imask = 0.30, 1.0
    exc_neurons.V_thresh, exc_neurons.V_reset, exc_neurons.g_inc = 0.5, -0.5, 0.10
    exc_neurons.namespace['t_ref'] = 2*ms
    exc_neurons.V = 'rand() * 0.1 - 0.25'  # Random initial conditions
    exc_neurons.g_ad = 0.0
    exc_neurons.s_on = 0.0
    exc_neurons.s_off = 0.0
    
    # Inhibitory Population
    inh_neurons = NeuronGroup(
        N_inh, model=eqs,
        threshold="V > V_thresh",
        reset="V = V_reset; g_ad += g_inc",
        refractory="t_ref",
        method="euler"
    )
    
    inh_neurons.tau = 10*ms  # Faster dynamics
    inh_neurons.tau_ad = 100*ms
    inh_neurons.tau_s_on = 5*ms
    inh_neurons.tau_s_off = 10*ms
    inh_neurons.E_L, inh_neurons.E_k, inh_neurons.E_on, inh_neurons.E_off = 0.0, -1.0, 1.0, -1.0
    inh_neurons.R, inh_neurons.g_on, inh_neurons.g_off = 1.0, 0.04, 0.02
    inh_neurons.Itonic, inh_neurons.Imask = 0.35, 1.0  # Slightly higher drive
    inh_neurons.V_thresh, inh_neurons.V_reset, inh_neurons.g_inc = 0.5, -0.5, 0.05
    inh_neurons.namespace['t_ref'] = 1*ms
    inh_neurons.V = 'rand() * 0.1 - 0.25'
    inh_neurons.g_ad = 0.0
    inh_neurons.s_on = 0.0
    inh_neurons.s_off = 0.0
    
    # SYNAPSES (4 connection types) 
    
    # S1: Poisson → Excitatory
    S1 = Synapses(poisson_input, exc_neurons, model="w : 1", on_pre="s_on_post += w", method="euler")
    S1.connect(p=0.2)  # 20% connection probability
    S1.w = 0.15
    
    # S2: Poisson → Inhibitory
    S2 = Synapses(poisson_input, inh_neurons, model="w : 1", on_pre="s_on_post += w", method="euler")
    S2.connect(p=0.2)
    S2.w = 0.15
    
    # S3: Excitatory → Inhibitory (E drives I)
    S3 = Synapses(exc_neurons, inh_neurons, model="w : 1", on_pre="s_on_post += w", method="euler")
    S3.connect(p=0.3)  # 30% connectivity
    S3.w = 0.25
    
    # S4: Inhibitory → Excitatory (I suppresses E - feedback control)
    S4 = Synapses(inh_neurons, exc_neurons, model="w : 1", on_pre="s_off_post += w", method="euler")
    S4.connect(p=0.5)  # 50% connectivity (strong inhibitory control)
    S4.w = 0.30
    
    # Monitors
    M_exc = StateMonitor(exc_neurons, ["V"], record=True)
    M_inh = StateMonitor(inh_neurons, ["V"], record=True)
    Sp_exc = SpikeMonitor(exc_neurons)
    Sp_inh = SpikeMonitor(inh_neurons)
    
    # Inside build_network()
    print(f"    Synapse counts:")
    if get_device().__class__.__name__ == "CPPStandaloneDevice":
        s1_count = int(round(0.2 * N_poisson * N_exc))
        s2_count = int(round(0.2 * N_poisson * N_inh))
        s3_count = int(round(0.3 * N_exc * N_inh))
        s4_count = int(round(0.5 * N_inh * N_exc))
    else:
        s1_count = len(S1)
        s2_count = len(S2)
        s3_count = len(S3)
        s4_count = len(S4)

    print(f"      S1 (Poisson→Exc): {s1_count}")
    print(f"      S2 (Poisson→Inh): {s2_count}")
    print(f"      S3 (Exc→Inh): {s3_count}")
    print(f"      S4 (Inh→Exc): {s4_count}")
    print(f"      Total synapses: {s1_count + s2_count + s3_count + s4_count}")
    
    net = Network(poisson_input, exc_neurons, inh_neurons, 
                  S1, S2, S3, S4, M_exc, M_inh, Sp_exc, Sp_inh)
    return net


# MAIN BENCHMARK CONFIGURATION 

runtime = 10_000 * defaultclock.dt  # 10k steps
scales = [10, 50, 100, 500, 1000]
num_trials = 10

if __name__ == "__main__":

    if USE_GPU:
        print("MODE: GPU (CUDA Standalone)")
        try:
            set_device('cuda_standalone', build_on_run=False)
        except Exception as e:
            print("ERROR: Brian2CUDA not installed or GPU not available.")
            print(f"Details: {e}")
            sys.exit(1)
    elif USE_CPP_STANDALONE:
        print("MODE: CPU (C++ Standalone)")
        set_device('cpp_standalone', build_on_run=False)
    else:
        print("MODE: Python Runtime")

    print(f"Runtime per trial: {runtime}")
    print(f"Timestep: {defaultclock.dt}")
    print(f"Trials per scale: {num_trials}")

    results = []

    for scale in scales:
        print(f"\n{'='*70}")
        print(f"SCALE = {scale}")

        start_scope()

        if USE_CPP_STANDALONE or USE_GPU:
            BUILD_DIR = os.path.join(os.getcwd(), f"b2_network_build_scale_{scale}")
            set_device(
                'cuda_standalone' if USE_GPU else 'cpp_standalone',
                directory=BUILD_DIR, build_on_run=False
            )

        net = build_network(scale)

        if not (USE_CPP_STANDALONE or USE_GPU):
            print(f"\n  Running {num_trials} trials...")
            trial_times = []
            for trial in range(num_trials):
                start_time = time.time()
                net.run(runtime, report=None)
                elapsed = time.time() - start_time
                trial_times.append(elapsed)
                print(f"    Trial {trial+1}: {elapsed:.3f} s")

            mean_time = np.mean(trial_times)
            std_time = np.std(trial_times)
            print(f"  Mean: {mean_time:.3f} s, Std: {std_time:.3f} s")
            results.append((scale, mean_time, std_time))

        else:
            print(f"\n  Generating and building {'CUDA' if USE_GPU else 'C++'} code...")
            gen_start = time.time()
            net.run(runtime)
            device.build(directory=BUILD_DIR, compile=True, run=False, clean=True)
            gen_time = time.time() - gen_start
            print(f"  Generation + build time: {gen_time:.3f} s")

            bin_path = os.path.join(BUILD_DIR, "main")
            if os.path.exists(bin_path):
                print(f"  Running compiled binary {num_trials} times...")
                trial_times = []
                for trial in range(num_trials):
                    run_start = time.time()
                    subprocess.run([bin_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    run_time = time.time() - run_start
                    trial_times.append(run_time)
                    print(f"    Trial {trial+1}: {run_time:.3f} s")

                mean_time = np.mean(trial_times)
                std_time = np.std(trial_times)
                print(f"  Mean: {mean_time:.3f} s, Std: {std_time:.3f} s")
                results.append((scale, mean_time, std_time))
            else:
                print("  WARNING: Compiled binary not found")
                results.append((scale, None, None))

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    mode_str = "GPU (CUDA Standalone)" if USE_GPU else ("CPU (C++ Standalone)" if USE_CPP_STANDALONE else "Python Runtime")
    print(f"Mode: {mode_str}")
    print(f"Trials per scale: {num_trials}")
    print(f"\n{'Scale':<10} {'Mean Time (s)':<18} {'Std Dev (s)':<15} {'Time/Neuron (ms)':<20}")
    print("-" * 70)
    
    for scale, mean_time, std_time in results:
        if mean_time is not None:
            total_neurons = 2 * scale
            time_per_neuron_ms = (mean_time / total_neurons) * 1000
            print(f"{scale:<10} {mean_time:<18.3f} {std_time:<15.3f} {time_per_neuron_ms:<20.3f}")
        else:
            print(f"{scale:<10} {'N/A':<18} {'N/A':<15} {'N/A':<20}")
    
    print("=" * 70)
