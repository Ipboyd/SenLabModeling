# build_pairwise_10x10.py
from brian2 import *
from pathlib import Path

set_device('cpp_standalone', build_on_run=False)
start_scope()

defaultclock.dt = 0.1*ms

# Simple LIF with a decaying current; synapses inject current (amp)
eqs = '''
dv/dt     = (-(v-EL) + R*I_syn)/tau : volt (unless refractory)
dI_syn/dt = -I_syn/tau_syn          : amp
'''
EL = -65*mV; Vt = -50*mV; Vr = -60*mV
gL = 10*nS; C = 200*pF
R  = 1/gL
tau = C*R
tau_syn = 5*ms

Npre = 10
Npost = 10

# Pre and post populations
Gpre = NeuronGroup(Npre, eqs, threshold='v>Vt', reset='v=Vr',
                   refractory=2*ms, method='euler',
                   namespace=dict(EL=EL, R=R, tau=tau, tau_syn=tau_syn))
Gpost = NeuronGroup(Npost, eqs, threshold='v>Vt', reset='v=Vr',
                    refractory=2*ms, method='euler',
                    namespace=dict(EL=EL, R=R, tau=tau, tau_syn=tau_syn))

# Drive pres with Poisson source so we get spikes
PG = PoissonGroup(Npre, rates=20*Hz)

# Poisson -> Gpre (to make pres spike)
Sdrive = Synapses(PG, Gpre, model='w:amp', on_pre='I_syn_post += w', method='euler')
Sdrive.connect(j='i')         # 1:1 drive
Sdrive.w = 200*pA

# Gpre -> Gpost all-to-all
S = Synapses(Gpre, Gpost, model='w:amp', on_pre='I_syn_post += w', method='euler')
S.connect(i=numpy.arange(10), j=numpy.arange(10))
S.w = 100*pA

# Initialize voltages
Gpre.v = EL + (Vt-EL)*0.5
Gpost.v = EL

# Set how long you want to run (baked into the C++ main)
sim_dur = 2*second
run(sim_dur)


outdir = Path('cpp_pairwise_10x10')
device.build(directory=str(outdir), compile=True, run=False, clean=False)

print("Built pairwise project at:", outdir.resolve())
print("Executable is usually named 'main' or 'main.exe' inside that folder.")
