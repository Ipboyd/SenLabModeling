# build_single_1neuron.py
from brian2 import *
from pathlib import Path

set_device('cpp_standalone', build_on_run=False)
start_scope()

defaultclock.dt = 0.1*ms

eqs = '''
dv/dt     = (-(v-EL) + R*I_syn)/tau : volt (unless refractory)
dI_syn/dt = -I_syn/tau_syn          : amp
'''
EL = -65*mV; Vt = -50*mV; Vr = -60*mV
gL = 10*nS; C = 200*pF
R  = 1/gL
tau = C*R
tau_syn = 5*ms

Npre = 1
Npost = 1

# Pre and post populations
Gpre = NeuronGroup(Npre, eqs, threshold='v>Vt', reset='v=Vr',
                   refractory=2*ms, method='euler',
                   namespace=dict(EL=EL, R=R, tau=tau, tau_syn=tau_syn))
Gpost = NeuronGroup(Npost, eqs, threshold='v>Vt', reset='v=Vr',
                    refractory=2*ms, method='euler',
                    namespace=dict(EL=EL, R=R, tau=tau, tau_syn=tau_syn))

# Drive pres with Poisson source so we get spikes
PG = PoissonGroup(Npre, rates=20*Hz)
# G = NeuronGroup(1, eqs, threshold='v>Vt', reset='v=Vr',
#                 refractory=2*ms, method='euler',
#                 namespace=dict(EL=EL, R=R, tau=tau, tau_syn=tau_syn))
# G.v = EL + (Vt-EL)*0.2

# Give it a bit of Poisson drive so it does something
#PG = PoissonGroup(1, rates=50*Hz)
Sdrive = Synapses(PG, Gpre, model='w:amp', on_pre='I_syn_post += w', method='euler')
Sdrive.connect(j='i')         # 1:1 drive
Sdrive.w = 200*pA

# Gpre -> Gpost all-to-all
S = Synapses(Gpre, Gpost, model='w:amp', on_pre='I_syn_post += w', method='euler')
S.connect(p=1.0)
S.w = 100*pA

sim_dur = 2*second
run(sim_dur)

outdir = Path('cpp_single_1neuron')
device.build(directory=str(outdir), compile=True, run=False, clean=False)

print("Built single-neuron project at:", outdir.resolve())
print("Executable is usually 'main' or 'main.exe'.")
