import os
import yaml
import numpy as np
import pickle as pk

from argparse import ArgumentParser

# local imports
from prep_input_data import PrepInput
from gen_strf import GenSTRF


args = ArgumentParser()
args.add_argument('--chans', type=int, default=4, help='Number of channels')
args.add_argument('--trials', type=int, default=10, help='Number of trials')
args.add_argument('--padToTime', type=int, default=3500, help='Time to pad spike trains to (ms)')
args.add_argument('--sigma', type=int, default=300, help='Standard deviation for Gaussian tuning curves')    
args.add_argument('--dt', type=float, default=0.1, help='Time step in ms')
args.add_argument('--FR', type=float, default=8.0, help='Firing rate of neurons in Hz')
args.add_argument('--std', type=float, default=0.0, help='Standard deviation of the firing rate.')
args.add_argument('--simlen', type=int, default=35000, help='Number of time steps (default = 35000).')
args.add_argument('--target_dir', type=str, default='resampled-stimuli/target', help='directory containing target stimuli')
args.add_argument('--masker_dir', type=str, default='resampled-stimuli/masker', help='directory containing masker stimuli')

    
parsed_args = args.parse_args()


# the yaml cofig file is constant and does not need to be changed for different runs
yaml_path = 'config/config.yaml'
config = yaml.safe_load(open(yaml_path, 'r'))
sub_config = config['input_spike_train']

strf_config = config['strf_config']

lst_target_stim = [os.path.join(parsed_args.target_dir, stim_path) for stim_path in os.listdir(parsed_args.target_dir)]
lst_masker_stim = [os.path.join(parsed_args.masker_dir, stim_path) for stim_path in os.listdir(parsed_args.masker_dir)]

gen_strfs = GenSTRF(parsed_args, strf_config, lst_target_stim[0])

fr_target_on = []
fr_target_off = []
for stim_path in lst_target_stim:
        fr_on, fr_off = gen_strfs.process_stimulus(stim_path, strf_config['targetlvl'], strf_config['stimGain'])
        fr_target_on.append(fr_on.transpose())
        fr_target_off.append(fr_off.transpose())
fr_target_on = np.concatenate(fr_target_on, axis=0)
fr_target_off = np.concatenate(fr_target_off, axis=0)
print(f'>>>> Number of Stimuli: {fr_target_on.shape[0]}')

fr_maskers = []
for stim_path in lst_masker_stim:
        fr_on, fr_off = gen_strfs.process_stimulus(stim_path, strf_config['maskerlvl'], strf_config['stimGain'])
        fr_maskers.append([fr_off.transpose()])

fr_maskers = np.concatenate(fr_maskers, axis=0)
print(f'>>>> Number of trials: {fr_maskers.shape[0]}')
    

prep_input = PrepInput(parsed_args, sub_config)

# masker_locs, target_locs = prep_input.make_grid_target_masker_locs()
list_locs = [(None, 0), 
            (None, 1), 
            (None, 2), 
            (None, 3)]

print(config['input_spike_train'])
spks = prep_input.process_input_from_raw_stim(
            fr_target_on=fr_target_on,
            fr_target_off=fr_target_off,
            fr_masker=fr_maskers,
            strfGain=strf_config['strfGain'],
            list_locs=list_locs, 
            on_neuron=True, 
            off_neuron=True,)

print("Generated spike train keys:", spks.keys())

file_path = 'no_masker.pkl'
with open(file_path, 'wb') as file:
    pk.dump(spks, file)