from prep_input_data import PrepInput
from argparse import ArgumentParser
import yaml

def call_inputs():
    '''
    Example usage of the PrepInput class to generate spike trains based on STRF data.
    1. Load configuration from YAML file.
    2. Initialize PrepInput with command-line arguments and configuration.
    3. Generate masker and target locations.
    4. Process input STRF data to generate spike trains.
    5. Print the keys of the generated spike train dictionary to visualize.
    6. Adjust parameters as needed for different scenarios.
    '''
    
    path = "strfMat.mat"

    args = ArgumentParser()
    args.add_argument('--chans', type=int, default=1, help='Number of channels')
    args.add_argument('--trials', type=int, default=10, help='Number of trials')
    args.add_argument('--padToTime', type=int, default=3500, help='Time to pad spike trains to (ms)')
    args.add_argument('--sigma', type=int, default=300, help='Standard deviation for Gaussian tuning curves')    
    args.add_argument('--dt', type=float, default=0.1, help='Time step in ms')
    args.add_argument('--FR', type=float, default=8.0, help='Firing rate of neurons in Hz')
    args.add_argument('--std', type=float, default=0.0, help='Standard deviation of the firing rate.')
    args.add_argument('--simlen', type=int, default=35000, help='Number of time steps (default = 35000).')
    args.add_argument('--strfGain', type=int, default=0.1, help='Flat multiplier applied to the STRF firing rates.')
    args.add_argument('--target_dir', type=str, default='../resampled-stimuli/target', help='directory containing target stimuli')
    args.add_argument('--masker_dir', type=str, default='../resampled-stimuli/masker', help='directory containing masker stimuli')
        
    parsed_args = args.parse_args()
    

    # the yaml cofig file is constant and does not need to be changed for different runs
    yaml_path = '../config/config.yaml'
    config = yaml.safe_load(open(yaml_path, 'r'))
    sub_config = config['input_spike_train']
    
    prep_input = PrepInput(parsed_args, sub_config)
    
    #masker_locs, target_locs = prep_input.make_grid_target_masker_locs()
    masker_locs = [None]
    target_locs = [0]
    list_locs = list(zip(masker_locs, target_locs))

    spks = prep_input.process_input(
                strf_path=path, 
                list_locs=list_locs, 
                on_neuron=True, 
                off_neuron=True,)

    return spks
    
    #print("Generated spike train keys:", spks.keys())