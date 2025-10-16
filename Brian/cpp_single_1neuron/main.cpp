#include <stdlib.h>
#include "objects.h"
#include <csignal>
#include <ctime>
#include <time.h>

#include "run.h"
#include "brianlib/common_math.h"

#include "code_objects/neurongroup_1_spike_resetter_codeobject.h"
#include "code_objects/neurongroup_1_spike_thresholder_codeobject.h"
#include "code_objects/after_run_neurongroup_1_spike_thresholder_codeobject.h"
#include "code_objects/neurongroup_1_stateupdater_codeobject.h"
#include "code_objects/neurongroup_spike_resetter_codeobject.h"
#include "code_objects/neurongroup_spike_thresholder_codeobject.h"
#include "code_objects/after_run_neurongroup_spike_thresholder_codeobject.h"
#include "code_objects/neurongroup_stateupdater_codeobject.h"
#include "code_objects/poissongroup_spike_thresholder_codeobject.h"
#include "code_objects/after_run_poissongroup_spike_thresholder_codeobject.h"
#include "code_objects/synapses_1_pre_codeobject.h"
#include "code_objects/synapses_1_pre_push_spikes.h"
#include "code_objects/before_run_synapses_1_pre_push_spikes.h"
#include "code_objects/synapses_1_synapses_create_generator_codeobject.h"
#include "code_objects/synapses_pre_codeobject.h"
#include "code_objects/synapses_pre_push_spikes.h"
#include "code_objects/before_run_synapses_pre_push_spikes.h"
#include "code_objects/synapses_synapses_create_generator_codeobject.h"


#include <iostream>
#include <fstream>
#include <string>




void set_from_command_line(const std::vector<std::string> args)
{
    for (const auto& arg : args) {
		// Split into two parts
		size_t equal_sign = arg.find("=");
		auto name = arg.substr(0, equal_sign);
		auto value = arg.substr(equal_sign + 1, arg.length());
		brian::set_variable_by_name(name, value);
	}
}

void _int_handler(int signal_num) {
	if (Network::_globally_running && !Network::_globally_stopped) {
		Network::_globally_stopped = true;
	} else {
		std::signal(signal_num, SIG_DFL);
		std::raise(signal_num);
	}
}

int main(int argc, char **argv)
{
	std::signal(SIGINT, _int_handler);
	std::random_device _rd;
	std::vector<std::string> args(argv + 1, argv + argc);
	if (args.size() >=2 && args[0] == "--results_dir")
	{
		brian::results_dir = args[1];
		#ifdef DEBUG
		std::cout << "Setting results dir to '" << brian::results_dir << "'" << std::endl;
		#endif
		args.erase(args.begin(), args.begin()+2);
	}
        

	brian_start();
        

	{
		using namespace brian;

		
                
        _array_defaultclock_dt[0] = 0.0001;
        _array_defaultclock_dt[0] = 0.0001;
        _array_defaultclock_dt[0] = 0.0001;
        _array_defaultclock_dt[0] = 0.0001;
        _array_neurongroup_lastspike[0] = - 10000.0;
        _array_neurongroup_not_refractory[0] = true;
        _array_neurongroup_1_lastspike[0] = - 10000.0;
        _array_neurongroup_1_not_refractory[0] = true;
        _array_poissongroup_rates[0] = 20.0;
        _run_synapses_synapses_create_generator_codeobject();
        
                        
                        for(int i=0; i<_dynamic_array_synapses_w.size(); i++)
                        {
                            _dynamic_array_synapses_w[i] = 2e-10;
                        }
                        
        _run_synapses_1_synapses_create_generator_codeobject();
        
                        
                        for(int i=0; i<_dynamic_array_synapses_1_w.size(); i++)
                        {
                            _dynamic_array_synapses_1_w[i] = 1e-10;
                        }
                        
        _array_defaultclock_timestep[0] = 0;
        _array_defaultclock_t[0] = 0.0;
        _before_run_synapses_1_pre_push_spikes();
        _before_run_synapses_pre_push_spikes();
        magicnetwork.clear();
        magicnetwork.add(&defaultclock, _run_neurongroup_1_stateupdater_codeobject);
        magicnetwork.add(&defaultclock, _run_neurongroup_stateupdater_codeobject);
        magicnetwork.add(&defaultclock, _run_neurongroup_1_spike_thresholder_codeobject);
        magicnetwork.add(&defaultclock, _run_neurongroup_spike_thresholder_codeobject);
        magicnetwork.add(&defaultclock, _run_poissongroup_spike_thresholder_codeobject);
        magicnetwork.add(&defaultclock, _run_synapses_1_pre_push_spikes);
        magicnetwork.add(&defaultclock, _run_synapses_1_pre_codeobject);
        magicnetwork.add(&defaultclock, _run_synapses_pre_push_spikes);
        magicnetwork.add(&defaultclock, _run_synapses_pre_codeobject);
        magicnetwork.add(&defaultclock, _run_neurongroup_1_spike_resetter_codeobject);
        magicnetwork.add(&defaultclock, _run_neurongroup_spike_resetter_codeobject);
        set_from_command_line(args);
        magicnetwork.run(2.0, NULL, 10.0);
        _after_run_neurongroup_1_spike_thresholder_codeobject();
        _after_run_neurongroup_spike_thresholder_codeobject();
        _after_run_poissongroup_spike_thresholder_codeobject();
        #ifdef DEBUG
        _debugmsg_synapses_1_pre_codeobject();
        #endif
        
        #ifdef DEBUG
        _debugmsg_synapses_pre_codeobject();
        #endif

	}
        

	brian_end();
        

	return 0;
}