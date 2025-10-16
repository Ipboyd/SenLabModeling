

#include "objects.h"
#include "synapses_classes.h"
#include "brianlib/clocks.h"
#include "brianlib/dynamic_array.h"
#include "brianlib/stdint_compat.h"
#include "network.h"
#include<random>
#include<vector>
#include<iostream>
#include<fstream>
#include<map>
#include<tuple>
#include<cstdlib>
#include<string>

namespace brian {

std::string results_dir = "results/";  // can be overwritten by --results_dir command line arg

// For multhreading, we need one generator for each thread. We also create a distribution for
// each thread, even though this is not strictly necessary for the uniform distribution, as
// the distribution is stateless.
std::vector< RandomGenerator > _random_generators;

//////////////// networks /////////////////
Network magicnetwork;

void set_variable_from_value(std::string varname, char* var_pointer, size_t size, char value) {
    #ifdef DEBUG
    std::cout << "Setting '" << varname << "' to " << (value == 1 ? "True" : "False") << std::endl;
    #endif
    std::fill(var_pointer, var_pointer+size, value);
}

template<class T> void set_variable_from_value(std::string varname, T* var_pointer, size_t size, T value) {
    #ifdef DEBUG
    std::cout << "Setting '" << varname << "' to " << value << std::endl;
    #endif
    std::fill(var_pointer, var_pointer+size, value);
}

template<class T> void set_variable_from_file(std::string varname, T* var_pointer, size_t data_size, std::string filename) {
    ifstream f;
    streampos size;
    #ifdef DEBUG
    std::cout << "Setting '" << varname << "' from file '" << filename << "'" << std::endl;
    #endif
    f.open(filename, ios::in | ios::binary | ios::ate);
    size = f.tellg();
    if (size != data_size) {
        std::cerr << "Error reading '" << filename << "': file size " << size << " does not match expected size " << data_size << std::endl;
        return;
    }
    f.seekg(0, ios::beg);
    if (f.is_open())
        f.read(reinterpret_cast<char *>(var_pointer), data_size);
    else
        std::cerr << "Could not read '" << filename << "'" << std::endl;
    if (f.fail())
        std::cerr << "Error reading '" << filename << "'" << std::endl;
}

//////////////// set arrays by name ///////
void set_variable_by_name(std::string name, std::string s_value) {
    size_t var_size;
    size_t data_size;
    // C-style or Python-style capitalization is allowed for boolean values
    if (s_value == "true" || s_value == "True")
        s_value = "1";
    else if (s_value == "false" || s_value == "False")
        s_value = "0";
    // non-dynamic arrays
    if (name == "neurongroup_1._spikespace") {
        var_size = 11;
        data_size = 11*sizeof(int32_t);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<int32_t>(name, _array_neurongroup_1__spikespace, var_size, (int32_t)atoi(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_neurongroup_1__spikespace, data_size, s_value);
        }
        return;
    }
    if (name == "neurongroup_1.I_syn") {
        var_size = 10;
        data_size = 10*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, _array_neurongroup_1_I_syn, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_neurongroup_1_I_syn, data_size, s_value);
        }
        return;
    }
    if (name == "neurongroup_1.lastspike") {
        var_size = 10;
        data_size = 10*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, _array_neurongroup_1_lastspike, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_neurongroup_1_lastspike, data_size, s_value);
        }
        return;
    }
    if (name == "neurongroup_1.not_refractory") {
        var_size = 10;
        data_size = 10*sizeof(char);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value(name, _array_neurongroup_1_not_refractory, var_size, (char)atoi(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_neurongroup_1_not_refractory, data_size, s_value);
        }
        return;
    }
    if (name == "neurongroup_1.v") {
        var_size = 10;
        data_size = 10*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, _array_neurongroup_1_v, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_neurongroup_1_v, data_size, s_value);
        }
        return;
    }
    if (name == "neurongroup._spikespace") {
        var_size = 11;
        data_size = 11*sizeof(int32_t);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<int32_t>(name, _array_neurongroup__spikespace, var_size, (int32_t)atoi(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_neurongroup__spikespace, data_size, s_value);
        }
        return;
    }
    if (name == "neurongroup.I_syn") {
        var_size = 10;
        data_size = 10*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, _array_neurongroup_I_syn, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_neurongroup_I_syn, data_size, s_value);
        }
        return;
    }
    if (name == "neurongroup.lastspike") {
        var_size = 10;
        data_size = 10*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, _array_neurongroup_lastspike, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_neurongroup_lastspike, data_size, s_value);
        }
        return;
    }
    if (name == "neurongroup.not_refractory") {
        var_size = 10;
        data_size = 10*sizeof(char);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value(name, _array_neurongroup_not_refractory, var_size, (char)atoi(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_neurongroup_not_refractory, data_size, s_value);
        }
        return;
    }
    if (name == "neurongroup.v") {
        var_size = 10;
        data_size = 10*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, _array_neurongroup_v, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_neurongroup_v, data_size, s_value);
        }
        return;
    }
    if (name == "poissongroup._spikespace") {
        var_size = 11;
        data_size = 11*sizeof(int32_t);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<int32_t>(name, _array_poissongroup__spikespace, var_size, (int32_t)atoi(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_poissongroup__spikespace, data_size, s_value);
        }
        return;
    }
    if (name == "poissongroup.rates") {
        var_size = 10;
        data_size = 10*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, _array_poissongroup_rates, var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, _array_poissongroup_rates, data_size, s_value);
        }
        return;
    }
    // dynamic arrays (1d)
    if (name == "synapses_1.delay") {
        var_size = _dynamic_array_synapses_1_delay.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_1_delay[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_1_delay[0], data_size, s_value);
        }
        return;
    }
    if (name == "synapses_1.w") {
        var_size = _dynamic_array_synapses_1_w.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_1_w[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_1_w[0], data_size, s_value);
        }
        return;
    }
    if (name == "synapses.delay") {
        var_size = _dynamic_array_synapses_delay.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_delay[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_delay[0], data_size, s_value);
        }
        return;
    }
    if (name == "synapses.w") {
        var_size = _dynamic_array_synapses_w.size();
        data_size = var_size*sizeof(double);
        if (s_value[0] == '-' || (s_value[0] >= '0' && s_value[0] <= '9')) {
            // set from single value
            set_variable_from_value<double>(name, &_dynamic_array_synapses_w[0], var_size, (double)atof(s_value.c_str()));

        } else {
            // set from file
            set_variable_from_file(name, &_dynamic_array_synapses_w[0], data_size, s_value);
        }
        return;
    }
    std::cerr << "Cannot set unknown variable '" << name << "'." << std::endl;
    exit(1);
}
//////////////// arrays ///////////////////
double * _array_defaultclock_dt;
const int _num__array_defaultclock_dt = 1;
double * _array_defaultclock_t;
const int _num__array_defaultclock_t = 1;
int64_t * _array_defaultclock_timestep;
const int _num__array_defaultclock_timestep = 1;
int32_t * _array_neurongroup_1__spikespace;
const int _num__array_neurongroup_1__spikespace = 11;
int32_t * _array_neurongroup_1_i;
const int _num__array_neurongroup_1_i = 10;
double * _array_neurongroup_1_I_syn;
const int _num__array_neurongroup_1_I_syn = 10;
double * _array_neurongroup_1_lastspike;
const int _num__array_neurongroup_1_lastspike = 10;
char * _array_neurongroup_1_not_refractory;
const int _num__array_neurongroup_1_not_refractory = 10;
double * _array_neurongroup_1_v;
const int _num__array_neurongroup_1_v = 10;
int32_t * _array_neurongroup__spikespace;
const int _num__array_neurongroup__spikespace = 11;
int32_t * _array_neurongroup_i;
const int _num__array_neurongroup_i = 10;
double * _array_neurongroup_I_syn;
const int _num__array_neurongroup_I_syn = 10;
double * _array_neurongroup_lastspike;
const int _num__array_neurongroup_lastspike = 10;
char * _array_neurongroup_not_refractory;
const int _num__array_neurongroup_not_refractory = 10;
double * _array_neurongroup_v;
const int _num__array_neurongroup_v = 10;
int32_t * _array_poissongroup__spikespace;
const int _num__array_poissongroup__spikespace = 11;
int32_t * _array_poissongroup_i;
const int _num__array_poissongroup_i = 10;
double * _array_poissongroup_rates;
const int _num__array_poissongroup_rates = 10;
int32_t * _array_synapses_1_N;
const int _num__array_synapses_1_N = 1;
int32_t * _array_synapses_1_sources;
const int _num__array_synapses_1_sources = 10;
int32_t * _array_synapses_1_targets;
const int _num__array_synapses_1_targets = 10;
int32_t * _array_synapses_N;
const int _num__array_synapses_N = 1;

//////////////// dynamic arrays 1d /////////
std::vector<int32_t> _dynamic_array_synapses_1__synaptic_post;
std::vector<int32_t> _dynamic_array_synapses_1__synaptic_pre;
std::vector<double> _dynamic_array_synapses_1_delay;
std::vector<int32_t> _dynamic_array_synapses_1_N_incoming;
std::vector<int32_t> _dynamic_array_synapses_1_N_outgoing;
std::vector<double> _dynamic_array_synapses_1_w;
std::vector<int32_t> _dynamic_array_synapses__synaptic_post;
std::vector<int32_t> _dynamic_array_synapses__synaptic_pre;
std::vector<double> _dynamic_array_synapses_delay;
std::vector<int32_t> _dynamic_array_synapses_N_incoming;
std::vector<int32_t> _dynamic_array_synapses_N_outgoing;
std::vector<double> _dynamic_array_synapses_w;

//////////////// dynamic arrays 2d /////////

/////////////// static arrays /////////////
int32_t * _static_array__array_synapses_1_sources;
const int _num__static_array__array_synapses_1_sources = 10;
int32_t * _static_array__array_synapses_1_targets;
const int _num__static_array__array_synapses_1_targets = 10;

//////////////// synapses /////////////////
// synapses
SynapticPathway synapses_pre(
    _dynamic_array_synapses__synaptic_pre,
    0, 10);
// synapses_1
SynapticPathway synapses_1_pre(
    _dynamic_array_synapses_1__synaptic_pre,
    0, 10);

//////////////// clocks ///////////////////
Clock defaultclock;  // attributes will be set in run.cpp

// Profiling information for each code object
}

void _init_arrays()
{
    using namespace brian;

    // Arrays initialized to 0
    _array_defaultclock_dt = new double[1];
    
    for(int i=0; i<1; i++) _array_defaultclock_dt[i] = 0;

    _array_defaultclock_t = new double[1];
    
    for(int i=0; i<1; i++) _array_defaultclock_t[i] = 0;

    _array_defaultclock_timestep = new int64_t[1];
    
    for(int i=0; i<1; i++) _array_defaultclock_timestep[i] = 0;

    _array_neurongroup_1__spikespace = new int32_t[11];
    
    for(int i=0; i<11; i++) _array_neurongroup_1__spikespace[i] = 0;

    _array_neurongroup_1_i = new int32_t[10];
    
    for(int i=0; i<10; i++) _array_neurongroup_1_i[i] = 0;

    _array_neurongroup_1_I_syn = new double[10];
    
    for(int i=0; i<10; i++) _array_neurongroup_1_I_syn[i] = 0;

    _array_neurongroup_1_lastspike = new double[10];
    
    for(int i=0; i<10; i++) _array_neurongroup_1_lastspike[i] = 0;

    _array_neurongroup_1_not_refractory = new char[10];
    
    for(int i=0; i<10; i++) _array_neurongroup_1_not_refractory[i] = 0;

    _array_neurongroup_1_v = new double[10];
    
    for(int i=0; i<10; i++) _array_neurongroup_1_v[i] = 0;

    _array_neurongroup__spikespace = new int32_t[11];
    
    for(int i=0; i<11; i++) _array_neurongroup__spikespace[i] = 0;

    _array_neurongroup_i = new int32_t[10];
    
    for(int i=0; i<10; i++) _array_neurongroup_i[i] = 0;

    _array_neurongroup_I_syn = new double[10];
    
    for(int i=0; i<10; i++) _array_neurongroup_I_syn[i] = 0;

    _array_neurongroup_lastspike = new double[10];
    
    for(int i=0; i<10; i++) _array_neurongroup_lastspike[i] = 0;

    _array_neurongroup_not_refractory = new char[10];
    
    for(int i=0; i<10; i++) _array_neurongroup_not_refractory[i] = 0;

    _array_neurongroup_v = new double[10];
    
    for(int i=0; i<10; i++) _array_neurongroup_v[i] = 0;

    _array_poissongroup__spikespace = new int32_t[11];
    
    for(int i=0; i<11; i++) _array_poissongroup__spikespace[i] = 0;

    _array_poissongroup_i = new int32_t[10];
    
    for(int i=0; i<10; i++) _array_poissongroup_i[i] = 0;

    _array_poissongroup_rates = new double[10];
    
    for(int i=0; i<10; i++) _array_poissongroup_rates[i] = 0;

    _array_synapses_1_N = new int32_t[1];
    
    for(int i=0; i<1; i++) _array_synapses_1_N[i] = 0;

    _array_synapses_1_sources = new int32_t[10];
    
    for(int i=0; i<10; i++) _array_synapses_1_sources[i] = 0;

    _array_synapses_1_targets = new int32_t[10];
    
    for(int i=0; i<10; i++) _array_synapses_1_targets[i] = 0;

    _array_synapses_N = new int32_t[1];
    
    for(int i=0; i<1; i++) _array_synapses_N[i] = 0;


    // Arrays initialized to an "arange"
    _array_neurongroup_1_i = new int32_t[10];
    
    for(int i=0; i<10; i++) _array_neurongroup_1_i[i] = 0 + i;

    _array_neurongroup_i = new int32_t[10];
    
    for(int i=0; i<10; i++) _array_neurongroup_i[i] = 0 + i;

    _array_poissongroup_i = new int32_t[10];
    
    for(int i=0; i<10; i++) _array_poissongroup_i[i] = 0 + i;


    // static arrays
    _static_array__array_synapses_1_sources = new int32_t[10];
    _static_array__array_synapses_1_targets = new int32_t[10];

    // Random number generator states
    std::random_device rd;
    for (int i=0; i<1; i++)
        _random_generators.push_back(RandomGenerator());
}

void _load_arrays()
{
    using namespace brian;

    ifstream f_static_array__array_synapses_1_sources;
    f_static_array__array_synapses_1_sources.open("static_arrays/_static_array__array_synapses_1_sources", ios::in | ios::binary);
    if(f_static_array__array_synapses_1_sources.is_open())
    {
        f_static_array__array_synapses_1_sources.read(reinterpret_cast<char*>(_static_array__array_synapses_1_sources), 10*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_1_sources." << endl;
    }
    ifstream f_static_array__array_synapses_1_targets;
    f_static_array__array_synapses_1_targets.open("static_arrays/_static_array__array_synapses_1_targets", ios::in | ios::binary);
    if(f_static_array__array_synapses_1_targets.is_open())
    {
        f_static_array__array_synapses_1_targets.read(reinterpret_cast<char*>(_static_array__array_synapses_1_targets), 10*sizeof(int32_t));
    } else
    {
        std::cout << "Error opening static array _static_array__array_synapses_1_targets." << endl;
    }
}

void _write_arrays()
{
    using namespace brian;

    ofstream outfile__array_defaultclock_dt;
    outfile__array_defaultclock_dt.open(results_dir + "_array_defaultclock_dt_1978099143", ios::binary | ios::out);
    if(outfile__array_defaultclock_dt.is_open())
    {
        outfile__array_defaultclock_dt.write(reinterpret_cast<char*>(_array_defaultclock_dt), 1*sizeof(_array_defaultclock_dt[0]));
        outfile__array_defaultclock_dt.close();
    } else
    {
        std::cout << "Error writing output file for _array_defaultclock_dt." << endl;
    }
    ofstream outfile__array_defaultclock_t;
    outfile__array_defaultclock_t.open(results_dir + "_array_defaultclock_t_2669362164", ios::binary | ios::out);
    if(outfile__array_defaultclock_t.is_open())
    {
        outfile__array_defaultclock_t.write(reinterpret_cast<char*>(_array_defaultclock_t), 1*sizeof(_array_defaultclock_t[0]));
        outfile__array_defaultclock_t.close();
    } else
    {
        std::cout << "Error writing output file for _array_defaultclock_t." << endl;
    }
    ofstream outfile__array_defaultclock_timestep;
    outfile__array_defaultclock_timestep.open(results_dir + "_array_defaultclock_timestep_144223508", ios::binary | ios::out);
    if(outfile__array_defaultclock_timestep.is_open())
    {
        outfile__array_defaultclock_timestep.write(reinterpret_cast<char*>(_array_defaultclock_timestep), 1*sizeof(_array_defaultclock_timestep[0]));
        outfile__array_defaultclock_timestep.close();
    } else
    {
        std::cout << "Error writing output file for _array_defaultclock_timestep." << endl;
    }
    ofstream outfile__array_neurongroup_1__spikespace;
    outfile__array_neurongroup_1__spikespace.open(results_dir + "_array_neurongroup_1__spikespace_3155027917", ios::binary | ios::out);
    if(outfile__array_neurongroup_1__spikespace.is_open())
    {
        outfile__array_neurongroup_1__spikespace.write(reinterpret_cast<char*>(_array_neurongroup_1__spikespace), 11*sizeof(_array_neurongroup_1__spikespace[0]));
        outfile__array_neurongroup_1__spikespace.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1__spikespace." << endl;
    }
    ofstream outfile__array_neurongroup_1_i;
    outfile__array_neurongroup_1_i.open(results_dir + "_array_neurongroup_1_i_3674354357", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_i.is_open())
    {
        outfile__array_neurongroup_1_i.write(reinterpret_cast<char*>(_array_neurongroup_1_i), 10*sizeof(_array_neurongroup_1_i[0]));
        outfile__array_neurongroup_1_i.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_i." << endl;
    }
    ofstream outfile__array_neurongroup_1_I_syn;
    outfile__array_neurongroup_1_I_syn.open(results_dir + "_array_neurongroup_1_I_syn_1608470925", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_I_syn.is_open())
    {
        outfile__array_neurongroup_1_I_syn.write(reinterpret_cast<char*>(_array_neurongroup_1_I_syn), 10*sizeof(_array_neurongroup_1_I_syn[0]));
        outfile__array_neurongroup_1_I_syn.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_I_syn." << endl;
    }
    ofstream outfile__array_neurongroup_1_lastspike;
    outfile__array_neurongroup_1_lastspike.open(results_dir + "_array_neurongroup_1_lastspike_1163579662", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_lastspike.is_open())
    {
        outfile__array_neurongroup_1_lastspike.write(reinterpret_cast<char*>(_array_neurongroup_1_lastspike), 10*sizeof(_array_neurongroup_1_lastspike[0]));
        outfile__array_neurongroup_1_lastspike.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_lastspike." << endl;
    }
    ofstream outfile__array_neurongroup_1_not_refractory;
    outfile__array_neurongroup_1_not_refractory.open(results_dir + "_array_neurongroup_1_not_refractory_897855399", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_not_refractory.is_open())
    {
        outfile__array_neurongroup_1_not_refractory.write(reinterpret_cast<char*>(_array_neurongroup_1_not_refractory), 10*sizeof(_array_neurongroup_1_not_refractory[0]));
        outfile__array_neurongroup_1_not_refractory.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_not_refractory." << endl;
    }
    ofstream outfile__array_neurongroup_1_v;
    outfile__array_neurongroup_1_v.open(results_dir + "_array_neurongroup_1_v_1443512128", ios::binary | ios::out);
    if(outfile__array_neurongroup_1_v.is_open())
    {
        outfile__array_neurongroup_1_v.write(reinterpret_cast<char*>(_array_neurongroup_1_v), 10*sizeof(_array_neurongroup_1_v[0]));
        outfile__array_neurongroup_1_v.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_1_v." << endl;
    }
    ofstream outfile__array_neurongroup__spikespace;
    outfile__array_neurongroup__spikespace.open(results_dir + "_array_neurongroup__spikespace_3522821529", ios::binary | ios::out);
    if(outfile__array_neurongroup__spikespace.is_open())
    {
        outfile__array_neurongroup__spikespace.write(reinterpret_cast<char*>(_array_neurongroup__spikespace), 11*sizeof(_array_neurongroup__spikespace[0]));
        outfile__array_neurongroup__spikespace.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup__spikespace." << endl;
    }
    ofstream outfile__array_neurongroup_i;
    outfile__array_neurongroup_i.open(results_dir + "_array_neurongroup_i_2649026944", ios::binary | ios::out);
    if(outfile__array_neurongroup_i.is_open())
    {
        outfile__array_neurongroup_i.write(reinterpret_cast<char*>(_array_neurongroup_i), 10*sizeof(_array_neurongroup_i[0]));
        outfile__array_neurongroup_i.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_i." << endl;
    }
    ofstream outfile__array_neurongroup_I_syn;
    outfile__array_neurongroup_I_syn.open(results_dir + "_array_neurongroup_I_syn_2623062472", ios::binary | ios::out);
    if(outfile__array_neurongroup_I_syn.is_open())
    {
        outfile__array_neurongroup_I_syn.write(reinterpret_cast<char*>(_array_neurongroup_I_syn), 10*sizeof(_array_neurongroup_I_syn[0]));
        outfile__array_neurongroup_I_syn.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_I_syn." << endl;
    }
    ofstream outfile__array_neurongroup_lastspike;
    outfile__array_neurongroup_lastspike.open(results_dir + "_array_neurongroup_lastspike_1647074423", ios::binary | ios::out);
    if(outfile__array_neurongroup_lastspike.is_open())
    {
        outfile__array_neurongroup_lastspike.write(reinterpret_cast<char*>(_array_neurongroup_lastspike), 10*sizeof(_array_neurongroup_lastspike[0]));
        outfile__array_neurongroup_lastspike.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_lastspike." << endl;
    }
    ofstream outfile__array_neurongroup_not_refractory;
    outfile__array_neurongroup_not_refractory.open(results_dir + "_array_neurongroup_not_refractory_1422681464", ios::binary | ios::out);
    if(outfile__array_neurongroup_not_refractory.is_open())
    {
        outfile__array_neurongroup_not_refractory.write(reinterpret_cast<char*>(_array_neurongroup_not_refractory), 10*sizeof(_array_neurongroup_not_refractory[0]));
        outfile__array_neurongroup_not_refractory.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_not_refractory." << endl;
    }
    ofstream outfile__array_neurongroup_v;
    outfile__array_neurongroup_v.open(results_dir + "_array_neurongroup_v_283966581", ios::binary | ios::out);
    if(outfile__array_neurongroup_v.is_open())
    {
        outfile__array_neurongroup_v.write(reinterpret_cast<char*>(_array_neurongroup_v), 10*sizeof(_array_neurongroup_v[0]));
        outfile__array_neurongroup_v.close();
    } else
    {
        std::cout << "Error writing output file for _array_neurongroup_v." << endl;
    }
    ofstream outfile__array_poissongroup__spikespace;
    outfile__array_poissongroup__spikespace.open(results_dir + "_array_poissongroup__spikespace_1019000416", ios::binary | ios::out);
    if(outfile__array_poissongroup__spikespace.is_open())
    {
        outfile__array_poissongroup__spikespace.write(reinterpret_cast<char*>(_array_poissongroup__spikespace), 11*sizeof(_array_poissongroup__spikespace[0]));
        outfile__array_poissongroup__spikespace.close();
    } else
    {
        std::cout << "Error writing output file for _array_poissongroup__spikespace." << endl;
    }
    ofstream outfile__array_poissongroup_i;
    outfile__array_poissongroup_i.open(results_dir + "_array_poissongroup_i_1277690444", ios::binary | ios::out);
    if(outfile__array_poissongroup_i.is_open())
    {
        outfile__array_poissongroup_i.write(reinterpret_cast<char*>(_array_poissongroup_i), 10*sizeof(_array_poissongroup_i[0]));
        outfile__array_poissongroup_i.close();
    } else
    {
        std::cout << "Error writing output file for _array_poissongroup_i." << endl;
    }
    ofstream outfile__array_poissongroup_rates;
    outfile__array_poissongroup_rates.open(results_dir + "_array_poissongroup_rates_3353413371", ios::binary | ios::out);
    if(outfile__array_poissongroup_rates.is_open())
    {
        outfile__array_poissongroup_rates.write(reinterpret_cast<char*>(_array_poissongroup_rates), 10*sizeof(_array_poissongroup_rates[0]));
        outfile__array_poissongroup_rates.close();
    } else
    {
        std::cout << "Error writing output file for _array_poissongroup_rates." << endl;
    }
    ofstream outfile__array_synapses_1_N;
    outfile__array_synapses_1_N.open(results_dir + "_array_synapses_1_N_1771729519", ios::binary | ios::out);
    if(outfile__array_synapses_1_N.is_open())
    {
        outfile__array_synapses_1_N.write(reinterpret_cast<char*>(_array_synapses_1_N), 1*sizeof(_array_synapses_1_N[0]));
        outfile__array_synapses_1_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_1_N." << endl;
    }
    ofstream outfile__array_synapses_1_sources;
    outfile__array_synapses_1_sources.open(results_dir + "_array_synapses_1_sources_93121092", ios::binary | ios::out);
    if(outfile__array_synapses_1_sources.is_open())
    {
        outfile__array_synapses_1_sources.write(reinterpret_cast<char*>(_array_synapses_1_sources), 10*sizeof(_array_synapses_1_sources[0]));
        outfile__array_synapses_1_sources.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_1_sources." << endl;
    }
    ofstream outfile__array_synapses_1_targets;
    outfile__array_synapses_1_targets.open(results_dir + "_array_synapses_1_targets_2022871461", ios::binary | ios::out);
    if(outfile__array_synapses_1_targets.is_open())
    {
        outfile__array_synapses_1_targets.write(reinterpret_cast<char*>(_array_synapses_1_targets), 10*sizeof(_array_synapses_1_targets[0]));
        outfile__array_synapses_1_targets.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_1_targets." << endl;
    }
    ofstream outfile__array_synapses_N;
    outfile__array_synapses_N.open(results_dir + "_array_synapses_N_483293785", ios::binary | ios::out);
    if(outfile__array_synapses_N.is_open())
    {
        outfile__array_synapses_N.write(reinterpret_cast<char*>(_array_synapses_N), 1*sizeof(_array_synapses_N[0]));
        outfile__array_synapses_N.close();
    } else
    {
        std::cout << "Error writing output file for _array_synapses_N." << endl;
    }

    ofstream outfile__dynamic_array_synapses_1__synaptic_post;
    outfile__dynamic_array_synapses_1__synaptic_post.open(results_dir + "_dynamic_array_synapses_1__synaptic_post_1999337987", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_1__synaptic_post.is_open())
    {
        if (! _dynamic_array_synapses_1__synaptic_post.empty() )
        {
            outfile__dynamic_array_synapses_1__synaptic_post.write(reinterpret_cast<char*>(&_dynamic_array_synapses_1__synaptic_post[0]), _dynamic_array_synapses_1__synaptic_post.size()*sizeof(_dynamic_array_synapses_1__synaptic_post[0]));
            outfile__dynamic_array_synapses_1__synaptic_post.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_1__synaptic_post." << endl;
    }
    ofstream outfile__dynamic_array_synapses_1__synaptic_pre;
    outfile__dynamic_array_synapses_1__synaptic_pre.open(results_dir + "_dynamic_array_synapses_1__synaptic_pre_681065502", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_1__synaptic_pre.is_open())
    {
        if (! _dynamic_array_synapses_1__synaptic_pre.empty() )
        {
            outfile__dynamic_array_synapses_1__synaptic_pre.write(reinterpret_cast<char*>(&_dynamic_array_synapses_1__synaptic_pre[0]), _dynamic_array_synapses_1__synaptic_pre.size()*sizeof(_dynamic_array_synapses_1__synaptic_pre[0]));
            outfile__dynamic_array_synapses_1__synaptic_pre.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_1__synaptic_pre." << endl;
    }
    ofstream outfile__dynamic_array_synapses_1_delay;
    outfile__dynamic_array_synapses_1_delay.open(results_dir + "_dynamic_array_synapses_1_delay_2373823482", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_1_delay.is_open())
    {
        if (! _dynamic_array_synapses_1_delay.empty() )
        {
            outfile__dynamic_array_synapses_1_delay.write(reinterpret_cast<char*>(&_dynamic_array_synapses_1_delay[0]), _dynamic_array_synapses_1_delay.size()*sizeof(_dynamic_array_synapses_1_delay[0]));
            outfile__dynamic_array_synapses_1_delay.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_1_delay." << endl;
    }
    ofstream outfile__dynamic_array_synapses_1_N_incoming;
    outfile__dynamic_array_synapses_1_N_incoming.open(results_dir + "_dynamic_array_synapses_1_N_incoming_3469555706", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_1_N_incoming.is_open())
    {
        if (! _dynamic_array_synapses_1_N_incoming.empty() )
        {
            outfile__dynamic_array_synapses_1_N_incoming.write(reinterpret_cast<char*>(&_dynamic_array_synapses_1_N_incoming[0]), _dynamic_array_synapses_1_N_incoming.size()*sizeof(_dynamic_array_synapses_1_N_incoming[0]));
            outfile__dynamic_array_synapses_1_N_incoming.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_1_N_incoming." << endl;
    }
    ofstream outfile__dynamic_array_synapses_1_N_outgoing;
    outfile__dynamic_array_synapses_1_N_outgoing.open(results_dir + "_dynamic_array_synapses_1_N_outgoing_3922806560", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_1_N_outgoing.is_open())
    {
        if (! _dynamic_array_synapses_1_N_outgoing.empty() )
        {
            outfile__dynamic_array_synapses_1_N_outgoing.write(reinterpret_cast<char*>(&_dynamic_array_synapses_1_N_outgoing[0]), _dynamic_array_synapses_1_N_outgoing.size()*sizeof(_dynamic_array_synapses_1_N_outgoing[0]));
            outfile__dynamic_array_synapses_1_N_outgoing.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_1_N_outgoing." << endl;
    }
    ofstream outfile__dynamic_array_synapses_1_w;
    outfile__dynamic_array_synapses_1_w.open(results_dir + "_dynamic_array_synapses_1_w_1857285062", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_1_w.is_open())
    {
        if (! _dynamic_array_synapses_1_w.empty() )
        {
            outfile__dynamic_array_synapses_1_w.write(reinterpret_cast<char*>(&_dynamic_array_synapses_1_w[0]), _dynamic_array_synapses_1_w.size()*sizeof(_dynamic_array_synapses_1_w[0]));
            outfile__dynamic_array_synapses_1_w.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_1_w." << endl;
    }
    ofstream outfile__dynamic_array_synapses__synaptic_post;
    outfile__dynamic_array_synapses__synaptic_post.open(results_dir + "_dynamic_array_synapses__synaptic_post_1801389495", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses__synaptic_post.is_open())
    {
        if (! _dynamic_array_synapses__synaptic_post.empty() )
        {
            outfile__dynamic_array_synapses__synaptic_post.write(reinterpret_cast<char*>(&_dynamic_array_synapses__synaptic_post[0]), _dynamic_array_synapses__synaptic_post.size()*sizeof(_dynamic_array_synapses__synaptic_post[0]));
            outfile__dynamic_array_synapses__synaptic_post.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses__synaptic_post." << endl;
    }
    ofstream outfile__dynamic_array_synapses__synaptic_pre;
    outfile__dynamic_array_synapses__synaptic_pre.open(results_dir + "_dynamic_array_synapses__synaptic_pre_814148175", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses__synaptic_pre.is_open())
    {
        if (! _dynamic_array_synapses__synaptic_pre.empty() )
        {
            outfile__dynamic_array_synapses__synaptic_pre.write(reinterpret_cast<char*>(&_dynamic_array_synapses__synaptic_pre[0]), _dynamic_array_synapses__synaptic_pre.size()*sizeof(_dynamic_array_synapses__synaptic_pre[0]));
            outfile__dynamic_array_synapses__synaptic_pre.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses__synaptic_pre." << endl;
    }
    ofstream outfile__dynamic_array_synapses_delay;
    outfile__dynamic_array_synapses_delay.open(results_dir + "_dynamic_array_synapses_delay_3246960869", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_delay.is_open())
    {
        if (! _dynamic_array_synapses_delay.empty() )
        {
            outfile__dynamic_array_synapses_delay.write(reinterpret_cast<char*>(&_dynamic_array_synapses_delay[0]), _dynamic_array_synapses_delay.size()*sizeof(_dynamic_array_synapses_delay[0]));
            outfile__dynamic_array_synapses_delay.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_delay." << endl;
    }
    ofstream outfile__dynamic_array_synapses_N_incoming;
    outfile__dynamic_array_synapses_N_incoming.open(results_dir + "_dynamic_array_synapses_N_incoming_1151751685", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_N_incoming.is_open())
    {
        if (! _dynamic_array_synapses_N_incoming.empty() )
        {
            outfile__dynamic_array_synapses_N_incoming.write(reinterpret_cast<char*>(&_dynamic_array_synapses_N_incoming[0]), _dynamic_array_synapses_N_incoming.size()*sizeof(_dynamic_array_synapses_N_incoming[0]));
            outfile__dynamic_array_synapses_N_incoming.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_N_incoming." << endl;
    }
    ofstream outfile__dynamic_array_synapses_N_outgoing;
    outfile__dynamic_array_synapses_N_outgoing.open(results_dir + "_dynamic_array_synapses_N_outgoing_1673144031", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_N_outgoing.is_open())
    {
        if (! _dynamic_array_synapses_N_outgoing.empty() )
        {
            outfile__dynamic_array_synapses_N_outgoing.write(reinterpret_cast<char*>(&_dynamic_array_synapses_N_outgoing[0]), _dynamic_array_synapses_N_outgoing.size()*sizeof(_dynamic_array_synapses_N_outgoing[0]));
            outfile__dynamic_array_synapses_N_outgoing.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_N_outgoing." << endl;
    }
    ofstream outfile__dynamic_array_synapses_w;
    outfile__dynamic_array_synapses_w.open(results_dir + "_dynamic_array_synapses_w_441891901", ios::binary | ios::out);
    if(outfile__dynamic_array_synapses_w.is_open())
    {
        if (! _dynamic_array_synapses_w.empty() )
        {
            outfile__dynamic_array_synapses_w.write(reinterpret_cast<char*>(&_dynamic_array_synapses_w[0]), _dynamic_array_synapses_w.size()*sizeof(_dynamic_array_synapses_w[0]));
            outfile__dynamic_array_synapses_w.close();
        }
    } else
    {
        std::cout << "Error writing output file for _dynamic_array_synapses_w." << endl;
    }

    // Write last run info to disk
    ofstream outfile_last_run_info;
    outfile_last_run_info.open(results_dir + "last_run_info.txt", ios::out);
    if(outfile_last_run_info.is_open())
    {
        outfile_last_run_info << (Network::_last_run_time) << " " << (Network::_last_run_completed_fraction) << std::endl;
        outfile_last_run_info.close();
    } else
    {
        std::cout << "Error writing last run info to file." << std::endl;
    }
}

void _dealloc_arrays()
{
    using namespace brian;


    // static arrays
    if(_static_array__array_synapses_1_sources!=0)
    {
        delete [] _static_array__array_synapses_1_sources;
        _static_array__array_synapses_1_sources = 0;
    }
    if(_static_array__array_synapses_1_targets!=0)
    {
        delete [] _static_array__array_synapses_1_targets;
        _static_array__array_synapses_1_targets = 0;
    }
}

