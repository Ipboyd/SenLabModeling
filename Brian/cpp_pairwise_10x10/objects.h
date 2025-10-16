
#ifndef _BRIAN_OBJECTS_H
#define _BRIAN_OBJECTS_H

#include "synapses_classes.h"
#include "brianlib/clocks.h"
#include "brianlib/dynamic_array.h"
#include "brianlib/stdint_compat.h"
#include "network.h"
#include<random>
#include<vector>


namespace brian {

extern std::string results_dir;

class RandomGenerator {
    private:
        std::mt19937 gen;
        double stored_gauss;
        bool has_stored_gauss = false;
    public:
        RandomGenerator() {
            seed();
        }
        void seed() {
            std::random_device rd;
            gen.seed(rd());
            has_stored_gauss = false;
        }
        void seed(unsigned long seed) {
            gen.seed(seed);
            has_stored_gauss = false;
        }
        double rand() {
            /* shifts : 67108864 = 0x4000000, 9007199254740992 = 0x20000000000000 */
            const long a = gen() >> 5;
            const long b = gen() >> 6;
            return (a * 67108864.0 + b) / 9007199254740992.0;
        }

        double randn() {
            if (has_stored_gauss) {
                const double tmp = stored_gauss;
                has_stored_gauss = false;
                return tmp;
            }
            else {
                double f, x1, x2, r2;

                do {
                    x1 = 2.0*rand() - 1.0;
                    x2 = 2.0*rand() - 1.0;
                    r2 = x1*x1 + x2*x2;
                }
                while (r2 >= 1.0 || r2 == 0.0);

                /* Box-Muller transform */
                f = sqrt(-2.0*log(r2)/r2);
                /* Keep for next call */
                stored_gauss = f*x1;
                has_stored_gauss = true;
                return f*x2;
            }
        }
};

// In OpenMP we need one state per thread
extern std::vector< RandomGenerator > _random_generators;

//////////////// clocks ///////////////////
extern Clock defaultclock;

//////////////// networks /////////////////
extern Network magicnetwork;



void set_variable_by_name(std::string, std::string);

//////////////// dynamic arrays ///////////
extern std::vector<int32_t> _dynamic_array_synapses_1__synaptic_post;
extern std::vector<int32_t> _dynamic_array_synapses_1__synaptic_pre;
extern std::vector<double> _dynamic_array_synapses_1_delay;
extern std::vector<int32_t> _dynamic_array_synapses_1_N_incoming;
extern std::vector<int32_t> _dynamic_array_synapses_1_N_outgoing;
extern std::vector<double> _dynamic_array_synapses_1_w;
extern std::vector<int32_t> _dynamic_array_synapses__synaptic_post;
extern std::vector<int32_t> _dynamic_array_synapses__synaptic_pre;
extern std::vector<double> _dynamic_array_synapses_delay;
extern std::vector<int32_t> _dynamic_array_synapses_N_incoming;
extern std::vector<int32_t> _dynamic_array_synapses_N_outgoing;
extern std::vector<double> _dynamic_array_synapses_w;

//////////////// arrays ///////////////////
extern double *_array_defaultclock_dt;
extern const int _num__array_defaultclock_dt;
extern double *_array_defaultclock_t;
extern const int _num__array_defaultclock_t;
extern int64_t *_array_defaultclock_timestep;
extern const int _num__array_defaultclock_timestep;
extern int32_t *_array_neurongroup_1__spikespace;
extern const int _num__array_neurongroup_1__spikespace;
extern int32_t *_array_neurongroup_1_i;
extern const int _num__array_neurongroup_1_i;
extern double *_array_neurongroup_1_I_syn;
extern const int _num__array_neurongroup_1_I_syn;
extern double *_array_neurongroup_1_lastspike;
extern const int _num__array_neurongroup_1_lastspike;
extern char *_array_neurongroup_1_not_refractory;
extern const int _num__array_neurongroup_1_not_refractory;
extern double *_array_neurongroup_1_v;
extern const int _num__array_neurongroup_1_v;
extern int32_t *_array_neurongroup__spikespace;
extern const int _num__array_neurongroup__spikespace;
extern int32_t *_array_neurongroup_i;
extern const int _num__array_neurongroup_i;
extern double *_array_neurongroup_I_syn;
extern const int _num__array_neurongroup_I_syn;
extern double *_array_neurongroup_lastspike;
extern const int _num__array_neurongroup_lastspike;
extern char *_array_neurongroup_not_refractory;
extern const int _num__array_neurongroup_not_refractory;
extern double *_array_neurongroup_v;
extern const int _num__array_neurongroup_v;
extern int32_t *_array_poissongroup__spikespace;
extern const int _num__array_poissongroup__spikespace;
extern int32_t *_array_poissongroup_i;
extern const int _num__array_poissongroup_i;
extern double *_array_poissongroup_rates;
extern const int _num__array_poissongroup_rates;
extern int32_t *_array_synapses_1_N;
extern const int _num__array_synapses_1_N;
extern int32_t *_array_synapses_1_sources;
extern const int _num__array_synapses_1_sources;
extern int32_t *_array_synapses_1_targets;
extern const int _num__array_synapses_1_targets;
extern int32_t *_array_synapses_N;
extern const int _num__array_synapses_N;

//////////////// dynamic arrays 2d /////////

/////////////// static arrays /////////////
extern int32_t *_static_array__array_synapses_1_sources;
extern const int _num__static_array__array_synapses_1_sources;
extern int32_t *_static_array__array_synapses_1_targets;
extern const int _num__static_array__array_synapses_1_targets;

//////////////// synapses /////////////////
// synapses
extern SynapticPathway synapses_pre;
// synapses_1
extern SynapticPathway synapses_1_pre;

// Profiling information for each code object
}

void _init_arrays();
void _load_arrays();
void _write_arrays();
void _dealloc_arrays();

#endif


