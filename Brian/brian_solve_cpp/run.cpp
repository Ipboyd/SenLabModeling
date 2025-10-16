#include<stdlib.h>
#include "objects.h"
#include<ctime>
#include<random>

#include "code_objects/synapses_synapses_create_array_codeobject.h"


void brian_start()
{
	_init_arrays();
	_load_arrays();
	// Initialize clocks (link timestep and dt to the respective arrays)
}

void brian_end()
{
	_write_arrays();
	_dealloc_arrays();
}


