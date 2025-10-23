def Euler_Compiler(neurons,synapses,projections,options):
    #1. Declare all of the variables
    variable_declaration = declare_vars(neurons,synapses,options)
    #print(variable_declaration)
    #2. Declare the necessary holders
    holder_declaration = declare_holders(neurons,synapses,options)
    #print(holder_declaration)
    #3. Build Euler Loop
    Euler_loop_declaration = declare_loop(options)
    #print(Euler_loop_declaration)
    #4. Declare ODEs
    ODE_declaration = declare_odes(neurons,synapses,projections,options)
    #print(ODE_declaration)
    #5. Declare State Updates
    State_Update_declaration = declare_state_updates(neurons,synapses,options)
    #print(State_Update_declaration)
    #6. Declare conditionals
    Conditionals_declaration = declare_condtionals(neurons,synapses)
    #print(Conditionals_declaration)
    #7. Declare return statement
    Returns_declaration = declare_returns(neurons)

    solve_file_body = variable_declaration+holder_declaration+Euler_loop_declaration+ODE_declaration+State_Update_declaration+Conditionals_declaration+Returns_declaration

    return solve_file_body

def declare_vars(neurons,synapses,options):
    
    all_declares = [neurons,synapses]

    #Set header
    variable_declaration = '\n\n    #Declare Variables\n'
    
    #Loop through all of the neurons,synapses, and options and grab all of the variables
    for k in all_declares:
        for variable in k:
            for j in variable.keys():
                if j != 'name':
                    variable_declaration += f'\n    {j} = {variable[j]}'

    return variable_declaration

def declare_holders(neurons, synapses, options):

    #Set header
    holder_declaration = '\n\n    #Declare Holders\n'

    #Declare holders relevent to each neuron
    for k in neurons:
        #Using inplace operations (only saving current and previous step) for memory
        neuron_name = k["name"]
        #Voltage -- initialized at resting potential E_L
        holder_declaration += f'\n    {neuron_name}_V = np.ones(({options["N_batch"]},{options["N_trials"]},{options["N_channels"]},2)) * np.array([{neuron_name}_E_L,{neuron_name}_E_L])'
        #Adaptation -- initilized at 0
        holder_declaration += f'\n    {neuron_name}_g_ad = np.zeros(({options["N_batch"]},{options["N_trials"]},{options["N_channels"]},2))'
        #tspike -- initilized with sentinel (A sentinel is a "large" number that should minimally effect spiking activity) previous 1e32 --> made -30 because -1e32 is overkill and effects gradients
        #tspike is a circular buffer
        holder_declaration += f'\n    {neuron_name}_tspike = np.ones(({options["N_batch"]},{options["N_trials"]},{options["N_channels"]},5)) * -30'
        #Buffer index -- Holds the index in which the spike will be inserted into tpike per batchxtrialxchannel
        holder_declaration += f'\n    {neuron_name}_buffer_index = np.ones(({options["N_batch"]},{options["N_trials"]},{options["N_channels"]}))'
        #Spike holder -- Holds the output of the network -- only save the outputs to the designated output neurons to save memory
        if k["is_output"] == 1:
            holder_declaration += f'\n    {neuron_name}_spikes_holder = np.zeros(({options["N_batch"]},{options["N_trials"]},{options["N_channels"]},{options["sim_len"]}))'
        #Noise PSC_like terms (Still just within a single neuron)
        if k["is_noise"] == 1:
            holder_declaration += f'\n    {neuron_name}_noise_sn = np.zeros(({options["N_batch"]},{options["N_trials"]},{options["N_channels"]},2))'
            holder_declaration += f'\n    {neuron_name}_noise_xn = np.zeros(({options["N_batch"]},{options["N_trials"]},{options["N_channels"]},2))'

    #Declare holders relevent to each synapse
    for m in synapses:
        #PSCs
        synapse_name = m["name"]
        holder_declaration += f'\n    {synapse_name}_PSC_s = np.zeros(({options["N_batch"]},{options["N_trials"]},{options["N_channels"]},2))'
        holder_declaration += f'\n    {synapse_name}_PSC_x = np.zeros(({options["N_batch"]},{options["N_trials"]},{options["N_channels"]},2))'
        holder_declaration += f'\n    {synapse_name}_PSC_F = np.ones(({options["N_batch"]},{options["N_trials"]},{options["N_channels"]},2))'
        holder_declaration += f'\n    {synapse_name}_PSC_P = np.ones(({options["N_batch"]},{options["N_trials"]},{options["N_channels"]},2))'
        holder_declaration += f'\n    {synapse_name}_PSC_q = np.ones(({options["N_batch"]},{options["N_trials"]},{options["N_channels"]},2))'

    return holder_declaration

def declare_loop(options):

    return f"\n\n    for timestep,t in enumerate(np.arange(0,{options['sim_len']}+{options['dt']},{options['dt']})):\n"
    
def declare_odes(neurons,synapses,projections,options):

    #---------------#
    # Equation List #
    #---------------#

    # Input Neuron Equation
    #
    # ((E_L - V_t) - R*g_ad_t*(V_t-E_k) - R*g_postIC*input_t*input_netcon*(V_t-E_exc) + R*Itonic*Imask )/ tau
      
    # Non-Input Neuron Equation
    #                             
    # ((E_L - V_t) - R*g_ad_t*(V_t-E_k) - sum(R*gsyn_pre_post*pscs_t*pre_post_netcon*(V_t-PSC_ESYN)) + R*Itonic*Imask )/ tau
    #                                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #                                                  sum over all projecting cells

    # Noise injected Neuron Addition
    #
    # -R*nSYN*NoiseV3_sn*(V-noise_E_exc) / tau    -- Note: Got rid of noise-netcon because I don't think it makes sense to have a noise netcon.

    # PSC equations
    # S -> (PSC_scale * PSC_x_t - PSC_s_t)/tauR
    # x -> PSC_x_t/tauD
    # F -> (1 - PSC_F_t)/tauF
    # P -> (1 - PSC_P_t)/tauP
    # q -> 0

    # Noise equatins
    # (scale*noise_xn_t - noise_sn_t)/tauR_N
    # -noise_xn_t/tauD_N + noise_token_t/dt

    #Set header
    ODE_declaration = '\n\n        #Declare ODES\n'

    for k in neurons:
        neuron_name = k["name"]
        #If the node is an input : Use the above equation to write the ODE
        if k['is_input'] == 1:
            ODE_declaration += f'\n        {neuron_name}_V_k1 = ((({neuron_name}_E_L - {neuron_name}_V[:,:,:,-1]) - {neuron_name}_R*{neuron_name}_g_ad[:,:,:,-1]*({neuron_name}_V[:,:,:,-1]-{neuron_name}_E_k) - {neuron_name}_R*{neuron_name}_g_postIC*inputs[timestep]*{neuron_name}_netcon*({neuron_name}_V[:,:,:,-1]-{neuron_name}_E_exc) + {neuron_name}_R*{neuron_name}_Itonic*{neuron_name}_Imask) / {neuron_name}_tau)'
        
        #If the node is not an input : Use the projections to write the ODE as shown above
        else:
            projections_declaration = ''
            for j in projections[neuron_name]:
                projections_declaration += f'{j}_gSYN*{j}_PSC_s[:,:,:,-1]*{j}_netcon*({neuron_name}_V[:,:,:,-1]-{j}_ESYN) +'
            projections_declaration = projections_declaration[:-1]
            
            ODE_declaration += f'\n        {neuron_name}_V_k1 = ((({neuron_name}_E_L - {neuron_name}_V[:,:,:,-1]) - {neuron_name}_R*{neuron_name}_g_ad[:,:,:,-1]*({neuron_name}_V[:,:,:,-1]-{neuron_name}_E_k) - {neuron_name}_R*({projections_declaration}) + {neuron_name}_R*{neuron_name}_Itonic*{neuron_name}_Imask) / {neuron_name}_tau)'
        
        if k['is_noise'] == 1:
            #If this is a noise-injected node add onto the same equation as shown above
            ODE_declaration += f' + (-{neuron_name}_R * {neuron_name}_nSYN * {neuron_name}_noise_sn[:,:,:,-1]*({neuron_name}_V[:,:,:,-1])-{neuron_name}_noise_E_exc) / {neuron_name}_tau)'
            #Add in the PSC like ODEs for the noise terms
            ODE_declaration += f'\n        {neuron_name}_noise_sn_k1 = ({neuron_name}_noise_scale * {neuron_name}_noise_xn[:,:,:,-1] - {neuron_name}_noise_sn[:,:,:,-1]) / {neuron_name}_tauR_N'
            ODE_declaration += f'\n        {neuron_name}_noise_xn_k1 = -({neuron_name}_noise_xn[:,:,:,-1]/{neuron_name}_tauD_N) + noise_token[timestep]/{options["dt"]}'

        #Declare the adaptation per neuron
        ODE_declaration += f'\n        {neuron_name}_g_ad_k1 = {neuron_name}_g_ad[:,:,:,-1] / {neuron_name}_tau_ad'

    for m in synapses:
        synapse_name = m["name"]

        #Declare PSC odes
        ODE_declaration += f'\n        {synapse_name}_PSC_s_k1 = ({synapse_name}_scale*{synapse_name}_PSC_x[:,:,:,-1] - {synapse_name}_PSC_s[:,:,:,-1]) / {synapse_name}_tauR'
        ODE_declaration += f'\n        {synapse_name}_PSC_x_k1 = -{synapse_name}_PSC_x[:,:,:,-1]/{synapse_name}_tauD'
        ODE_declaration += f'\n        {synapse_name}_PSC_F_k1 = (1 - {synapse_name}_PSC_F[:,:,:,-1])/{synapse_name}_tauF'
        ODE_declaration += f'\n        {synapse_name}_PSC_P_k1 = (1 - {synapse_name}_PSC_P[:,:,:,-1])/{synapse_name}_tauP'
        ODE_declaration += f'\n        {synapse_name}_PSC_q_k1 = 0'

    return ODE_declaration

def declare_state_updates(neurons,synapses,options):

    #Set header
    state_update_declaration = '\n\n        #Declare State Updates\n'

    #Trade the indexes and then step forwards in time
    for k in neurons:
        neuron_name = k["name"]
        
        #Voltage
        state_update_declaration += f'\n        {neuron_name}_V[:,:,:,-2] = {neuron_name}_V[:,:,:,-1]'
        state_update_declaration += f'\n        {neuron_name}_V[:,:,:,-1] = {neuron_name}_V[:,:,:,-1] + {options["dt"]}*{neuron_name}_V_k1'
        #Adaptation
        state_update_declaration += f'\n        {neuron_name}_g_ad[:,:,:,-2] = {neuron_name}_g_ad[:,:,:,-1]'
        state_update_declaration += f'\n        {neuron_name}_g_ad[:,:,:,-1] = {neuron_name}_g_ad[:,:,:,-1] + {options["dt"]}*{neuron_name}_g_ad_k1'
        #Noise Updates
        if k['is_noise'] == 1:
            state_update_declaration += f'\n        {neuron_name}_noise_sn[:,:,:,-2] = {neuron_name}_noise_sn[:,:,:,-1]'
            state_update_declaration += f'\n        {neuron_name}_noise_sn[:,:,:,-1] = {neuron_name}_noise_sn[:,:,:,-1] + {options["dt"]}*{neuron_name}_noise_sn_k1'
            state_update_declaration += f'\n        {neuron_name}_noise_xn[:,:,:,-2] = {neuron_name}_noise_xn[:,:,:,-1]'
            state_update_declaration += f'\n        {neuron_name}_noise_xn[:,:,:,-1] = {neuron_name}_noise_xn[:,:,:,-1] + {options["dt"]}*{neuron_name}_noise_xn_k1'

    for j in synapses:
        synapse_name = j["name"]

        #PSC -- updates
        state_update_declaration += f'\n        {synapse_name}_PSC_s[:,:,:,-2] = {synapse_name}_PSC_s[:,:,:,-1]'
        state_update_declaration += f'\n        {synapse_name}_PSC_s[:,:,:,-1] = {synapse_name}_PSC_s[:,:,:,-1] + {options["dt"]}*{synapse_name}_PSC_s_k1'
        state_update_declaration += f'\n        {synapse_name}_PSC_x[:,:,:,-2] = {synapse_name}_PSC_x[:,:,:,-1]'
        state_update_declaration += f'\n        {synapse_name}_PSC_x[:,:,:,-1] = {synapse_name}_PSC_x[:,:,:,-1] + {options["dt"]}*{synapse_name}_PSC_x_k1'
        state_update_declaration += f'\n        {synapse_name}_PSC_F[:,:,:,-2] = {synapse_name}_PSC_F[:,:,:,-1]'
        state_update_declaration += f'\n        {synapse_name}_PSC_F[:,:,:,-1] = {synapse_name}_PSC_F[:,:,:,-1] + {options["dt"]}*{synapse_name}_PSC_F_k1'
        state_update_declaration += f'\n        {synapse_name}_PSC_P[:,:,:,-2] = {synapse_name}_PSC_P[:,:,:,-1]'
        state_update_declaration += f'\n        {synapse_name}_PSC_P[:,:,:,-1] = {synapse_name}_PSC_P[:,:,:,-1] + {options["dt"]}*{synapse_name}_PSC_P_k1'
        state_update_declaration += f'\n        {synapse_name}_PSC_q[:,:,:,-2] = {synapse_name}_PSC_q[:,:,:,-1]'
        state_update_declaration += f'\n        {synapse_name}_PSC_q[:,:,:,-1] = {synapse_name}_PSC_q[:,:,:,-1] + {options["dt"]}*{synapse_name}_PSC_q_k1'

    return state_update_declaration

def declare_condtionals(neurons,synapses):
    
    #Set header
    conditionals_declaration = '\n\n        #Declare Conditionals\n'

    for k in neurons:
        neuron_name = k["name"]
        #------------------------------------------------#
        # Condition 1 (Spiking Condition & Thresholding) #   ?TODO Make the buffer size changeable? Im not sure why it is 5, thats just what it has always been
        #------------------------------------------------#
        conditionals_declaration += f'\n        {neuron_name}_mask = (({neuron_name}_V[:,:,:,-1] >= {neuron_name}_V_thresh) & ({neuron_name}_V[:,:,:,-2] < {neuron_name}_V_thresh)).astype(np.int8)'
        if k["is_output"] == 1:
            conditionals_declaration += f'\n        {neuron_name}_spikes_holder[:,:,:,t] = {neuron_name}_mask'

        #Take care of what happens at threshold   %Note -- This is somehwat changed from the dynasim implementation, in that if we are at thrsehold we reset.
        conditionals_declaration += f'\n        {neuron_name}_V[:,:,:,-2] = np.where({neuron_name}_mask,{neuron_name}_V[:,:,:,-1], {neuron_name}_V[:,:,:,-2])'
        conditionals_declaration += f'\n        {neuron_name}_V[:,:,:,-1] = np.where({neuron_name}_mask,{neuron_name}_V_reset, {neuron_name}_V[:,:,:,-1])'
        conditionals_declaration += f'\n        {neuron_name}_g_ad[:,:,:,-2] = np.where({neuron_name}_mask,{neuron_name}_g_ad[:,:,:,-1], {neuron_name}_g_ad[:,:,:,-2])'
        conditionals_declaration += f'\n        {neuron_name}_g_ad[:,:,:,-1] = np.where({neuron_name}_mask,{neuron_name}_g_ad[:,:,:,-1]+{neuron_name}_g_inc,{neuron_name}_g_ad[:,:,:,-1])'
       
        #Get the shape of the mask
        conditionals_declaration += f'\n        B_{neuron_name}, Tr_{neuron_name}, N_{neuron_name} = {neuron_name}_mask.shape'
        
        #Find where the spiking activity is
        conditionals_declaration += f'\n        b_{neuron_name}, tr_{neuron_name}, n_{neuron_name} = np.where({neuron_name}_mask != 0)'

        #Define a vecotr that corresponds to the flattented locations off all of the spikes
        conditionals_declaration += f'\n        flat_{neuron_name} = (b_{neuron_name}*Tr_{neuron_name}*tr_{neuron_name}) * N_{neuron_name} + n_{neuron_name}'

        #Find in 3D space where the spiking activity would correspond to on the buffer index
        conditionals_declaration += f'\n        tspike_flat_{neuron_name} = {neuron_name}_tspike.reshape(B_{neuron_name}*Tr_{neuron_name}*N_{neuron_name} * 5)'

        #Update Tspike
        conditionals_declaration += f'\n        buffer_flat_{neuron_name} = {neuron_name}_buffer_index.reshape(B_{neuron_name}*Tr_{neuron_name}*N_{neuron_name})'

        #Find the rows that we are updating
        conditionals_declaration += f'\n        row_{neuron_name} = (buffer_flat_{neuron_name}[flat_{neuron_name}]-1 % 5)'

        #Find the location of each indicitdual spike within tspike and update then update it
        conditionals_declaration += f'\n        lin_{neuron_name} = (flat_{neuron_name}*5 + row_{neuron_name}).astype(np.int64)'
        conditionals_declaration += f'\n        tspike_flat_{neuron_name}[lin_{neuron_name}] = t'

        #Update the buffer index
        conditionals_declaration += f'\n        mask_flat_{neuron_name} = ({neuron_name}_mask.reshape(B_{neuron_name}*Tr_{neuron_name}*N_{neuron_name})).astype(np.int64)'
        conditionals_declaration += f'\n        buffer_flat_{neuron_name}[:] = ((buffer_flat_{neuron_name} - 1) + mask_flat_{neuron_name}) % K + 1'

        #Reshape the holders back to their original forms
        conditionals_declaration += f'\n        {neuron_name}_tspike = tspike_flat_{neuron_name}.reshape(B_{neuron_name},Tr_{neuron_name},N_{neuron_name},5)'
        conditionals_declaration += f'\n        {neuron_name}_buffer_index = buffer_flat_{neuron_name}.reshape(B_{neuron_name},Tr_{neuron_name},N_{neuron_name})'

        #------------------------------------------#
        # Condition 2 (absolute refractory period) #
        #------------------------------------------#

        #Look along the last axis of tspike (the circular buffer) to see if there are any violations of the absolute refractory period
        conditionals_declaration += f'\n        {neuron_name}_mask_ref = np.any(t <= ({neuron_name}_tspike + {neuron_name}_t_ref), axis=-1)'

        #If so reset
        conditionals_declaration += f'\n        {neuron_name}_V[:,:,:,-2] = np.where({neuron_name}_mask_ref,{neuron_name}_V[:,:,:,-1], {neuron_name}_V[:,:,:,-2])'
        conditionals_declaration += f'\n        {neuron_name}_V[:,:,:,-1] = np.where({neuron_name}_mask_ref, {neuron_name}_V_reset,{neuron_name}_V[:,:,:,-1])'

    for j in synapses:
        
        synapse_name = j["name"]
        pre_neuron_name = synapse_name.split('_',-1)[0]

        #---------------------------#
        # Condition 3 (Update PSCs) #
        #---------------------------#

        # Look along the tspike axis to see if it is time yet for the PSCs to update
        conditionals_declaration += f'\n        {synapse_name}_mask_psc = np.any(t == ({pre_neuron_name}_tspike + {synapse_name}_PSC_delay), axis=-1)'
        conditionals_declaration += f'\n        {synapse_name}_PSC_x[:,:,:,-2] = np.where({synapse_name}_mask_psc,{synapse_name}_PSC_x[:,:,:,-1], {synapse_name}_PSC_x[:,:,:,-2])'
        conditionals_declaration += f'\n        {synapse_name}_PSC_q[:,:,:,-2] = np.where({synapse_name}_mask_psc,{synapse_name}_PSC_q[:,:,:,-1], {synapse_name}_PSC_q[:,:,:,-2])'
        conditionals_declaration += f'\n        {synapse_name}_PSC_F[:,:,:,-2] = np.where({synapse_name}_mask_psc,{synapse_name}_PSC_F[:,:,:,-1], {synapse_name}_PSC_F[:,:,:,-2])'
        conditionals_declaration += f'\n        {synapse_name}_PSC_P[:,:,:,-2] = np.where({synapse_name}_mask_psc,{synapse_name}_PSC_P[:,:,:,-1], {synapse_name}_PSC_P[:,:,:,-2])'

        #Insert the PSC updates
        conditionals_declaration += f'\n        {synapse_name}_PSC_x[:,:,:,-1] = np.where({synapse_name}_mask_psc,{synapse_name}_PSC_x[:,:,:,-1] + {synapse_name}_PSC_q[:,:,:,-1], {synapse_name}_PSC_x[:,:,:,-1])'
        conditionals_declaration += f'\n        {synapse_name}_PSC_q[:,:,:,-1] = np.where({synapse_name}_mask_psc,{synapse_name}_PSC_F[:,:,:,-1] * {synapse_name}_PSC_P[:,:,:,-1], {synapse_name}_PSC_q[:,:,:,-1])'
        conditionals_declaration += f'\n        {synapse_name}_PSC_F[:,:,:,-1] = np.where({synapse_name}_mask_psc,{synapse_name}_PSC_F[:,:,:,-1] + {synapse_name}_PSC_fF * ({synapse_name}_PSC_maxF - {synapse_name}_PSC_F[:,:,:,-1]), {synapse_name}_PSC_F[:,:,:,-1])'
        conditionals_declaration += f'\n        {synapse_name}_PSC_P[:,:,:,-1] = np.where({synapse_name}_mask_psc,{synapse_name}_PSC_P[:,:,:,-1] * (1 - {synapse_name}_PSC_fP), {synapse_name}_PSC_P[:,:,:,-1])'

    return conditionals_declaration

def declare_returns(neurons):

    for k in neurons:

        return_declaration = ''

        #Using inplace operations (only saving current and previous step) for memory
        neuron_name = k["name"]

        #Spike holder -- Holds the output of the network -- only save the outputs to the designated output neurons to save memory
        if k["is_output"] == 1:
            return_declaration += f'{neuron_name}_spikes_holder,'

    return_declaration = return_declaration[:-1]
    #Build out peripherals
    return_declaration = '\n\n    return [' + return_declaration + ']'

    return return_declaration