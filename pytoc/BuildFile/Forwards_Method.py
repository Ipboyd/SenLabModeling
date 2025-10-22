def Euler_Compiler(neurons,synapses,projections,options):
    #1. Declare all of the variables
    variable_declaration = declare_vars(neurons,synapses,options)
    print(variable_declaration)
    #2. Declare the necessary holders
    holder_declaration = declare_holders(neurons,synapses,options)
    print(holder_declaration)
    #3. Build Euler Loop
    Euler_loop_declaration = declare_loop(options)
    print(Euler_loop_declaration)
    #4. Declare ODEs
    ODE_declaration = declare_odes(neurons,synapses,projections,options)
    print(ODE_declaration)
    #5. Declare State Updates
    State_Update_declaration = declare_state_updates(neurons,synapses,options)
    print(State_Update_declaration)
    #6. Declare conditionals

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
            ODE_declaration += f'\n        {neuron_name}_V_k1 = ((({neuron_name}_E_L - {neuron_name}_V[:,:,:,-1]) - {neuron_name}_R*{neuron_name}_g_ad[:,:,:,-1]*({neuron_name}_V[:,:,:,-1])-{neuron_name}_E_k) - {neuron_name}_R*{neuron_name}_g_post_IC*input[timestep]*{neuron_name}_netcon*({neuron_name}_V[:,:,:,-1])-{neuron_name}_E_Exc) + {neuron_name}_R*{neuron_name}_Itonic*{neuron_name}_Imask) / {neuron_name}_tau)'
        
        #If the node is not an input : Use the projections to write the ODE as shown above
        else:
            projections_declaration = ''
            for j in projections[neuron_name]:
                projections_declaration += f'{j}_gSYN*{j}_PSC_s[:,:,:,-1]*{j}_netcon*({neuron_name}_V-{j}_ESYN + '
            projections_declaration = projections_declaration[:-1]
            
            ODE_declaration += f'\n        {neuron_name}_V_k1 = ((({neuron_name}_E_L - {neuron_name}_V[:,:,:,-1]) - {neuron_name}_R*{neuron_name}_g_ad[:,:,:,-1]*({neuron_name}_V[:,:,:,-1])-{neuron_name}_E_k) - {neuron_name}_R*{projections_declaration} + {neuron_name}_R*{neuron_name}_Itonic*{neuron_name}_Imask) / {neuron_name}_tau)'
        
        if k['is_noise'] == 1:
            #If this is a noise-injected node add onto the same equation as shown above
            ODE_declaration += f' + (-{neuron_name}_R * {neuron_name}_nSYN * {neuron_name}_Noise_sn[:,:,:,-1]*({neuron_name}_V[:,:,:,-1])-{neuron_name}_noise_E_exc) / {neuron_name}_tau)'
            #Add in the PSC like ODEs for the noise terms
            ODE_declaration += f'\n        {neuron_name}_noise_sn_k1 = ({neuron_name}_noise_scale * {neuron_name}_noise_xn[:,:,:,-1] - {neuron_name}_noise_sn[:,:,:,-1]) / {neuron_name}_tauR_N'
            ODE_declaration += f'\n        {neuron_name}_noise_xn_k1 = -({neuron_name}_noise_xn[:,:,:,-1]/{neuron_name}_tauD_N) + noise_token[timestep]/dt'

        #Declare the adaptation per neuron
        ODE_declaration += f'\n        {neuron_name}_g_ad_k1 = {neuron_name}_g_ad[:,:,:,-1] / {neuron_name}_On_tau_ad'

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
        state_update_declaration += f'\n        {neuron_name}_V[:,:,:,-1] = {neuron_name}_V[:,:,:,-1] + dt*{neuron_name}_V_k1'
        #Adaptation
        state_update_declaration += f'\n        {neuron_name}_g_ad[:,:,:,-2] = {neuron_name}_g_ad[:,:,:,-1]'
        state_update_declaration += f'\n        {neuron_name}_g_ad[:,:,:,-1] = {neuron_name}_g_ad[:,:,:,-1] + dt*{neuron_name}_g_ad_k1'
        #Noise Updates
        if k['is_noise'] == 1:
            state_update_declaration += f'\n        {neuron_name}_noise_sn[:,:,:,-2] = {neuron_name}_noise_sn[:,:,:,-1]'
            state_update_declaration += f'\n        {neuron_name}_noise_sn[:,:,:,-1] = {neuron_name}_noise_sn[:,:,:,-1] + dt*{neuron_name}_sn_k1'
            state_update_declaration += f'\n        {neuron_name}_noise_xn[:,:,:,-2] = {neuron_name}_noise_xn[:,:,:,-1]'
            state_update_declaration += f'\n        {neuron_name}_noise_xn[:,:,:,-1] = {neuron_name}_noise_xn[:,:,:,-1] + dt*{neuron_name}_xn_k1'

    for j in synapses:
        synapse_name = j["name"]

        #PSC -- updates
        state_update_declaration += f'\n        {synapse_name}_PSC_s[:,:,:,-2] = {synapse_name}_PSC_s[:,:,:,-1]'
        state_update_declaration += f'\n        {synapse_name}_PSC_s[:,:,:,-1] = {synapse_name}_PSC_s[:,:,:,-1] + dt*{synapse_name}_PSC_s_k1'
        state_update_declaration += f'\n        {synapse_name}_PSC_x[:,:,:,-2] = {synapse_name}_PSC_x[:,:,:,-1]'
        state_update_declaration += f'\n        {synapse_name}_PSC_x[:,:,:,-1] = {synapse_name}_PSC_x[:,:,:,-1] + dt*{synapse_name}_PSC_x_k1'
        state_update_declaration += f'\n        {synapse_name}_PSC_F[:,:,:,-2] = {synapse_name}_PSC_F[:,:,:,-1]'
        state_update_declaration += f'\n        {synapse_name}_PSC_F[:,:,:,-1] = {synapse_name}_PSC_F[:,:,:,-1] + dt*{synapse_name}_PSC_F_k1'
        state_update_declaration += f'\n        {synapse_name}_PSC_P[:,:,:,-2] = {synapse_name}_PSC_P[:,:,:,-1]'
        state_update_declaration += f'\n        {synapse_name}_PSC_P[:,:,:,-1] = {synapse_name}_PSC_P[:,:,:,-1] + dt*{synapse_name}_PSC_P_k1'
        state_update_declaration += f'\n        {synapse_name}_PSC_q[:,:,:,-2] = {synapse_name}_PSC_q[:,:,:,-1]'
        state_update_declaration += f'\n        {synapse_name}_PSC_q[:,:,:,-1] = {synapse_name}_PSC_q[:,:,:,-1] + dt*{synapse_name}_PSC_q_k1'

    return state_update_declaration

def declare_condtionals():


