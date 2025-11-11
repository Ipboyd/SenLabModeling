import numpy as np

def compileGrad(neurons,synapses,projections,options):

    #Start with Basic derivatvies
    #Then Do parameter derivatives

    #Backbone
    #1. Spiking Derivatives
    #2. PSC Derivatives
    
    #w.r.t Parameter
    #1. Gsyn

    #Compute the graph of form {Neuron : Paths}
    local_graph, local_graph_synapses = compute_local_backprop_graph(neurons,synapses)

    #Build the necessary derivatives that we hook off of for parameters

    #Spiking derivative
    spikewrtV_declaration = compile_spiking_wrt_odeVoltage(neurons)
    #PSC bridge derivative
    Vwrtspike_declaration = compile_odeVoltage_wrt_spiking(synapses,options)
    #Gsyn learneable parameter derivatvie
    VwrtGsyn_declaration = compile_odeVoltage_wrt_gsyn(synapses)

    #Update Derivatvies
    update_declaration = compile_updates(local_graph,local_graph_synapses,synapses,VwrtGsyn_declaration = VwrtGsyn_declaration)

    #Add return statement based on derivatvies
    return_declaration = compile_return(synapses,VwrtGsyn_declaration = VwrtGsyn_declaration)

    loss_declaration = compile_loss()

    return spikewrtV_declaration, Vwrtspike_declaration, VwrtGsyn_declaration , update_declaration, loss_declaration, return_declaration

#########################################
#     Derivative related functions      #
#########################################

#Spiking derivative
def compile_spiking_wrt_odeVoltage(neurons):

    #Build the spiking related derivatives

    #Spiking deriviatve --- This is the third primary try at deriving this equation
    # - Here is the following logic that goes into this derivtavie
    # ----
    # The spiking derivative should consider tspike -- the timing of the spike since that is what is eventually used in the loss calculation.
    # The spiking activity can be modeled by a surroagte as shown below:
    #------------------------
    # tspike = tspike_-1 + (t - tspike_-1)(1/(e^-(V-Vth))(1/e^(V_prev-Vth)))
    #-------------------------
    # Note! At some point this might be moved around. It would be wise just to make it its own independent derivative
    spiking_deriv = '\n'

    for k in neurons:
        neuron_name = k['name']

        #More stable version
        spiking_deriv += f'        tspike_wrt_odeVoltage_{neuron_name} = (t - {neuron_name}_tspike[:,:,:,-1])*np.tanh(-({neuron_name}_V[:,:,:,-2]-{neuron_name}_V_thresh))*(1-np.tanh({neuron_name}_V[:,:,:,-1]-{neuron_name}_V_thresh)**2)\n'

        #spiking_deriv += f'        tspike_wrt_odeVoltage_{neuron_name} = (t - {neuron_name}_tspike[:,:,:,-1])*(1/(1+np.exp({neuron_name}_V[:,:,:,-2]-{neuron_name}_V_thresh)))*(np.exp(-({neuron_name}_V[:,:,:,-1]-{neuron_name}_V_thresh))/(1+np.exp(-({neuron_name}_V[:,:,:,-1]-{neuron_name}_V_thresh)))**2)\n'
        #spiking_deriv += f'        print(np.shape(tspike_wrt_odeVoltage_{neuron_name}))\n'


    return spiking_deriv

#PSC derivatives -- for bridging nodes
def compile_odeVoltage_wrt_spiking(synapses,options):
    #Build the PSC related derivative
    
    #These derivtaves are used to take the spiking activity of the output cell and bridge to upstream nodes. We start by taking the derivtive of a cells voltage
    #wrt the post synaptic current. Then we can take the derivative of the post synaptic current wrt the spiking activity. We can either repeat this process or pull out a derivative wrt a parameter if need be.

    #The following are the two equations currently used that we will take the derivative of:
    #--------------------------------------------------------
    # V = ((El-V) - adaptation - psc_s + itonic) / tau
    # psc_s = psc_s_-1 + dt*(Scale*psc_x_-1 - psc_s_-1)/tauR
    #--------------------------------------------------------
    
    voltage_wrt_psc = '\n'

    for k in synapses:
        synapse_name = k['name']
        post_node = k['name'].rsplit('_', 1)[1]
        #voltage_wrt_psc += f'        voltage_wrt_psc_{synapse_name} = {post_node}_R * ({synapse_name}_gSYN[:,None,None]*np.dot(({post_node}_V[:,:,:,-2]-{synapse_name}_ESYN),{synapse_name}_netcon))/{post_node}_tau\n'

        voltage_wrt_psc += f'        voltage_wrt_psc_{synapse_name} = {post_node}_R * ({synapse_name}_gSYN*np.dot(({post_node}_V[:,:,:,-2]-{synapse_name}_ESYN).reshape(np.shape({post_node}_V[:,:,:,-2])[0]*np.shape({post_node}_V[:,:,:,-2])[1],np.shape({post_node}_V[:,:,:,-2])[2]),{synapse_name}_netcon).reshape(np.shape({post_node}_V[:,:,:,-2])[0],np.shape({post_node}_V[:,:,:,-2])[1],np.shape({post_node}_V[:,:,:,-2])[2]))/{post_node}_tau\n'
       
        #voltage_wrt_psc += f'        print(np.shape(voltage_wrt_psc_{synapse_name}))\n'

        #1. Reshape to be np.dot blas-compatilbe
        #2. Do the dot product
        #3. Revert back to the previous shape
        
        #Example usage of blas-compatible line
        #np.dot(({post_node}_V[:,:,:,-2]-{synapse_name}_ESYN).reshape(np.shape({post_node}_V[:,:,:,-2])[0]*np.shape({post_node}_V[:,:,:,-2])[1],np.shape({post_node}_V[:,:,:,-2])[2]),{synapse_name}_netcon).reshape(np.shape({post_node}_V[:,:,:,-2])[0],np.shape({post_node}_V[:,:,:,-2])[1],np.shape({post_node}_V[:,:,:,-2])[2])

    psc_wrt_spiking = '\n'

    for k in synapses:
        synapse_name = k['name']
        pre_node = k['name'].rsplit('_', 1)[0]
        #psc_wrt_spiking += f'        psc_wrt_spiking_{synapse_name} = {options["dt"]}*({synapse_name}_scale*({synapse_name}_PSC_x[:,:,:,-2] + ((-{synapse_name}_PSC_q[:,:,:,-2]*(-np.exp(t-({pre_node}_tspike[:,:,:,-1]+{synapse_name}_PSC_delay)))))/(1+np.exp(t-({pre_node}_tspike[:,:,:,-1]+{synapse_name}_PSC_delay)))**2) - {synapse_name}_PSC_s[:,:,:,-1])/{synapse_name}_tauR\n'
        #psc_wrt_spiking += f'        print((1+np.exp(t-({pre_node}_tspike[:,:,:,-1]+{synapse_name}_PSC_delay)))**2)\n'
        #psc_wrt_spiking += f'        print(t)\n'
        #psc_wrt_spiking += f'        print({pre_node}_tspike[:,:,:,-1])\n'

        #Try stable version
        #psc_wrt_spiking += f'        psc_wrt_spiking_{synapse_name} = {options["dt"]}*({synapse_name}_scale*({synapse_name}_PSC_x[:,:,:,-2] + {synapse_name}_PSC_q[:,:,:,-2]*(1/(np.cosh(t-({pre_node}_tspike[:,:,:,-1]+{synapse_name}_PSC_delay))))**2))\n'
        #Stable version v2 replace 1/cosh with 1-tanh
        psc_wrt_spiking += f'        psc_wrt_spiking_{synapse_name} = {options["dt"]}*({synapse_name}_scale*({synapse_name}_PSC_x[:,:,:,-2] + {synapse_name}_PSC_q[:,:,:,-2]*(1-(np.tanh(t-({pre_node}_tspike[:,:,:,-1]+{synapse_name}_PSC_delay))))**2))\n'

        #psc_wrt_spiking += f'        print(np.shape(psc_wrt_spiking_{synapse_name}))\n'

    return voltage_wrt_psc + psc_wrt_spiking

#gsyn derivative
def compile_odeVoltage_wrt_gsyn(synapses):
    #This function creates takes the derivative of the same equation in the PSC derivative but wrt gsyn (synaptic weight)
    # -- This is a learnable parameter
    #-------------------------------------------
    # V = R*gsyn*pscs*netcon*(V-ESYN)
    #-------------------------------------------

    voltage_wrt_gsyn = '\n'

    for k in synapses:
        synapse_name = k['name']
        post_node = k['name'].rsplit('_', 1)[1]

        #python version
        voltage_wrt_gsyn += f'        voltage_wrt_gsyn_{synapse_name} = ({post_node}_R * {synapse_name}_PSC_s[:,:,:,-1]*{synapse_name}_netcon*({post_node}_V[:,:,:,-1]-{synapse_name}_ESYN))/{post_node}_tau\n'

        #c++ version
        #voltage_wrt_gsyn += f'        voltage_wrt_gsyn_{synapse_name} = {post_node}_R * ({synapse_name}_PSC_s[:,:,:,-1]*np.dot(({post_node}_V[:,:,:,-1]-{synapse_name}_ESYN).reshape(np.shape({post_node}_V[:,:,:,-1])[0]*np.shape({post_node}_V[:,:,:,-1])[1],np.shape({post_node}_V[:,:,:,-1])[2]),{synapse_name}_netcon).reshape(np.shape({post_node}_V[:,:,:,-1])[0],np.shape({post_node}_V[:,:,:,-1])[1],np.shape({post_node}_V[:,:,:,-1])[2]))/{post_node}_tau\n'


        #voltage_wrt_gsyn += f'        print(np.shape(voltage_wrt_gsyn_{synapse_name}))\n'

    return voltage_wrt_gsyn

##############################################
#               Calculate Loss               #
##############################################

def compile_loss():

    # trials timecourse channel
    # batch trials channel timecourse

    loss_statement = '\n\n'
    loss_statement += f'    bin_width = 200\n'
    loss_statement += f'    data = np.transpose(data,(2,0,1))[None,:,:,:]\n'                                  #Rearange the data to match the shape that the network outputs
    loss_statement += f'    ROn_spikes_holder = np.transpose(ROn_spikes_holder,(2,0,1,3))\n'                                  #Rearange the data to match the shape that the network outputs
    loss_statement += f'    num_bins, remainder = divmod(np.shape(data)[-1], bin_width)\n'              #find the number of bins we would use given a bin width of ex. 200 = 20ms
    loss_statement += f'    forwards_out_r = ROn_spikes_holder[:,:,:,:]\n'                 #Using the old syntax (see calc_output_grad.py) shave off the last couple of samples so that there is a clean PSTH without leftover
    loss_statement += f'    data_r = data[:,:,:,remainder:]\n'                              #Notice! Looks like simulation is 1 sample short? Fix this at some point
    
    #Note! Temporary fix. Pythran can't interpret a reshape of >4 dims. For our simulation that means we need to colapse the first two dims to still do this summ across binwidth trick
    
    #New version
    loss_statement += f'    forwards_out_reshaped = forwards_out_r.reshape((np.shape(forwards_out_r)[0]*np.shape(forwards_out_r)[1],np.shape(forwards_out_r)[2],num_bins,bin_width))\n'
    loss_statement += f'    data_reshaped = data_r.reshape((np.shape(data_r)[0]*np.shape(data_r)[1],np.shape(data_r)[2],num_bins,bin_width))\n'
    #Old version
    #loss_statement += f'    forwards_out_reshaped = forwards_out_r.reshape((np.shape(forwards_out_r)[0],np.shape(forwards_out_r)[1],np.shape(forwards_out_r)[2],num_bins,bin_width))\n'
    #loss_statement += f'    data_reshaped = data_r.reshape((np.shape(data_r)[0],np.shape(data_r)[1],np.shape(data_r)[2],num_bins,bin_width))\n'
    loss_statement += f'    forwards_out_hist = np.sum(np.sum(forwards_out_reshaped,axis=-1),axis=-2)\n'

    #Now we need to pop back up and reinstate our batch dimention.

    

    loss_statement += f'    data_hist = np.sum(np.sum(data_reshaped,axis=-1),axis=-2)\n'


    loss_statement += f'    forwards_out_hist = forwards_out_hist.reshape(np.shape(forwards_out_r)[0],np.shape(forwards_out_r)[1],num_bins)\n'
    loss_statement += f'    data_hist = data_hist.reshape(np.shape(data_r)[0],np.shape(data_r)[1],num_bins)\n'

    loss_statement += f'    diff = forwards_out_hist - data_hist\n'

    #loss_statement += f'    print(np.shape(diff))\n'


    loss_statement += f'    PSTH_loss_avg = np.sum(diff * diff, axis=-1)\n'

    #loss_statement += f'    print(PSTH_loss_avg)\n'

    loss_statement += f'    PSTH_deriv_avg = 2.0 * np.sum(diff, axis=-1)\n'
    loss_statement += f'    out_grad = PSTH_deriv_avg[:,:,None] *np.sum(grads, axis=-2)\n'
    #loss_statement += f'    print(np.shape(out_grad))\n'
    loss_statement += f'    print(np.mean(PSTH_deriv_avg))\n'

    #loss_statement += f'    print(np.shape(PSTH_deriv_avg))\n'

    #loss_statement += f'    print(np.shape(PSTH_deriv_avg* np.sum(grads, axis=-2)))\n'

    #loss_statement += f'    print(np.shape(PSTH_deriv_avg* np.squeeze(np.sum(grads, axis=-2))))\n'

    #loss_statement += f'    print(PSTH_loss_avg)\n'

    #loss_statement += f'    print(np.shape(PSTH_deriv_avg))\n'
    #loss_statement += f'    print(np.shape(grads))\n'

    return loss_statement


##############################################
#     Helper Functions for doing local BP    #
##############################################

#For our entire algorithm we unroll our nework like an RNN and use eprop. However at each timestep we essentially do backpropegation starting from our output node
#The following function creates this graph
def compute_local_backprop_graph(neurons, synapses):

    #Goal: Find all the subpaths from the output node to any given node.

    #1. Find output node. 
    for k in neurons:
        if k['final_grad_node'] == 1:
            final_node = k['name']
    
    #2. Do a depth first recursion and just append all paths to a list or something
    local_graph = []
    super_path = []
    completed_local_graph = depth_recursion(final_node,synapses,local_graph,super_path)

    #3. Gather nodal oriented representation within a dictionary -- ex. Node : [Paths containing that node]
    
    compiled_dict = {}

    for z in neurons:                                                     #Look through the neurons
        for m in completed_local_graph:                                   #Look through all the paths
            for q in m:                                                   #Look at all the nuerons withing those paths
                if z['name'] == q:                                        #Find the neuron within those paths that matches the neuron we are currently building into our dictionary
                    if q not in compiled_dict:                            #Check to see if it is currently in our dictionary
                        compiled_dict[q] = [m[:m.index(q)+1]]             #If not add it and append the path to the left of the current node inclusive
                    else:                                                 #Else make sure the path isn't already here and then add it
                        for p in compiled_dict[q]:
                            if p != m[:m.index(q)+1]:
                                compiled_dict[q].append(m[:m.index(q)+1])

    compiled_dict_synapses = {}

    for z in synapses:                                                    #Look through the synapses                                                                     
        for m in completed_local_graph:                                   #Match this to the paths that have a given synapse
            for q in range(len(m)-1):   
                synapse = f'{m[q+1]}_{m[q]}' 
                if z['name'] == synapse:                                        
                    if synapse not in compiled_dict_synapses:             #If the synapse exists within a path, add the path up to the synapse to the dictionary
                        compiled_dict_synapses[synapse] = [m[:q+2]]             
                    else:                                                 
                        for p in compiled_dict_synapses[synapse]:         #If it is already in the dictionary append it
                            if p != m[:q+2]:
                                compiled_dict_synapses[synapse].append(m[:q+2])
   
    
    #print(completed_local_graph)
    #print(compiled_dict)
    #print(compiled_dict_synapses)

    return compiled_dict, compiled_dict_synapses #This is the reference that we can use to build our local backprop

def depth_recursion(neuron,synapses,path,super_path):
    found_child = False #Gate appending : This removes duplicates when popping back up the depth tree
    for m in synapses: #Loop through all synapses
        pre_node = m['name'].rsplit('_', 1)[0]
        post_node = m['name'].rsplit('_', 1)[1]
                
        if neuron == post_node: #Check if the current node has a connection if so recurse
            
            found_child = True

            path.append(post_node)
            path.append(pre_node)

            depth_recursion(pre_node,synapses,path,super_path)

            #Instead of reseting the path, need to trim the path based on how many steps backwars we take
            path = path[:-2]

    if not found_child:
        sorted_path_unique, idx = np.unique(path, return_index = True)
        super_path.append(sorted_path_unique[np.argsort(idx)].tolist())

    return super_path


##############################################
#   Helper Functions for compiling derivs    #
##############################################

def compile_updates(local_graph,local_graph_synapses,synapses,VwrtGsyn_declaration=''):

    gsyn_declaration = '\n'
    if len(VwrtGsyn_declaration) > 0:    
        #Compile Gsyn related derivatives (one per synapse)
        for k in synapses:                      #Go through each synapse
            synapse_name = k['name']
            paths = local_graph_synapses[synapse_name]
            cur_gsyn = f'        spike_wrt_gsyn_{synapse_name} += '
            pre_node = synapse_name.rsplit('_', 1)[0]
            post_node = synapse_name.rsplit('_', 1)[1]
            for z in paths:
                for m in range(len(z)-1):
                    cur_gsyn += f'tspike_wrt_odeVoltage_{post_node}*voltage_wrt_psc_{synapse_name}*psc_wrt_spiking_{synapse_name}*'
                cur_gsyn += f'tspike_wrt_odeVoltage_{pre_node}*voltage_wrt_gsyn_{synapse_name}+'
            cur_gsyn = cur_gsyn[:-1]
            cur_gsyn += '\n'
            gsyn_declaration += cur_gsyn
            #gsyn_declaration += f'        print(np.shape(spike_wrt_gsyn_{synapse_name}))\n'
            #gsyn_declaration += f'        print(np.max(spike_wrt_gsyn_{synapse_name},axis=0))\n'

    return gsyn_declaration

def compile_return(synapses,VwrtGsyn_declaration=''):
    
    return_statement = '    grads = np.stack(['
    if len(VwrtGsyn_declaration) > 0:  
        #Compile Gsyn related derivatives (one per synapse)
        for k in synapses:                      #Go through each synapse
            synapse_name = k['name']
            return_statement += f'spike_wrt_gsyn_{synapse_name},'
        return_statement = return_statement[:-1]
        return_statement += '], axis = 0)'


    #return_statement += '\n    print(np.shape(grads))'

    return return_statement
