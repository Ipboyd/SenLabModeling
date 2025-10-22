def Declare_Architecture(opts):

    #This file serves as a user "front-end"
    #The user will declare their nodes and the connectivity between these nodes (synapses).
    #This architecture will then be interpreted by the dfs->euler_compiler to build the solve file

    #------------------------------------------------------------------#
    # Please first declared your nodes and thier respective properties #
    #------------------------------------------------------------------#
    # Description: Within this area, you can create neurons by calling the function Neuron.
    #              After all nuerons have been declared you must wrap all of the neurons
    #              within a list to be used downstream

    from mechs import Lif_Neuron
    from mechs import Lif_Synapse

    Onset_input = Lif_Neuron.Build_Vars(name = 'On',is_input=1, N_chans = opts['N_channels'])
    Offset_input = Lif_Neuron.Build_Vars(name = 'Off',is_input=1)
    Relay_1 = Lif_Neuron.Build_Vars(name = 'ROn',is_output=1,is_noise=1)

    neurons = [Onset_input, Relay_1]

    #---------------------------------------------------------------------#
    # Next, please declare your synapses and respective synapse properies #
    #---------------------------------------------------------------------#
    # Convention : Pre Node  ->  Post Node   Ex. On_ROn 
    On_R1_synapse = Lif_Synapse.Build_Vars(name = 'On_ROn')
    Off_R1_synapse = Lif_Synapse.Build_Vars(name = 'Off_ROn')

    synapses = [On_R1_synapse,Off_R1_synapse]
    #print(synapses)

    #--------------------------------------------------------------------------------------------------#
    # Finally this script will automaticall calculate all of the projections: Please do not edit below!#
    #--------------------------------------------------------------------------------------------------#

    projections = {}

    for k in synapses:

        #Step 1: Extract synapse post_node
        post_node = k['name'].split('_',-1)[1]

        #Step 2: Add to dictionary. If not in dictionary add it, else add to correct key.

        cur_keys = projections.keys()

        #If empty
        if len(projections) == 0:
            projections.update({post_node : [k['name']]})
        else:
            for m in list(cur_keys):
                if m == post_node:
                    projections[post_node].append(k['name'])
                else:
                    projections.update({post_node : [k['name']]})

    return [neurons,synapses,projections]



