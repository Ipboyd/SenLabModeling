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

    Onset_input = Lif_Neuron.Build_Vars(name = 'On',is_input=1, response = 'onset',g_postIC = 0.17,is_output=0)
    Offset_input = Lif_Neuron.Build_Vars(name = 'Off',is_input=1, response = 'offset',g_postIC = 0.17)
    PV_1 = Lif_Neuron.Build_Vars(name = 'SOnOff',is_output=0, g_L = 1/100, E_L = -57, V_reset = -52, t_ref = 0.5)
    Relay_1 = Lif_Neuron.Build_Vars(name = 'ROn',is_output=1,is_noise=1, g_inc = 0.0003, tau_ad = 100)

    neurons = [Onset_input,Offset_input,PV_1,Relay_1]

    #---------------------------------------------------------------------#
    # Next, please declare your synapses and respective synapse properies #
    #---------------------------------------------------------------------#
    # Convention : Pre Node  ->  Post Node   Ex. On_ROn 
    On_R1_synapse = Lif_Synapse.Build_Vars(name = 'On_ROn',fP=0.1,gSYN=0.02,tauP=30, tauR=0.7, tauD=1.5)
    On_S1_synapse = Lif_Synapse.Build_Vars(name = 'On_SOnOff', gSYN = 0.085, fP = 0.2,tauP=80, tauR=0.1, tauD=1)
    Off_S1_synapse = Lif_Synapse.Build_Vars(name = 'Off_SOnOff', gSYN = 0.045, fP = 0,tauP=80, tauR=0.1, tauD=1)
    S1_R1_synapse = Lif_Synapse.Build_Vars(name = 'SOnOff_ROn',gSYN=0.025,fP=0.5,tauP=120, tauR=1, tauD=4.5,ESYN=-80)

    synapses = [On_R1_synapse,On_S1_synapse,Off_S1_synapse,S1_R1_synapse]
    #print(synapses)

    #--------------------------------------------------------------------------------------------------#
    # Finally this script will automaticall calculate all of the projections: Please do not edit below!#
    #--------------------------------------------------------------------------------------------------#

    projections = {}

    # for k in synapses:

    #     #Step 1: Extract synapse post_node
    #     post_node = k['name'].split('_',-1)[1]

    #     print('1')
    #     print(post_node)

    #     #Step 2: Add to dictionary. If not in dictionary add it, else add to correct key.

    #     cur_keys = projections.keys()

    #     #If empty
    #     if len(projections) == 0:
    #         projections.update({post_node : [k['name']]})
    #     else:
    #         for m in list(cur_keys):
    #             if m == post_node:
    #                 projections[post_node].append(k['name'])
    #                 print('3')
    #                 print(k['name'])
    #             else:
    #                 projections.update({post_node : [k['name']]})
    #                 print('2')
    #                 print(k['name'])

    projections = {}
    for k in synapses:
        post_node = k['name'].rsplit('_', 1)[-1]
        if post_node in projections:
            projections[post_node].append(k['name'])
        else:
            projections[post_node] = [k['name']]

    print(projections)

    return [neurons,synapses,projections]



