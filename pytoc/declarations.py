#This file serves as a user "front-end"
#The user will declare their nodes and the connectivity between these nodes (synapses).
#This architecture will then be interpreted by the dfs->euler_compiler to build the solve file

#------------------------------------------------------------------#
# Please first declared your nodes and thier respective properties #
#------------------------------------------------------------------#
# Description: Within this area, you can create neurons by calling the function Neuron.
#              After all nuerons have been declared you must wrap all of the neurons
#              within a dictionary to be used downstream

from mechs import Lif_Neuron
from mechs import Lif_Synapse
from BuildGraph import Forwards_Method

Onset_input = Lif_Neuron.Build_Vars(name = 'On')
Relay_1 = Lif_Neuron.Build_Vars(name = 'ROn')

neurons = {'neurons' : [Onset_input, Relay_1]}

print(neurons)

#---------------------------------------------------------------------#
# Next, please declare your synapses and respective synapse properies #
#---------------------------------------------------------------------#
# Convention : Pre Node  ->  Post Node   Ex. On_ROn 
On_R1_synpase = Lif_Synapse.Build_Vars(name = 'On_ROn')

print(On_R1_synpase)