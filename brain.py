import numpy as np
from numpy.random import binomial
import heapq
from brain_utils import ConvergenceLogger
from assembly import Assembly


class Area(ConvergenceLogger):
    def __init__(self, areaName, n, p, k, beta, seed, synapse_type):
        super(ConvergenceLogger, self).__init__()

        """Brain area
        n: number of neurons in a area
        p: probability of connections
        k: k-cap
        seed: random seed
        """
        self.area = areaName #name of area
        self.n = int(n) #number of neurons in area
        self.p = p #probability of connection to area
        self.k = k #k-cap for an area
        self.beta = beta #plasticity coefficient 
        self.seed = seed
        self.synapse_type = synapse_type
        self.rng = np.random.default_rng(seed) #generator
        self.assemblies = {} #neurons of assembly

        self.connectome = {} #(weighted) connections (from neuron ID)
        self.hasFired = set() #which neurons have fired? 

        ###logging###
        self.logging = ConvergenceLogger()


    def stimulate_random_neurons(self, size): 
        """stimulate random neurons in an area
        size: the number of random neurons
        """
        neurons = self.rng.choice(a=self.n, size = self.k, replace=False)
        activations = np.zeros(shape = (self.n,1)) #store in column major format (easier memory)

        for neuron in neurons:
            #has the neuron fired before?
            if neuron not in self.hasFired:
                #add the neuron to whether it has fired
                self.hasFired.add(neuron)

                #generate its synapses "on demand"
                self.generate_synapse(neuron)

                #update activaton vector
                activations[neuron] = 1

        return neurons, activations
    
    def stimulate_neurons(self, neurons):
        """stimulate neurons in an area
        neurons: the neurons to stimulate
        """
        for neuron in neurons:
            #has the neuron fired before?
            if neuron not in self.hasFired:
                #add the neuron to whether it has fired
                self.hasFired.add(neuron)

                #generate its synapses "on demand"
                self.generate_synapse(neuron)

        return
    
    def generate_synapse(self, neuron):
        """generate the synapses for each neuron "on demand"
        """

        if neuron not in self.connectome:
            #generate the synapses from this neuron
            self.connectome[neuron] = self.rng.binomial(1,self.p,self.n).astype(float)           

        return
    
    def activate_synapses(self, activations):
        """generate the synapses that each "active" neuron synapses onto
        """

        for neuronID, activity in enumerate(activations):
            if activity != 0:
                for neuron, isSynapse in enumerate(self.connectome[neuronID]):
                    if isSynapse:
                        self.generate_synapse(neuron = neuron)

        return
            
    
    def compute_synaptic_input(self, activations):
        """compute the synaptic input (W^T*x) onto each neuron generated thus far
        """
        #activations  = activation vector

        synaptic_inputs = np.zeros(shape = (self.n,1))

        for index in range(0, self.n):
            #now, loop through connectome
            for neuronID, weights in self.connectome.items():
                synaptic_inputs[index] += weights[index]*activations[neuronID].item()

        return synaptic_inputs
    
    def winner_take_all(self, synaptic_inputs): 

        new_activations = np.zeros(shape = (self.n,1))

        winners = heapq.nlargest(self.k, range(len(synaptic_inputs)), synaptic_inputs.__getitem__)

        new_activations[winners] = 1

        return winners, new_activations
    
    def plasticity_rule(self, previous_activations, new_activations):

        #find neurons which are active at t-1
        previously_active_neurons = np.where((previous_activations == 1))[0]

        for neuronID in previously_active_neurons:
            #update the weights for co-active neurons
            for targetNeuron, weight in enumerate(self.connectome[neuronID]):
                #check if targetNeuron fired at t 
                if new_activations[targetNeuron] == 1: #fired!
                    #update synapse (multiplicative)
                    self.connectome[neuronID][targetNeuron] += self.beta
        return
    
    def normalize_synapses(self):

        for j in range(0, self.n):
            total_weight = 0
            for i in self.connectome:
                #compute total weights into neuron
                total_weight += self.connectome[i][j]
            
            #update weights of each neuron
            for i in self.connectome:
                if self.connectome[i][j] != 0:
                    self.connectome[i][j] = self.connectome[i][j]/total_weight

        return

class Brain():
    def __init__(self):
        self.logging = True 
        self.areas = {} #defined ares
        self.area_connections = {} #connections between areas

    def add_area(self, areaName, n, p, k, beta, seed, synapse_type):
        #Define Area
        area = Area(areaName=areaName, n=n, p=p, k=k, beta=beta, seed = seed, synapse_type=synapse_type)

        #add Area to Brain
        self.areas[areaName] = area 
        self.area_connections[areaName] = set() #initialize possible area connections

        if self.logging:
            print("{} added to Brain".format(areaName))

        return 
    
    def connect(self, sourceArea, targetAreas):
        """specify the topology of area connections
        sourceArea: name of source Area
        targetArea: list of targetAreas
        """

        for target in targetAreas:
            self.area_connections[sourceArea].add(target)

            if self.logging:
                print("Area Connection: {} --> {}".format(sourceArea, target))
            
        return


    def project(self, sourceArea, targetArea):
        #project from sourceArea to targetArea

        source = self.area_connections[sourceArea]
        target = self.area_connections[targetArea]

        return
    
    def stimulusProjection(self, stimulus, sourceArea): 
        #project a stimulus into a sourceArea

        source = self.areas[sourceArea]

        #first, generate a random activation vector of size n with only k active neurons
        k = source.k    
        winners, activations = source.stimulate_random_neurons(size=k)

        #log winners
        source.logging.update(timestep=0, winners=winners)

        #TODO: Define formal convergence criteria
        convergence = True
        T = 40 #11
        new_activations = np.zeros(shape = (source.n,1)) #store in column major format (easier on memory)

        for time in range(1, T):
            #now, generate the synapses that new activations synapse onto 
            source.activate_synapses(activations)

            #compute synaptic input into each neuron
            synaptic_inputs = source.compute_synaptic_input(activations)
            
            #apply stimulus input to endogenous synaptic inputs
            synaptic_inputs += stimulus

            #now, apply thresholding (k-cap)
            new_winners, new_activations = source.winner_take_all(synaptic_inputs)

            #stimulate the firing of new winners (logging purposes)
            source.stimulate_neurons(new_winners)

            #update weights
            source.plasticity_rule(previous_activations=activations, new_activations=new_activations)


            print("time step {}: {}".format(time, len(set(new_winners)-set(winners))/k))

            #update 
            activations = new_activations
            winners = new_winners

            #log updates
            source.logging.update(timestep=time, winners=new_winners)


        if convergence:
            #now, we have an assembly
            assemblyName = "stimulus1"
            source.assemblies[assemblyName] = Assembly(areaName = source.area, assemblyName=assemblyName, 
                                                   assembly_neurons=winners, connectome=source.connectome)

            fig_name = "assembly3.png"
            source.assemblies[assemblyName].draw_assembly(fig_name, time)
            unique_reciprocal_nodes = source.assemblies[assemblyName].find_reciprocal_connections()
            unique_triangle_nodes = source.assemblies[assemblyName].find_triangles()

            print(int(len(unique_reciprocal_nodes))/2)
            print(int(len(unique_triangle_nodes))/3)


        return convergence


