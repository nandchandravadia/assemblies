from brain import Brain, Area
import numpy as np
import random

#rng = np.random.default_rng(64193) #generator
random.seed(889)



def generate_binary_vector(n, k):
    # Create a vector of size n initialized with zeros
    vector = [0] * n
    
    # Generate k random indices to place the 1's
    indices = random.sample(range(n), k)
    #indices = rng.choice(range(n), size=k, replace=False)

    
    # Place 1's at the selected indices
    for index in indices:
        vector[index] = 1
    
    return np.array(vector).reshape((n,1))


#======= PARAMETERS OF MODEL=========== #
n = 2* 10 ** 3
k = 89
p = 10 ** -2
beta = 0.0 #0.01
synapse_type = "random" #G_{n,p}
#########################################


# Specify Area Connections
stimulusArea = "stimulus"

hippocampus = "HF"
HF_seed = 1239

cortex = "cortex"
cortex_seed = 2349

# Initialize brain!
brain = Brain()

# Add brain areas
brain.add_area(areaName = hippocampus, n = n, p = p, k = k, beta=beta, seed = HF_seed, synapse_type=synapse_type)
brain.add_area(areaName = cortex, n = n, p = p, k = k, beta=beta, seed = cortex_seed, synapse_type = synapse_type)

#specify brain area connection topology
HF_targets = [cortex]
cortex_targets = [hippocampus]

brain.connect(sourceArea=hippocampus, targetAreas=HF_targets)
brain.connect(sourceArea=cortex, targetAreas=cortex_targets)


#perform a stimulus projection into a source area
#stimulus = rng.uniform(low = 0, high = 1, size=(n,1)) # a stimulus vector
#stimulus = np.zeros(shape=(n,1))
stimulus = generate_binary_vector(n, k)

brain.stimulusProjection(stimulus=stimulus, sourceArea = hippocampus)



