import numpy as np
import networkx as nx
import matplotlib.pyplot as plt



class Assembly():
    def __init__(self, areaName, assemblyName, assembly_neurons, connectome):
        """An Assembly Representation (pointer to assembly neurons)
        """
        self.area = areaName #name of area
        self.assemblyName = assemblyName #name of assembly
        self.connectome = connectome #connectome of area
        self.assembly_neurons = assembly_neurons #neurons in assembly
        self.assembly = self.create_assembly_representation() #Digraph()
        
    def create_assembly_representation(self):

        # Create an empty directed graph
        G = nx.DiGraph()
        
        assembly = {} 

        #add neurons
        for neuron in self.assembly_neurons:
            assembly[neuron] = self.connectome[neuron]
            G.add_node(neuron)

        for neuron, weights in assembly.items():
            #now loop through connections that this neuron connects to
            for synapseONTO, weight in enumerate(weights):
                if (weight != 0) & (synapseONTO in self.assembly_neurons):
                    #add synapse
                    G.add_weighted_edges_from([(neuron, synapseONTO, weight)])
                    #print("Added: {}".format((neuron, synapseONTO, weight)))
       
        return G
    
    def find_reciprocal_connections(self):
    
        reciprocal_edges = {}
        for i, j, data in self.assembly.edges(data=True):
            if self.assembly.has_edge(j, i):
                weight = data.get('weight')  
                reciprocal_edges[(i,j)] = weight
            
        return reciprocal_edges #divide by 2
    
    def find_triangles(self):

        unique_triangle_nodes = {}
        for i in self.assembly.nodes():
            for j in self.assembly.successors(i):
                if i!=j:
                    for k in self.assembly.successors(j):
                        if i!=k and j!=k:
                            if self.assembly.has_edge(k, i):
                                weight_ij = self.assembly[i][j]['weight']
                                weight_jk = self.assembly[j][k]['weight']
                                weight_ki = self.assembly[k][i]['weight']

                                unique_triangle_nodes[(i,j,k)] = (weight_ij,weight_jk,weight_ki)

        return unique_triangle_nodes #divide by 3

    def draw_assembly(self, fig_name, timestep):

        # Create a figure and axis with larger size
        fig, ax = plt.subplots(figsize=(12, 12))

        # Define a colormap for edge colors based on weights
        edge_colors = [self.assembly[u][v]['weight'] for u, v in self.assembly.edges()]
        cmap = plt.cm.coolwarm

        # Calculate edge widths based on weights
        edge_widths = [self.assembly[u][v]['weight'] * 2 for u, v in self.assembly.edges()]

        # Draw the graph with weighted edges
        pos = nx.spring_layout(self.assembly, k=0.5, iterations=50)
        nx.draw_networkx_nodes(self.assembly, pos, node_size=800, node_color='lightblue', ax=ax)
        nx.draw_networkx_labels(self.assembly, pos, font_size=14, ax=ax)
        edges = nx.draw_networkx_edges(self.assembly, pos, edge_color=edge_colors, 
                                       edge_cmap=cmap, width=edge_widths, ax=ax, arrows=True, arrowsize=20, 
                                       alpha = 0.7)

        # Create a colorbar for edge weights
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(edge_colors), vmax=max(edge_colors)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Edge Weight', fontsize=14)

        # Set the title and adjust the layout
        ax.set_title('Time Step: {}'.format(timestep), fontsize=18)
        plt.axis('off')
        plt.tight_layout()
        
        # Adjust the margins to prevent clipping
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        
        plt.show()

        fig.savefig(fname=fig_name)

        

        return
    










