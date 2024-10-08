import scipy.io
import numpy as np
import networkx as nx
import pandas as pd
import os
import logging

class DataLoader:
    """
    Class to handle loading datasets from different formats (supports .mat and .csv).
    """

    def __init__(self, file_path):
        """
        Initialize the DataLoader with the file path.
        
        :param file_path: The path to the dataset file.
        """
        self.file_path = file_path
        self.node_classes = {}

        # Set up logger
        self.logger = logging.getLogger(__name__)

    def load_data(self):
        """
        Load data from the specified file. Supports both .mat and .csv formats.

        Returns:
            network: A NetworkX graph created from the adjacency matrix.
        """
        # Get the file extension
        file_extension = os.path.splitext(self.file_path)[1]

        # Load data based on the file extension
        if file_extension == '.mat':
            network = self._load_mat_file()
            node_classes = None  # No node classes for .mat files
            return network, node_classes
        elif file_extension == '.csv':
            network, node_classes = self._load_csv_file()
            connectance = self._calculate_connectance(network)
            self.logger.info(f"    Connectance of the network: {connectance}")
            return network, node_classes
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

    def _load_mat_file(self):
        """
        Load a .mat file and convert it to a NetworkX graph.

        Returns:
            network: A NetworkX graph created from the adjacency matrix in the .mat file.
        """
        # Load .mat file
        data = scipy.io.loadmat(self.file_path)
        # Extract the adjacency matrix and convert it to dense format
        adjacency_matrix = data['net'].todense()
        # Convert the adjacency matrix to a NetworkX graph
        network = nx.from_numpy_array(adjacency_matrix)
        return network

    def _load_csv_file(self):
        """
        Load a food web .csv file and create an adjacency matrix based on the 'con.taxonomy' and 'res.taxonomy' columns.
        Additionally, create a node classification dictionary.
        
        Returns:
            network: A NetworkX graph created from the adjacency matrix in the food web CSV file.
            node_classes: Dictionary classifying nodes as 'resource', 'consumer', or 'top consumer'.
        """
        # Load the CSV file
        df = pd.read_csv(self.file_path)

        # Extract 'con.taxonomy' and 'res.taxonomy' columns
        consumers = df['con.taxonomy']
        resources = df['res.taxonomy']

        # Combine consumers and resources into a unique list of species (nodes)
        species = pd.concat([consumers, resources]).unique()

        # Create a mapping of species to indices
        species_to_index = {species: index for index, species in enumerate(species)}

        # Initialize an adjacency matrix of size (n_species, n_species), where n_species is the number of unique species
        n_species = len(species)
        adj_matrix = np.zeros((n_species, n_species))

        # Fill the adjacency matrix based on consumer-resource interactions
        for consumer, resource in zip(consumers, resources):
            consumer_idx = species_to_index[consumer]
            resource_idx = species_to_index[resource]
            adj_matrix[consumer_idx, resource_idx] = 1  # Set 1 to indicate an interaction

        # Convert the adjacency matrix to a NetworkX graph
        network = nx.from_numpy_array(adj_matrix)

        # Classify nodes based on their roles
        self.node_classes = self._classify_nodes(consumers, resources)

        # Log the number of node classes
        self.logger.info(f"    Number of resources: {sum(1 for node, role in self.node_classes.items() if role == 'resource')}")
        self.logger.info(f"    Number of consumers: {sum(1 for node, role in self.node_classes.items() if role == 'consumer')}")
        self.logger.info(f"    Number of top consumers: {sum(1 for node, role in self.node_classes.items() if role == 'top consumer')}")

        return network, self.node_classes

    def _classify_nodes(self, consumers, resources):
        """
        Classify nodes as 'resource', 'consumer', or 'top consumer'.
        
        :param consumers: List of consumer species.
        :param resources: List of resource species.
        :return: A dictionary with node IDs as keys and their classification as values.
        """
        node_classes = {}

        # Mark all resources as 'resource'
        for resource in resources:
            node_classes[resource] = 'resource'

        # Mark consumers based on whether they are consumed or consume others
        for consumer in consumers:
            if consumer in node_classes:
                node_classes[consumer] = 'top consumer'
            else:
                node_classes[consumer] = 'consumer'

        return node_classes

    # Function to calculate the connectance of a network
    def _calculate_connectance(self, network):
        """
        Calculate the connectance of the network (ratio of existing links to all possible links).
        """
        num_nodes = network.number_of_nodes()
        if num_nodes > 1:
            # For directed networks, the possible links are n * (n-1)
            n_possible_links = num_nodes * (num_nodes - 1)  # Directed graph possible links
            n_existing_links = network.number_of_edges()
            connectance = n_existing_links / n_possible_links
        else:
            connectance = 0  # If the network has 1 or fewer nodes, connectance is 0
        return connectance