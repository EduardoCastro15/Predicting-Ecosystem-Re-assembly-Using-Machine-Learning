import scipy.io
import numpy as np
import networkx as nx
import pandas as pd
import os

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
            return self._load_mat_file()
        elif file_extension == '.csv':
            return self._load_csv_file()
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

        Returns:
            network: A NetworkX graph created from the adjacency matrix in the food web CSV file.
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

        return network
