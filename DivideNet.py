import math
import pandas as pd
import numpy as np
import networkx as nx
import scipy.sparse
import logging

class DivideNet:
    """Class to handle the division of a network into positive and negative training and testing networks."""
    
    def __init__(self, network, test_ratio=0.1, neg_ratio=2, node_classes=None):
        """
        Initialize the DivideNet with a NetworkX graph, a test ratio, and a negative sampling ratio.
        
        :param network: The input NetworkX graph.
        :param test_ratio: The proportion of edges to be used for testing (default: 0.1).
        :param neg_ratio: The ratio of negative samples to positive samples (default: 2).
        :param node_classes: A dictionary mapping node IDs to their classification ('resource', 'consumer', 'top consumer').
        """
        self.network = network
        self.test_ratio = test_ratio
        self.neg_ratio = neg_ratio
        self.node_classes = node_classes
        self.network_train = None
        self.network_test = None
        self.neg_network_train = None
        self.neg_network_test = None

        self.logger = logging.getLogger(__name__)
    
    def split_network(self):
        """Splits the network into training and testing sets based on the test ratio."""
        
        # Copies the original network for training
        self.network_train = self.network.copy()
        # Creates an empty graph with the same number of nodes for testing
        self.network_test = nx.empty_graph(self.network_train.number_of_nodes())
        
        # Calculates the number of edges in the train network
        n_links = self.network_train.number_of_edges()
        self.logger.info(f"    Total number of links in the network: {n_links}")
        # Calculates the number of edges to be used for testing
        n_links_test = math.ceil(self.test_ratio * n_links)
        self.logger.info(f"    Number of links for testing: {n_links_test}")

        # Convert the adjacency matrix to a sparse matrix and extract the upper triangular portion
        network_adj_matrix = nx.to_scipy_sparse_array(self.network)
        network_adj_matrix = scipy.sparse.triu(network_adj_matrix, k=1)
        
        # Get the non-zero elements, which correspond to edges
        row_index, col_index = network_adj_matrix.nonzero()
        links = [(x, y) for x, y in zip(row_index, col_index)]
        
        # Ensure that we don't select more links than available
        if n_links_test > len(links):
            raise ValueError(f"Test ratio too high. Requested {n_links_test} links but only {len(links)} available.")

        # Randomly selects edges for the test set
        selected_links_id = np.random.choice(np.arange(len(links)), size=n_links_test, replace=False)

        # Select the subset of edges for testing
        selected_links = [links[link_id] for link_id in selected_links_id]
        
        # Remove the selected edges from the training network and add them to the test network
        self.network_train.remove_edges_from(selected_links)
        self.network_test.add_edges_from(selected_links)

        # Log the number of positive edges in training and testing
        self.logger.info(f"    Training Positive Samples: {self.network_train.number_of_edges()}")
        self.logger.info(f"    Testing Positive Samples: {self.network_test.number_of_edges()}")
    
    def get_train_test_networks(self):
        """
        Returns the train and test networks after splitting.
        
        :return: Tuple of (network_train, network_test)
        """
        if self.network_train is None or self.network_test is None:
            self.split_network()
        return self.network_train, self.network_test

    def classify_nodes(self, resources, top_consumers):
        """
        Classify nodes in the network as 'resource', 'top consumer', or 'consumer'.
        
        :param resources: List of node IDs classified as resources.
        :param top_consumers: List of node IDs classified as top consumers.
        """
        self.node_classes = {}
        for node in self.network.nodes:
            if node in resources:
                self.node_classes[node] = 'resource'
            elif node in top_consumers:
                self.node_classes[node] = 'top consumer'
            else:
                self.node_classes[node] = 'consumer'
    
    def generate_biologically_realistic_negative_samples(self):
        """Generates negative samples, limited to resource-resource and top consumer-top consumer pairs."""
        
        # Calculate the number of positive edges
        n_links_train_pos = self.network_train.number_of_edges()
        n_links_test_pos = self.network_test.number_of_edges()
        
        # Calculate the number of negative samples needed
        n_links_train_neg = self.neg_ratio * n_links_train_pos
        n_links_test_neg = self.neg_ratio * n_links_test_pos

        # Create lists for valid negative links
        resource_nodes = [node for node, role in self.node_classes.items() if role == 'resource']
        top_consumer_nodes = [node for node, role in self.node_classes.items() if role == 'top consumer']

        # Generate valid negative links
        resource_resource_neg_links = list(nx.non_edges(self.network.subgraph(resource_nodes)))
        top_consumer_top_consumer_neg_links = list(nx.non_edges(self.network.subgraph(top_consumer_nodes)))
        
        # Combine the valid negative links
        valid_neg_links = resource_resource_neg_links + top_consumer_top_consumer_neg_links
        
        # Log if there are no valid negative links
        if len(valid_neg_links) == 0:
            self.logger.warning("No biologically realistic negative links found between resource-resource or top consumer-top consumer pairs.")
        
        # If no valid negative links are available, fallback to random sampling
        if len(valid_neg_links) < (n_links_train_neg + n_links_test_neg):
            self.logger.warning("Falling back to random negative sampling.")
            valid_neg_links = list(nx.non_edges(self.network))  # Use all non-edges in the network

        # Shuffle and select the required number of negative links for train and test
        np.random.shuffle(valid_neg_links)
        
        selected_neg_links_train = valid_neg_links[:n_links_train_neg]
        selected_neg_links_test = valid_neg_links[n_links_train_neg:n_links_train_neg + n_links_test_neg]
        
        # Create empty negative training and testing networks
        self.neg_network_train = nx.empty_graph(self.network.number_of_nodes())
        self.neg_network_test = nx.empty_graph(self.network.number_of_nodes())
        
        # Add the selected negative links
        self.neg_network_train.add_edges_from(selected_neg_links_train)
        self.neg_network_test.add_edges_from(selected_neg_links_test)

        # Log the number of negative samples
        self.logger.info(f"    Training Negative Samples: {len(self.neg_network_train.edges)}")
        self.logger.info(f"    Testing Negative Samples: {len(self.neg_network_test.edges)}")
        self.logger.info(f"    Ratio of Training Positive/Negative: {len(self.network_train.edges()) / max(len(self.neg_network_train.edges()), 1)}")
        self.logger.info(f"    Ratio of Testing Positive/Negative: {len(self.network_test.edges()) / max(len(self.neg_network_test.edges()), 1)}")
    
    def get_train_test_negative_networks(self):
        """
        Returns the negative train and test networks after generating negative samples.
        
        :return: Tuple of (neg_network_train, neg_network_test)
        """
        if self.neg_network_train is None or self.neg_network_test is None:
            self.generate_biologically_realistic_negative_samples()
        return self.neg_network_train, self.neg_network_test

    def get_combined_train_test_links(self):
        """
        Combines the positive and negative links for both training and testing sets.
        
        :return: Tuple of (all_links_train, all_links_test)
        """
        # Combine positive and negative links for training
        all_links_train = list(self.network_train.edges) + list(self.neg_network_train.edges)
        # Combine positive and negative links for testing
        all_links_test = list(self.network_test.edges) + list(self.neg_network_test.edges)
        return all_links_train, all_links_test

    def get_labels(self):
        """
        Creates labels for the training and test sets.
        1 for positive edges (real connections) and 0 for negative edges (non-existent connections).
        
        :return: Tuple of (label_train, label_test)
        """
        # Create labels for training: 1 for positive, 0 for negative
        label_train = [1] * len(self.network_train.edges) + [0] * len(self.neg_network_train.edges)
        # Create labels for testing: 1 for positive, 0 for negative
        label_test = [1] * len(self.network_test.edges) + [0] * len(self.neg_network_test.edges)
        return label_train, label_test