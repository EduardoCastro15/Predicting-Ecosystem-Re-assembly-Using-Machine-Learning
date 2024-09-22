import math
import numpy as np
import networkx as nx
import scipy.sparse

class DivideNet:
    """Class to handle the division of a network into positive and negative training and testing networks."""
    
    def __init__(self, network, test_ratio=0.1, neg_ratio=2):
        """
        Initialize the DivideNet with a NetworkX graph, a test ratio, and a negative sampling ratio.
        
        :param network: The input NetworkX graph.
        :param test_ratio: The proportion of edges to be used for testing (default: 0.1).
        :param neg_ratio: The ratio of negative samples to positive samples (default: 2).
        """
        self.network = network
        self.test_ratio = test_ratio
        self.neg_ratio = neg_ratio
        self.network_train = None
        self.network_test = None
        self.neg_network_train = None
        self.neg_network_test = None
    
    def split_network(self):
        """Splits the network into training and testing sets based on the test ratio."""
        
        # Copies the original network for training
        self.network_train = self.network.copy()
        # Creates an empty graph with the same number of nodes for testing
        self.network_test = nx.empty_graph(self.network_train.number_of_nodes())
        
        # Calculates the number of edges in the train network
        n_links = self.network_train.number_of_edges()
        print(f"n_links: {n_links}")
        # Calculates the number of edges to be used for testing
        n_links_test = math.ceil(self.test_ratio * n_links)
        print(f"n_links_test: {n_links_test}")

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
    
    def get_train_test_networks(self):
        """
        Returns the train and test networks after splitting.
        
        :return: Tuple of (network_train, network_test)
        """
        if self.network_train is None or self.network_test is None:
            self.split_network()
        return self.network_train, self.network_test

    def generate_negative_samples(self):
        """Generates negative samples for both the training and test networks."""
        
        # Calculate the number of positive edges
        n_links_train_pos = self.network_train.number_of_edges()
        n_links_test_pos = self.network_test.number_of_edges()
        
        # Calculate the number of negative samples needed
        n_links_train_neg = self.neg_ratio * n_links_train_pos
        n_links_test_neg = self.neg_ratio * n_links_test_pos
        
        # Create an empty graph for negative links
        neg_network = nx.empty_graph(self.network.number_of_nodes())
        links_neg = list(nx.non_edges(self.network))
        neg_network.add_edges_from(links_neg)

        n_links_neg = neg_network.number_of_edges()

        # Randomly select negative links for training and testing
        selected_links_neg_id = np.random.choice(np.arange(n_links_neg), size=n_links_train_neg + n_links_test_neg, replace=False)
        
        # Create empty negative training and testing networks
        self.neg_network_train = nx.empty_graph(self.network.number_of_nodes())
        self.neg_network_test = nx.empty_graph(self.network.number_of_nodes())

        # Assign negative edges to the training set
        selected_train_neg_links = [links_neg[link_id] for link_id in selected_links_neg_id[:n_links_train_neg]]
        self.neg_network_train.add_edges_from(selected_train_neg_links)

        # Assign negative edges to the testing set
        selected_test_neg_links = [links_neg[link_id] for link_id in selected_links_neg_id[n_links_train_neg:]]
        self.neg_network_test.add_edges_from(selected_test_neg_links)
    
    def get_train_test_negative_networks(self):
        """
        Returns the negative train and test networks after generating negative samples.
        
        :return: Tuple of (neg_network_train, neg_network_test)
        """
        if self.neg_network_train is None or self.neg_network_test is None:
            self.generate_negative_samples()
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