import math
import numpy as np
import networkx as nx
import pandas as pd
from functools import partial
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

class WLNM:
    """Class to handle the Weisfeiler-Lehman Neural Machine (WLNM) operations."""
    
    def __init__(self, network_train, network_test, all_links_train, all_links_test, y_train, y_test, size=10):
        """
        Initialize the WLNM class with the training and test networks, links, and labels.
        
        :param network_train: The training graph.
        :param network_test: The test graph.
        :param all_links_train: List of training links (edges).
        :param all_links_test: List of test links (edges).
        :param y_train: Labels for the training links.
        :param y_test: Labels for the test links.
        :param size: The size of the subgraph used for encoding the links.
        """
        self.network_train = network_train
        self.network_test = network_test
        self.all_links_train = all_links_train
        self.all_links_test = all_links_test
        self.y_train = y_train
        self.y_test = y_test
        self.size = size
        self.model = None
    
    def enclosing_subgraph(self, fringe, network, subgraph, distance):
        """
        Expands a subgraph centered around a given set of edges (the fringe)
        by including neighboring edges from the original graph.

        Parameters:
            fringe: A list of edges around which the subgraph is to be expanded.
            network: The full graph from which the subgraph is extracted.
            subgraph: The current subgraph being built.
            distance: A distance metric to annotate the edges.
        Returns:
            neighbor_links: The list of new edges added.
            tmp_subgraph: The updated subgraph with new edges and annotations.
        """
        neighbor_links = []
        for link in fringe:
            u, v = link[0], link[1]
            neighbor_links += list(network.edges(u)) + list(network.edges(v))
        tmp_subgraph = subgraph.copy()
        tmp_subgraph.add_edges_from(neighbor_links)
        neighbor_links = [li for li in tmp_subgraph.edges() if li not in subgraph.edges()]
        tmp_subgraph.add_edges_from(neighbor_links, distance=distance, inverse_distance=1/distance)
        return neighbor_links, tmp_subgraph
    
    def extract_enclosing_subgraph(self, link, network, size=10):
        """
        Extracts an enclosing subgraph around a given link in a network.

        Parameters:
            link: The target link (an edge) around which the subgraph will be extracted.
            network: The original graph from which the subgraph is being extracted.
            size=10: The maximum number of nodes the subgraph should contain.
        Returns:
            subgraph: Final enclosing subgraph that includes the target link and grows to the desired size.
        """
        fringe = [link]
        subgraph = nx.Graph()
        distance = 0
        subgraph.add_edge(link[0], link[1], distance=distance)
        while subgraph.number_of_nodes() < size and len(fringe) > 0:
            distance += 1
            fringe, subgraph = self.enclosing_subgraph(fringe, network, subgraph, distance)
        tmp_subgraph = network.subgraph(subgraph.nodes)
        additional_edges = [li for li in tmp_subgraph.edges if li not in subgraph.edges]
        subgraph.add_edges_from(additional_edges, distance=distance+1, inverse_distance=1/(distance+1))
        return subgraph
    
    def compute_geometric_mean_distance(self, subgraph, link):
        """
        Calculates the geometric mean distance between all nodes in a subgraph
        and two specific nodes connected by a particular link.

        Parameters:
            subgraph: A NetworkX graph representing the subgraph around the link.
            link: A tuple representing the edge (u, v) connecting two nodes u and v in the subgraph.
        Returns:
            subgraph: Subgraph with the avg_dist attribute updated for each node.
        """
        u, v = link[0], link[1]
        subgraph.remove_edge(u, v)
        n_nodes = subgraph.number_of_nodes()
        u_reachable = nx.descendants(subgraph, source=u)
        v_reachable = nx.descendants(subgraph, source=v)
        for node in subgraph.nodes:
            distance_to_u = distance_to_v = 0
            if node != u:
                distance_to_u = nx.shortest_path_length(subgraph, source=node, target=u) if node in u_reachable else 2 ** n_nodes
            if node != v:
                distance_to_v = nx.shortest_path_length(subgraph, source=node, target=v) if node in v_reachable else 2 ** n_nodes
            subgraph.nodes[node]['avg_dist'] = math.sqrt(distance_to_u * distance_to_v)
        subgraph.add_edge(u, v, distance=0)
        return subgraph
    
    def palette_wl(self, subgraph, link):
        """
        Weisfeiler-Lehman (WL) node relabeling to create a unique ordering of nodes in the subgraph.

        Parameters:
            subgraph: A NetworkX graph object representing the subgraph.
            link: A tuple representing the edge (u, v) in the subgraph.
        Returns:
            nodelist: Nodes in the order determined by the final order values.
        """
        tmp_subgraph = subgraph.copy()
        if tmp_subgraph.has_edge(link[0], link[1]):
            tmp_subgraph.remove_edge(link[0], link[1])
        avg_dist = nx.get_node_attributes(tmp_subgraph, 'avg_dist')
        df = pd.DataFrame.from_dict(avg_dist, orient='index', columns=['hash_value']).sort_index()
        df['order'] = df['hash_value'].rank(axis=0, method='min').astype(int)
        df['previous_order'] = np.zeros(df.shape[0], dtype=int)
        adj_matrix = nx.adjacency_matrix(tmp_subgraph, nodelist=sorted(tmp_subgraph.nodes)).todense()
        prime_numbers = np.array([i for i in range(10000) if WLNM.prime(i)], dtype=int)

        while any(df.order != df.previous_order):
            df['log_prime'] = np.log(prime_numbers[df['order'].values])
            total_log_primes = np.ceil(np.sum(df.log_prime.values))
            df['hash_value'] = adj_matrix @ df.log_prime.values.reshape(-1, 1) / total_log_primes + df.order.values.reshape(-1, 1)
            df.previous_order = df.order
            df.order = df['hash_value'].rank(axis=0, method='min').astype(int)

        nodelist = df.order.sort_values().index.values
        return nodelist
    
    def sample(self, subgraph, nodelist, weight='weight', size=10):
        """
        Generate a feature vector from a given subgraph, focusing on the upper triangular portion of its adjacency matrix.

        Parameters:
            subgraph: A NetworkX graph or subgraph from which the adjacency matrix will be generated.
            nodelist: The list of nodes to include in the adjacency matrix, ensuring a consistent order.
            weight='weight': Specifies which edge attribute to use as weights in the adjacency matrix.
            size=10: The desired number of nodes to consider when constructing the feature vector.
        Returns:
            vector[1:]: The function returns the vector, excluding the first element.
        """
        adj_matrix = nx.to_numpy_array(subgraph, weight=weight, nodelist=nodelist)
        vector = np.asarray(adj_matrix)[np.triu_indices(len(adj_matrix), k=1)]
        d = size * (size - 1) // 2
        if len(vector) < d:
            vector = np.append(vector, np.zeros(d - len(vector)))
        return vector[1:]

    def encode_link(self, link, network, weight='weight', size=10):
        """
        Create a fixed-size feature vector (embedding) for a specific link in a graph.

        Parameters:
            link: A tuple representing the edge (u, v).
            network: The original graph from which the subgraph centered around the link will be extracted.
            weight='weight': Specifies which edge attribute to use as weights in the adjacency matrix.
            size=10: The size of the subgraph used for encoding the links.
        Returns:
            embeded_link: The feature vector for the link.
        """
        e_subgraph = self.extract_enclosing_subgraph(link, network, size=size)
        e_subgraph = self.compute_geometric_mean_distance(e_subgraph, link)
        nodelist = self.palette_wl(e_subgraph, link)
        if len(nodelist) > size:
            nodelist = nodelist[:size]
            e_subgraph = e_subgraph.subgraph(nodelist)
            nodelist = self.palette_wl(e_subgraph, link)
        return self.sample(e_subgraph, nodelist, weight=weight, size=size)
    
    @staticmethod
    def prime(x):
        """Helper function to check if a number is prime."""
        if x < 2:
            return False
        if x == 2 or x == 3:
            return True
        for i in range(2, int(math.sqrt(x)) + 1):
            if x % i == 0:
                return False
        return True

    def generate_embeddings(self, links, network):
        """
        Generates feature embeddings for a list of links in the network.

        Parameters:
            links: List of links (edges) for which the embeddings will be generated.
            network: The original graph from which the subgraphs are extracted.
        Returns:
            embeddings: NumPy array of embeddings for the given links.
        """
        return np.array(list(map(partial(self.encode_link, network=network, weight='inverse_distance', size=self.size), links)))

    def train_model(self):
        """Trains the MLP classifier on the encoded training data."""
        X_train = self.generate_embeddings(self.all_links_train, self.network_train)
        X_train_shuffle, y_train_shuffle = shuffle(X_train, self.y_train)
        
        self.model = MLPClassifier(hidden_layer_sizes=(32, 32, 16),
                                   alpha=1e-3,
                                   batch_size=128,
                                   learning_rate_init=0.001,
                                   max_iter=200,
                                   verbose=True,
                                   early_stopping=False,
                                   tol=1e-4)
        self.model.fit(X_train_shuffle, y_train_shuffle)

        # Plot the loss curve
        plt.plot(self.model.loss_curve_)
        plt.title('Loss Curve')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.show()
    
    def evaluate_model(self):
        """Evaluates the trained model on the test data using AUC."""
        X_test = self.generate_embeddings(self.all_links_test, self.network_train)
        predictions = self.model.predict(X_test)
        fpr, tpr, thresholds = metrics.roc_curve(self.y_test, predictions, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        print(f"AUC: {auc}")
        return auc
