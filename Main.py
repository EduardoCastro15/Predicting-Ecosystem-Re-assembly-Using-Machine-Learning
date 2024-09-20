import numpy as np
import networkx as nx

from DivideNet import DivideNet
from DataLoader import DataLoader
from GraphVisualizer import GraphVisualizer

if __name__ == "__main__":
    """
    Importing the dataset
    """
    url = 'https://github.com/KienMN/Weisfeiler-Lehman-Neural-Machine/raw/e6bea9dc464fb693264677f3c1d5442b21385d66/data/USAir.mat'
    file_name = 'USAir.mat'
    
    # Create an instance of DataLoader
    data_loader = DataLoader(url, file_name)
    # Download the dataset
    data_loader.download_file()
    # Get file size
    data_loader.get_file_size()
    # Read and print the file header
    data_loader.read_file_header()
    # Load the MATLAB file
    data = data_loader.load_mat_file()

    """
    Create a graph from the adjacency matrix
    """
    # Convert the adjacency matrix data['net'] to a dense matrix and then create a NetworkX graph from it
    network = nx.from_numpy_array(data['net'].todense())
    
    # Visualize network
    GraphVisualizer(network).draw_graph()

    """
    Dividing the network into the training and test networks
    """
    # Create an instance of DivideNet and split the network into training and testing
    network_splitter = DivideNet(network, test_ratio=0.1)
    network_train, network_test = network_splitter.get_train_test_networks()

    # Visualize the training and test networks
    GraphVisualizer(network_train).draw_graph()
    GraphVisualizer(network_test).draw_graph()

    """
    Dividing the network into the training and test negative networks
    """
    # Generate negative samples for the training and testing sets
    neg_network_train, neg_network_test = network_splitter.get_train_test_negative_networks()

    # Visualize the negative training and test networks
    GraphVisualizer(neg_network_train).draw_graph()
    GraphVisualizer(neg_network_test).draw_graph()

    """
    Grouping training and test links
        Prepare a training dataset for a machine learning model
    """
    # Combine positive and negative links for training and testing
    all_links_train, all_links_test = network_splitter.get_combined_train_test_links()

    # Create labels for training and testing sets
    label_train, label_test = network_splitter.get_labels()

    # Convert labels to NumPy arrays
    y_train, y_test = np.array(label_train), np.array(label_test)