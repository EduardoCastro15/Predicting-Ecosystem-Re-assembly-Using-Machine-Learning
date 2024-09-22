import numpy as np
import time

from WLNM import WLNM
from DivideNet import DivideNet
from DataLoader import DataLoader
from GraphVisualizer import GraphVisualizer

if __name__ == "__main__":
    """
    Importing the dataset
    """
    # Specify the file path (can be either .mat or .csv)
    file_path = 'data/USAir.mat'  # or 'data/NS.mat'

    # Create an instance of DataLoader and load the data
    data_loader = DataLoader(file_path)
    network = data_loader.load_data()
    
    # Visualize network
    GraphVisualizer(network).draw_graph()

    # Variable to store AUC scores
    auc_scores = []

    # Loop for 10 iterations
    for i in range(1):
        print(f"Iteration {i+1}")

        """
        Dividing the network into the training and test networks
        """
        # Create an instance of DivideNet and split the network into training and testing
        network_splitter = DivideNet(network, test_ratio=0.1)
        network_train, network_test = network_splitter.get_train_test_networks()

        # Visualize the training and test networks
        # GraphVisualizer(network_train).draw_graph()
        # GraphVisualizer(network_test).draw_graph()

        """
        Dividing the network into the training and test negative networks
        """
        # Generate negative samples for the training and testing sets
        neg_network_train, neg_network_test = network_splitter.get_train_test_negative_networks()

        # Visualize the negative training and test networks
        # GraphVisualizer(neg_network_train).draw_graph()
        # GraphVisualizer(neg_network_test).draw_graph()

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

        """
        WLNM - Weisfeiler-Lehman Neural Machine
        """
        # Initialize and train the WLNM model
        wlnm = WLNM(network_train, network_test, all_links_train, all_links_test, label_train, label_test)

        # Measure the time taken to train the model
        start_time = time.time()  # Record the start time
        wlnm.train_model()  # Train the model
        end_time = time.time()  # Record the end time

        # Calculate and print the execution time
        print(f"Time taken to train the WLNM model: {end_time - start_time:.2f} seconds")

        # Evaluate the model and get the AUC
        auc = wlnm.evaluate_model()
        auc_scores.append(auc)  # Store the AUC score for this iteration

    # Calculate and print the average AUC after 10 iterations
    average_auc = np.mean(auc_scores)
    print(f"Average AUC over 10 iterations: {average_auc:.4f}")