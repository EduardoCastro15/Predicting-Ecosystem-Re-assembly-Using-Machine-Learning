import numpy as np
import time
import logging

from WLNM import WLNM
from DivideNet import DivideNet
from DataLoader import DataLoader
from GraphVisualizer import GraphVisualizer

# Set up logging
logging.basicConfig(filename='logs/L1P1_log.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    """
    Importing the dataset
    """
    # Specify the file path (can be either .mat or .csv)
    file_path = 'data/foodwebs/L1P1.csv'  # or 'data/NS.mat'

    # Log dataset information
    logging.info(f"Loading dataset from {file_path}")

    # Create an instance of DataLoader and load the data
    data_loader = DataLoader(file_path)
    network, node_classes = data_loader.load_data()
    
    # Visualize network
    # GraphVisualizer(network).draw_graph()

    # Check if node_classes is available (it will be None for .mat files)
    if node_classes is None:
        logging.info("Node classification not available for this dataset (e.g., .mat file).")
        node_classes = {}  # Fallback if node_classes is not available

    # Variable to store AUC scores
    auc_scores = []

    # Number of iterations
    iterations = 1

    # Loop for 10 iterations
    for i in range(iterations):
        logging.info(f"Iteration {i+1} started")

        """
        Dividing the network into the training and test networks
        """
        # Create an instance of DivideNet and split the network into training and testing
        network_splitter = DivideNet(network, test_ratio=0.1, neg_ratio=2, node_classes=node_classes)
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

        # Log the execution time
        execution_time = end_time - start_time
        logging.info(f"Iteration {i+1} - Time taken to train the model: {execution_time:.2f} seconds")

        # Evaluate the model and get the AUC
        auc = wlnm.evaluate_model()
        auc_scores.append(auc)  # Store the AUC score for this iteration

        # Log the AUC score for the current iteration
        logging.info(f"Iteration {i+1} - AUC: {auc:.4f}")

    # Calculate and print the average AUC after 10 iterations
    average_auc = np.mean(auc_scores)
    # Log the final average AUC
    logging.info(f"Final average AUC over {iterations} iterations: {average_auc:.4f}")
    logging.info(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
