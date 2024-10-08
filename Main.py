import numpy as np
import time
import logging
import pandas as pd

from WLNM import WLNM
from CSVLogger import CSVLogger
from DivideNet import DivideNet
from DataLoader import DataLoader
from GraphVisualizer import GraphVisualizer

# Set up logging
logging.basicConfig(filename='logs/general_log.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load the list of food webs from the CSV file
foodweb_csv_path = 'data/foodweb_metrics_small.csv'
foodweb_data = pd.read_csv(foodweb_csv_path)

if __name__ == "__main__":
    size = 10
    test_ratio = 0.1
    iterations = 10

    # Loop through each food web listed in the CSV file
    for index, row in foodweb_data.iterrows():
        foodweb_name = row['Foodweb']  # Get the foodweb name

        """
        Creating csv to store AUC scores
        """
        # Path for the CSV file
        log_file_path = f'logs/AUC_Results/iteration_results_{foodweb_name}.csv'

        # Initialize the CSVLogger
        csv_logger = CSVLogger(log_file_path)

        """
        Importing the dataset
        """
        # Specify the file path (can be either .mat or .csv)
        data_file_path = f'data/foodwebs/{foodweb_name}.csv'  # or 'data/NS.mat'

        # Log dataset information
        logging.info(f"Loading dataset from {data_file_path} for dataset: {foodweb_name}")
        logging.info(f"    Size of k = {size}")
        logging.info(f"    Test ratio = {test_ratio}")

        # Create an instance of DataLoader and load the data
        data_loader = DataLoader(data_file_path)
        network, node_classes = data_loader.load_data()
        
        # Visualize network
        # GraphVisualizer(network).draw_graph()

        # Check if node_classes is available (it will be None for .mat files)
        if node_classes is None:
            logging.warning(f"Node classification not available for this dataset {foodweb_name} (e.g., .mat file).")
            node_classes = {}  # Fallback if node_classes is not available

        # Loop through different sizes (from 1 to n)
        for size in range(5, size+1):
            logging.info(f"~~~~~~~Processing for Size {size} started for {foodweb_name}~~~~~~~")

            auc_scores = []
            # Loop for n iterations
            for i in range(iterations):
                logging.info(f"    ~~~~~~~~~~~~~~~Iteration {i+1} started for {foodweb_name}~~~~~~~~~~~~~~~")

                """
                Dividing the network into the training and test networks
                """
                network_splitter = DivideNet(network, test_ratio=0.1, neg_ratio=2, node_classes=node_classes)
                network_train, network_test = network_splitter.get_train_test_networks()

                # Visualize the training and test networks
                # GraphVisualizer(network_train).draw_graph()
                # GraphVisualizer(network_test).draw_graph()

                """
                Dividing the network into the training and test negative networks
                """
                neg_network_train, neg_network_test = network_splitter.get_train_test_negative_networks()

                # Visualize the negative training and test networks
                # GraphVisualizer(neg_network_train).draw_graph()
                # GraphVisualizer(neg_network_test).draw_graph()

                """
                Grouping training and test links
                    Prepare a training dataset for a machine learning model
                    Combine positive and negative links for training and testing
                """
                all_links_train, all_links_test = network_splitter.get_combined_train_test_links()

                # Create labels for training and testing sets
                label_train, label_test = network_splitter.get_labels()

                # Convert labels to NumPy arrays
                y_train, y_test = np.array(label_train), np.array(label_test)

                """
                WLNM - Weisfeiler-Lehman Neural Machine
                """
                # Initialize and train the WLNM model
                wlnm = WLNM(network_train, network_test, all_links_train, all_links_test, label_train, label_test, size)

                # Measure the time taken to train the model
                start_time = time.time()  # Record the start time
                wlnm.train_model()  # Train the model
                end_time = time.time()  # Record the end time

                # Log the execution time
                execution_time = end_time - start_time
                logging.info(f"Iteration {i+1} - Time taken to train the model for {foodweb_name}: {execution_time:.2f} seconds")

                # Evaluate the model and get the AUC
                auc = wlnm.evaluate_model()
                auc_scores.append(auc)  # Store the AUC score for this iteration

                # Log the AUC score for the current iteration
                logging.info(f"Iteration {i+1} - AUC: {auc:.4f} for {foodweb_name}")

                # Save results to CSV
                csv_logger.log(i+1, auc, execution_time, size)

            # Calculate and print the average AUC after 10 iterations
            average_auc = np.mean(auc_scores)
            # Log the final average AUC
            logging.info(f"Final average AUC for {foodweb_name} over {iterations} iterations: {average_auc:.4f}")
            logging.info(f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # Plot the AUC scores by size
    # GraphVisualizer().plot_auc_by_size(log_file_path)
