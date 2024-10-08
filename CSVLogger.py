import csv
import os

class CSVLogger:
    def __init__(self, file_path):
        """
        Initialize the CSVLogger with a file path.
        """
        self.file_path = file_path

        # If the file does not exist, create it with headers
        if not os.path.isfile(self.file_path):
            with open(self.file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Iteration", "AUC", "Execution Time (s)", "Size"])

    def log(self, iteration, auc, exec_time, size):
        """
        Log an entry with iteration number, AUC score, and execution time.
        """
        with open(self.file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([iteration, auc, exec_time, size])
