import os
import requests
import scipy.io
import logging

# Set up logging for better tracking of events and errors
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataLoader:
    """Class to handle downloading, saving, and loading of dataset files."""
    
    def __init__(self, url, file_name):
        self.url = url
        self.file_name = file_name
    
    def download_file(self):
        """Downloads the file from the URL and saves it locally."""
        try:
            logging.info(f"Downloading data from {self.url}")
            response = requests.get(self.url)
            response.raise_for_status()  # Raise an error for bad status codes
            with open(self.file_name, 'wb') as f:
                f.write(response.content)
            logging.info(f"File saved as {self.file_name}")
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to download the file: {e}")
            raise

    def get_file_size(self):
        """Returns the size of the downloaded file."""
        try:
            file_size = os.path.getsize(self.file_name)
            logging.info(f"File Size: {file_size} bytes")
            return file_size
        except OSError as e:
            logging.error(f"Error getting file size: {e}")
            raise
    
    def read_file_header(self, bytes_to_read=100):
        """Reads the first few bytes of the file for validation."""
        try:
            with open(self.file_name, 'rb') as f:
                header = f.read(bytes_to_read)
            logging.info(f"File Header: {header}")
            return header
        except OSError as e:
            logging.error(f"Error reading file: {e}")
            raise
    
    def load_mat_file(self):
        """Loads the MATLAB file into a Python dictionary."""
        try:
            data = scipy.io.loadmat(self.file_name)
            logging.info(f"Loaded MATLAB file with keys: {data.keys()}")
            return data
        except (OSError, scipy.io.matlab.miobase.MatReadError) as e:
            logging.error(f"Error loading .mat file: {e}")
            raise