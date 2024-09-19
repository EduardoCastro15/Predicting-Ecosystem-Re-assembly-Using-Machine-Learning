import networkx as nx
from DataLoader import DataLoader
from GraphVisualizer import GraphVisualizer

if __name__ == "__main__":
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

    # Convert the adjacency matrix data['net'] to a dense matrix and then create a NetworkX graph from it
    network = nx.from_numpy_array(data['net'].todense())
    
    # Create an instance of GraphVisualizer and draw the graph
    graph_visualizer = GraphVisualizer(network)
    graph_visualizer.draw_graph()