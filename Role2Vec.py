import numpy as np
import networkx as nx
from gensim.models import Word2Vec
import pandas as pd
import Pearson

# Generate graph structure
def create_graph(correlation_matrix, threshold):
    num_nodes = correlation_matrix.shape[0]
    graph = nx.Graph()

    for i in range(num_nodes):
        graph.add_node(i)

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if correlation_matrix[i, j] >= threshold:
                graph.add_edge(i, j)
    return graph


# Generate node sequences
def generate_walks(graph, num_walks, walk_length):
    walks = []
    for _ in range(num_walks):
        for node in graph.nodes():
            walk = random_walk(graph, walk_length, start_node=node)
            walks.append(walk)
    return walks

# Random walk to generate node sequences
def random_walk(graph, walk_length, start_node):
    walk = [start_node]
    for _ in range(walk_length - 1):
        neighbors = list(graph.neighbors(walk[-1]))
        if len(neighbors) > 0:
            walk.append(np.random.choice(neighbors))
        else:
            break
    return walk

# Use Role2Vec to train node vectors
def role2vec(graph, dimensions, walk_length, num_walks, window_size):
    walks = generate_walks(graph, num_walks, walk_length)
    model = Word2Vec(walks, vector_size=dimensions, window=window_size, min_count=1, sg=0, workers=4)
    return model.wv

# Example correlation matrix
def Correlation_Matrix():
    df = Pearson.get_correlation_matrix()
    data1 = df.iloc[:, 1:].values
    correlation_matrix = np.array(data1)
    return correlation_matrix

# Set threshold and other parameters
threshold = 0.4
dimensions = 128
walk_length = 50
num_walks = 20
window_size = 10

# Create graph structure
def Get_Graph():
    correlation_matrix = Pearson.get_correlation_matrix()
    # print(correlation_matrix)
    # data1 = correlation_matrix.iloc[:, :].values
    correlation_matrix1 = np.array(correlation_matrix)
    print(correlation_matrix1)
    graph = create_graph(correlation_matrix1, threshold)
    return graph

# Use Role2Vec to get node vectors
def get_expvec():
    graph = Get_Graph()
    node_vectors = role2vec(graph, dimensions, walk_length, num_walks, window_size)
    # Iterate through each node in the graph and output its vector representation
    vector_list = []
    for node in graph.nodes():
        vector = node_vectors[node]
        vector_list.append(vector)

    return vector_list

# df1 = pd.DataFrame(vector_list)
# df1.to_excel("E:\data\cancer\PRAD\PRAD_Vec\PRAD_miRNA_ExpVec.xlsx")
