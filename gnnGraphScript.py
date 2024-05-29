import pandas as pd
import numpy as np
import damask
import networkx as nx
import csv
import matplotlib.pyplot as plt
from io import BytesIO
import matplotlib.image as mpimg

from os import listdir
from os.path import isfile, join
edge_file = ""
node_file = ""


def add_cracked_edges(graph, node1, node2, edgelabel):
    # if (node1, node2) not in edgelabel:
        # print("Edge not found", edge_file, node_file, node1, node2)
    # nx.set_edge_attributes(graph, {(node1, node2): {'edge_color': 'blue'}})
    edgelabel[(node1, node2)] = 1

def create_graph(nodes, edges, filename, data = None):
    graph = nx.Graph()
    nodelabel = {}
    edgelabel = {}

    for row in nodes.index:
        name = str(round(nodes.loc[row, 'x'],2)) + ',' + str(round(nodes.loc[row, 'y'],2))
        nodelabel[nodes.loc[row, 'Node']] = name
        arr = np.asarray([float(nodes.loc[row, 'Phi']), float(nodes.loc[row, 'phi1']), float(nodes.loc[row, 'phi2'])])
        cur = damask.Rotation.from_Euler_angles(arr, degrees = True)
        graph.add_node(nodes.loc[row, 'Node'],\
            label = name, pos = (nodes.loc[row, 'x'],\
            nodes.loc[row, 'y']),\
            # x = [float(nodes.loc[row, 'GrainSize']), float(cur.quaternion[0]), float(cur.quaternion[1]), float(cur.quaternion[2]), float(cur.quaternion[3])], \
            x = [float(nodes.loc[row, 'GrainSize']), float(nodes.loc[row, 'Phi']), float(nodes.loc[row, 'phi1']), float(nodes.loc[row, 'phi2'])], \
            GrainSize = nodes.loc[row, 'GrainSize'],\
            phi1 = nodes.loc[row, 'phi1'],\
            Phi = nodes.loc[row, 'Phi'],\
            phi2 = nodes.loc[row, 'phi2'])
            
    for row in edges.index:
        node1 = min([edges.loc[row, 'EndNodes_1'], edges.loc[row, 'EndNodes_2']])
        node2 = max([edges.loc[row, 'EndNodes_1'], edges.loc[row, 'EndNodes_2']])
        edgelabel[(edges.loc[row, 'EndNodes_1'], edges.loc[row, 'EndNodes_2'])] = 0
        graph.add_edge(edges.loc[row, 'EndNodes_1'],\
                edges.loc[row, 'EndNodes_2'])

    # fig, ax = plt.subplots()

    for item in data[filename]:
        node1, node2 = min(item), max(item)
        add_cracked_edges(graph, node1, node2, edgelabel)

    for edge in graph.edges():
        nx.set_edge_attributes(graph, {edge: {'edge_label': edgelabel[(edge[0], edge[1])]}})
    '''
    nx.draw_networkx(graph,\
                    nx.get_node_attributes(graph, 'pos'),\
                    labels = nodelabel,\
                    with_labels=True,\
                    node_size=20,\
                    ax= ax,\
                    font_size = 6,
                    edge_color = nx.get_edge_attributes(graph, 'edge_color').values())

    nx.draw_networkx_edge_labels(graph, nx.get_node_attributes(graph, 'pos'),\
                                edge_labels = edgelabel,\
                                font_size = 3,\
                                ax = ax)

    ax = ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.savefig("graphVisualization/" + filename +"_viz.png", format="PNG")
    '''
    return graph

# Extracting all files from a directory

def extractDataFromFiles(edge_files, node_files, data):
    retdata = []
    for i in range(len(edge_files)):
        edge_file = edge_files[i]
        node_file = node_files[i]
        # print(edge_file, node_file)
        edges = pd.read_csv("edge_data/" + edge_files[i])
        nodes = pd.read_csv("node_data/" + node_files[i])
        graph = create_graph(nodes, edges, edge_files[i].split(".")[0].split('_')[0], data)
        retdata.append(graph)

    return retdata

def getCrackedEdge(fname):
    df = pd.read_excel(fname,index_col = 0)
    crackToEdge = {}
    for item in df.index:
        data = df.loc[item, 'Pair'].split(']')
        for subitem in data:
            if subitem != '':
                if subitem.strip() != '[':
                    subitem = subitem.split('[')[1]
                    edge1, edge2 = subitem.split(',')
                    if item not in crackToEdge:
                        crackToEdge[item] = []
                    crackToEdge[item].append([int(edge1), int(edge2)])
                else:
                    if item not in crackToEdge:
                        crackToEdge[item] = []
    return crackToEdge



# return list of networkx graphs
def get_data():
    edge_files = [f for f in listdir("edge_data/") if isfile(join("edge_data/", f)) if f != "rename.py"]
    node_files = [f for f in listdir("node_data/") if isfile(join("node_data/", f))]
    data = getCrackedEdge("cracks.xlsx")
    data = extractDataFromFiles(edge_files, node_files, data)
    return data



