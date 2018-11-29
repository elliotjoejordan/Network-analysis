
def readIn():
    with open ("coauthorship.txt", "r") as myfile:
        data=myfile.read()
    dataList = data.split("*")
    edges = dataList[2]
    edges = edges.split("\n")
    edges = edges[1:]
    edgeTuples = []
    for i in edges:
        edge = i.split(" ")
        if edge[2] != edge[3]:
            edgeTuples.append((int(edge[2]), int(edge[3])))
    return(edgeTuples)



import random


# first we need
class PATrial:
    """
    Used when each new node is added in creation of a PA graph.
    Maintains a list of node numbers with multiple instances of each number.
    The number of instances of each node number are in proportion to the
    probability that it is linked to.
    Uses random.choice() to select a node number from this list for each trial.
    """

    def __init__(self, num_nodes):
        """
        Initialize a PATrial object corresponding to a
        complete graph with num_nodes nodes

        Note the initial list of node numbers has num_nodes copies of
        each node number
        """
        self._num_nodes = num_nodes
        self._node_numbers = [node for node in range(num_nodes) for dummy_idx in range(num_nodes)]

    def run_trial(self, num_nodes):
        """
        Conduct num_node trials using by applying random.choice()
        to the list of node numbers

        Updates the list of node numbers so that the number of instances of
        each node number is in the same ratio as the desired probabilities

        Returns:
        Set of nodes
        """
        # compute the neighbors for the newly-created node
        new_node_neighbors = set()
        for dummy_idx in range(num_nodes):
            new_node_neighbors.add(random.choice(self._node_numbers))
        # update the list of node numbers so that each node number
        # appears in the correct ratio
        self._node_numbers.append(self._num_nodes)
        self._node_numbers.extend(list(new_node_neighbors))
        # update the number of nodes
        self._num_nodes += 1
        return new_node_neighbors


def make_complete_graph(num_nodes):
    """Takes the number of nodes num_nodes and returns a dictionary
    corresponding to a complete directed graph with the specified number of
    nodes. A complete graph contains all possible edges subject to the
    restriction that self-loops are not allowed. The nodes of the graph should
    be numbered 0 to num_nodes - 1 when num_nodes is positive. Otherwise, the
    function returns a dictionary corresponding to the empty graph."""
    # initialize empty graph
    complete_graph = {}
    # consider each vertex
    for vertex in range(num_nodes):
        # add vertex with list of neighbours
        complete_graph[vertex] = set([j for j in range(num_nodes) if j != vertex])
    return complete_graph


def make_PA_Graph(total_nodes, out_degree):
    """creates a PA_Graph on total_nodes where each vertex is iteratively
    connected to a number of existing nodes equal to out_degree"""
    # initialize graph by creating complete graph and trial object
    PA_graph = make_complete_graph(out_degree)
    trial = PATrial(out_degree)
    for vertex in range(out_degree, total_nodes):
        PA_graph[vertex] = trial.run_trial(out_degree)
    return PA_graph




import random

def grouped (m,k):
    n = 0
    count = 0
    a = 0
    dictOfNodes = {}
    while a < (m*k):
        if count == k:
            n += 1
            count = 0
        dictOfNodes[a] = n
        count += 1
        a += 1
    return dictOfNodes


def edges(p,q, groups, m, k):
    nodeList = []
    for x in range(m*k):
        nodeList.append(x)
    edgeList = []
    for i in nodeList:
        for j in nodeList:
            if i!=j:
                if groups.get(i) == groups.get(j):
                    if random.random() < p:
                        edgeList.append((i,j))
                elif groups.get(i) - groups.get(j) == 1:
                    if random.random() < p:
                        edgeList.append((i,j))
                elif groups.get(i) == m-1 and groups.get(j) == 0:
                    if random.random() < p:
                        edgeList.append((i,j))
                else:
                    if random.random() < q:
                        edgeList.append((i,j))
    return(edgeList)

def ringGraph(m, k, p, q):
    grouping = grouped(m,k)
    graph = edges(p, q, grouping, m, k)
    graphDict = dict()
    for i in graph:
        if i[0] in graphDict and i[1] in graphDict:
            if i[1] not in graphDict[i[0]]:
                graphDict[i[0]].append(i[1])
                graphDict[i[1]].append(i[0])
        elif i[0] in graphDict:
            graphDict[i[0]].append(i[1])
            graphDict[i[1]] = [(i[0])]
        elif i[1] in graphDict:
            graphDict[i[1]].append(i[0])
            graphDict[i[0]] = [(i[1])]
        else:
            graphDict[i[0]] = [(i[1])]
            graphDict[i[1]] = [(i[0])]
    return(graphDict)



def adjacency(edgeTuples):
    graphDict = dict()
    for i in edgeTuples:
        if i[0] in graphDict and i[1] in graphDict:
            if i[1] not in graphDict[i[0]]:
                graphDict[i[0]].append(i[1])
                graphDict[i[1]].append(i[0])
        elif i[0] in graphDict:
            graphDict[i[0]].append(i[1])
            graphDict[i[1]] = [(i[0])]
        elif i[1] in graphDict:
            graphDict[i[1]].append(i[0])
            graphDict[i[0]] = [(i[1])]
        else:
            graphDict[i[0]] = [(i[1])]
            graphDict[i[1]] = [(i[0])]
    return (graphDict)


def subgraphed(key, values, graph):
    subgraph = {}
    for i in values:
        connections = []
        for j in graph[i]:
            if j in values:
                connections.append(j)
        subgraph[i] = connections
    return subgraph


def nodeBrilliance(key, values, graph):
    subgraph = subgraphed(key, values, graph)
    connections = {}
    maxConnection = [list(subgraph.keys())[0], len(list(subgraph.values())[0])]
    connections[list(subgraph.keys())[0]] =  len(list(subgraph.values())[0])
    for i in range(1,len(subgraph)):
        current = [list(subgraph.keys())[i], len(list(subgraph.values())[i])]
        connections[list(subgraph.keys())[i]] = len(list(subgraph.values())[i])
        if current[1] > maxConnection[1]:
            maxConnection = current

    while maxConnection[1] > 0:
        others = subgraph[maxConnection[0]]
        connections.pop(maxConnection[0])
        subgraph.pop(maxConnection[0])
        for i in others:
            listed = subgraph[i]
            listed.remove(maxConnection[0])
            subgraph[i] = listed
            connections[i] = connections[i] - 1
        maxConnection = [-1, 0]
        for key, value in connections.items():
            if value > maxConnection[1]:
                maxConnection = [key, value]
    return len(connections)


def brilliance(graph):
    listed = []
    for key, value in graph.items():
        listed.append(nodeBrilliance(key, value, graph))
    return (listed)


#tuples = readIn()
#graph = adjacency(tuples)


'''
graph = make_PA_Graph(1559, 10)
graph2 = {}
for i in graph:
    for j in graph[i]:
        if i in graph2:
            if j in graph2[i]:
                x = 1
            else:
                graph2[i].append(j)
        else:
            graph2[i] = [j]
        if j in graph2:
            if i in graph2[j]:
                x = 1
            else:
                graph2[j].append(i)
        else:
            graph2[j] = [i]
'''


#graph = {0:[1,2,3,5], 1:[0,2,4], 2:[0,1,3,5], 3:[0,2,4,5], 4:[1,3], 5:[0,2,3]}
brillianceList = []
for i in range(100):
    print(i)
    graph = ringGraph(20, 10, 0.4, 0.1)
    for j in brilliance(graph):
        brillianceList.append(j)


brilldict = {}
for i in brillianceList:
    if i in brilldict:
        brilldict[i] += 1
    else:
        brilldict[i] = 1


dataLength = len(brillianceList)
xdata = []
ydata = []
for degree in brilldict:
    xdata += [degree]
    ydata += [brilldict[degree]/dataLength]

# import numpy as np
# import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# plot degree distribution
plt.xlabel('Brilliance')
plt.ylabel('Normalised Frequency')
plt.suptitle('Brilliance Distribution of Ring Group Graphs')
plt.title('Ring Graph (20,10,0.4,0.1)')
plt.plot(xdata, ydata, marker='.', linestyle='None', color='b')
plt.savefig('brilliance(ringGraphMultiple).png')


