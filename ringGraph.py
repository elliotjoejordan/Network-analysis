import random
import queue
import numpy as np


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


def max_dist(graph, source):
    """finds the distance (the length of the shortest path) from the source to
    every other vertex in the same component using breadth-first search, and
    returns the value of the largest distance found"""
    q = queue.Queue()
    found = {}
    distance = {}
    for vertex in graph:
        found[vertex] = 0
        distance[vertex] = -1
    max_distance = 0
    found[source] = 1
    distance[source] = 0
    q.put(source)
    while q.empty() == False:
        current = q.get()
        for neighbour in graph[current]:
            if found[neighbour] == 0:
                found[neighbour] = 1
                distance[neighbour] = distance[current] + 1
                max_distance = distance[neighbour]
                q.put(neighbour)
    return max_distance


def diameter(graph):
    distances = []
    for i in range(len(graph)):
        distances.append(max_dist(graph,i))
    return max(distances)


p = 0.01
diameters = []
probs = []
while p <= 0.99:
    print(p)
    graph = ringGraph(25,10,p,0.1)
    try:
        diameters.append([diameter(graph)])
        probs.append([p])
    except:
        x = 1
    p += 0.01

xdata = []
ydata = []
for i in range(len(probs)):
    xdata += probs[i]
    ydata += diameters[i]

# import numpy as np
# import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# plot degree distribution
plt.xlabel('Probability p')
plt.ylabel('Diameter')
plt.suptitle('Diameter Of Ring Graph Against Probability p', fontsize = 18)
plt.title('Ring Graph (25, 10, p, 0.1)', fontsize = 10)
plt.plot(xdata, ydata, marker='.', linestyle='None', color='b')
plt.savefig('Diameter6.png')

'''


def compute_in_degrees(digraph):
    in_degree = {}
    for vertex in digraph:
        in_degree[vertex] = 0
    for vertex in digraph:
        for neighbour in digraph[vertex]:
            in_degree[neighbour] += 1
    return in_degree


def in_degree_distribution(digraph):
    in_degree = compute_in_degrees(digraph)
    degree_distribution = {}
    for vertex in in_degree:
        if in_degree[vertex] in degree_distribution:
            degree_distribution[in_degree[vertex]] += 1
        else:
            degree_distribution[in_degree[vertex]] = 1
    return degree_distribution

graph = ringGraph(5,50,0.26,0.24)

graph_distribution = in_degree_distribution(graph)

distribution = {}
for degree in graph_distribution:
    distribution[degree] = graph_distribution[degree] / 1000.0

# create arrays for plotting
xdata = []
ydata = []
for degree in distribution:
    xdata += [degree]
    ydata += [distribution[degree]]

# import numpy as np
# import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# plot degree distribution
plt.xlabel('In-Degree')
plt.ylabel('Normalized Rate')
plt.title('In-Degree Distribution of Ring Graph')
plt.loglog(xdata, ydata, marker='.', linestyle='None', color='b')
plt.savefig('distribution(5,50,0.26,0.24).png')

'''