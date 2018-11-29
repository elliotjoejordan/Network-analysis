import random

def make_random_graph(num_nodes, prob):
    """Returns a dictionary to a random graph with the specified number of nodes
    and edge probability.  The nodes of the graph are numbered 0 to
    num_nodes - 1.  For every pair of nodes, i and j, the pair is considered
    twice: once to add an edge (i,j) with probability prob, and then to add an
    edge (j,i) with probability prob.
    """
    #initialize empty graph
    random_graph = {}
    #consider each vertex
    for vertex in range(num_nodes):
        out_neighbours = []
        for neighbour in range(num_nodes):
            if vertex != neighbour:
                random_number = random.random()
                if random_number < prob:
                    out_neighbours += [neighbour]
        #add vertex with list of out_ neighbours
        random_graph[vertex] = out_neighbours
    return random_graph


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
            graphDict[i[1]] = [grouping[i[1]], i[0]]
        elif i[1] in graphDict:
            graphDict[i[1]].append(i[0])
            graphDict[i[0]] = [grouping[i[0]], i[1]]
        else:
            graphDict[i[0]] = [grouping[i[0]], i[1]]
            graphDict[i[1]] = [grouping[i[1]], i[0]]
    return(graphDict)


def randomSearch(s, t, graph):
    neighbourList = {}
    for i in graph:
        neighbourList[i] = len(graph[i])
    current = s
    queries = 0
    while current != t:
        query = random.randint(0,neighbourList[current]-1)
        queries += 1
        current = graph[current][query]
    return queries


def ringSearch(s, t, graph, p, q, grouping):
    neighbourList = {}
    for i in graph:
        neighbourList[i] = len(graph[i])
    current = s
    queries = 0
    while current != t:
        neighboursLeft = graph[current].copy()
        step = False
        while step == False:
            if len(neighboursLeft) > 0:
                query = random.choice(neighboursLeft)
                neighboursLeft.remove(query)
                queries += 1
                currentDistance = min(abs(grouping[current]-grouping[t]), 20-abs(grouping[current]-grouping[t]))
                queryDistance = min(abs(grouping[query]-grouping[t]), 20-abs(grouping[query]-grouping[t]))
                if query == t:
                    current = t
                    step = True
                elif queryDistance == currentDistance:
                    if random.random() < p:
                        current = query
                        queries += 1
                        step = True
                elif queryDistance < currentDistance:
                    current = query
                    queries += 1
                    step = True
                else:
                    if random.random() < q:
                        current = query
                        queries += 1
                        step = True
            else:
                while step == False:
                    next = random.choice(graph[current])
                    if abs(grouping[next]-grouping[t]) <= abs(grouping[current]-grouping[t]):
                        current = next
                        step = True
    return queries



graph = make_random_graph(100, 0.1)

'''
graph = ringGraph(20,10,0.4,0.01)
grouping = {}
for i in graph:
    grouping[i] = graph[i][0]
    graph[i] = graph[i][1:]
    
'''


searchTimes = {}
for i in range(100):
    times = []
    for x in graph:
        for y in graph:
            if x != y:
                time = randomSearch(x,y,graph)
                times.append(time)
    time = int(round(sum(times) / float(len(times))))
    if time in searchTimes:
        searchTimes[time] += 1
    else:
        searchTimes[time] = 1



xdata = []
ydata = []
for time in searchTimes:
    xdata += [time]
    ydata += [searchTimes[time]]

import matplotlib.pyplot as plt

plt.xlabel('Search Times')
plt.ylabel('Number Of Instances')
plt.suptitle('Search Times Of Random Graphs', fontsize = 18)
plt.title('30 x Random Graph (100, 0.1)', fontsize = 10)
plt.plot(xdata, ydata, marker='.', linestyle='None', color='b')
plt.savefig('randomSearch(100,0.1)100.png')

