import numpy as np
import pandas as pd
import math

# Consider a hypothetical city with 9 districts, numbered 0 to 8

'''        They are connected as follows:-


        I ------ B ------- C
        | \      |       / |
        |  \     |     /   |
        |   \    |   /     |
        |     \  | /       |
        H--------A---------D
        |      / |  \      |
        |    /   |    \    |
        |   /    |     \   |
        |  /     |      \  |
        | /      |       \ |
        G--------F---------E

#   The distance of diagonales, ie, A->C, A->E, etc is 5 units each
#   The distance of the inner straight lines, ie, A->B, A->F, etc is 3 units each
#   The distance of outer edges, ie, A->H, A->D , etc is 4 units each

#############Geography:-
The cities on the top have lower temperature 
Middle cities  have moderate temperature
Lower cities have higher temperature

East cities have high rain and humidity
West cities have low rain and humidity
Middle cities have moderate rain and humidity
'''

# Geography matrix of 9 cities
temp = {'A': 27.0, 'B': 12.0, 'C': 8.0, 'D': 30.0,
        'E': 45.0, 'F': 42.0, 'G': 40.0, 'H': 25.0, 'I': 10.0}
rain = {'A': 211.0, 'B': 245.0, 'C': 450.0, 'D': 389.0,
        'E': 410.0, 'F': 220.0, 'G': 90.0, 'H': 113.0, 'I': 105.0}
humid = {'A': 67.0, 'B': 58.0, 'C': 92.0, 'D': 89.0,
         'E': 85.0, 'F': 63.0, 'G': 34.0, 'H': 45.0, 'I': 38.0}
irradiance = {'A': 67.0, 'B': 58.0, 'C': 92.0, 'D': 89.0,
              'E': 85.0, 'F': 63.0, 'G': 34.0, 'H': 45.0, 'I': 38.0}


# The below distance matrix is a representation of the above city

# Coordinates of regions (x,y)
regions = {
    'A': [0.0, 0.0],
    'B': [0.0, 3.0],
    'C': [4.0, 3.0],
    'D': [4.0, 0.0],
    'E': [4.0, -3.0],
    'F': [0.0, -3.0],
    'G': [-4.0, -3.0],
    'H': [-4.0, 0.0],
    'I': [-4.0, 3.0]
}


# Function to compute Euclidian distance between two cities
def dist(region1, region2):
    x1 = regions[region1][0]
    y1 = regions[region1][1]
    x2 = regions[region2][0]
    y2 = regions[region2][1]

    return math.sqrt((x1-x2)**2 + (y1-y2)**2)


# Find the minimum distance
min_dist = 1000000000000000000
for r1 in regions.keys():
    for r2 in regions.keys():
        if r1 != r2:
            d = dist(r1, r2)
            min_dist = min(d, min_dist)


# Function to Normalise data:-
# def normalise(a):
#     # Using simple feature scaling
#     max = pd.DataFrame(a, columns=["col"])["col"].max()
#     for i in range(len(a)):
#         a[i] = a[i]/max
#     return a


# Normalise the geographical data
# Normalisation is done so that each parameter has an equal impact on the
# susceptibility
# temp = normalise(temp)
# rain = normalise(rain)
# humid = normalise(humid)


# Function to calculate susceptibility
# Assuming direct proportion to rainfall and humidity
# And inverse proportion with temperature
# def calc_susceptibility(i):
# Return the susceptibility factor of the ith city
# return rain[i]*humid[i]/temp[i]


# Simulation Starts as time t=1
t = 1

# Decay constants for a particular disease
tov = np.random.random()
c = np.random.random()
phi = np.random.random()
psi = np.random.random()

# Function to find decay with time


def find_decay():
    # Calculate the decay as per the formula
    d = tov * np.exp(-psi*phi*t) + c
    return np.array([[d, d, d, d]])


# Function to find the coefficients in the Transport Factor Vn
def find_coeff(d):
    # Determine the parameters a,w,i,h of the Transport vector,
    # From random Gaussian distribution.
    # Then divide these parameters by the distance between the two nodes.
    # This division is done to make the factors inversely dependent on the distance,
    # to some extent.
    a = np.random.normal(min_dist/2, min_dist/4)/float(d)
    w = np.random.normal(min_dist/2, min_dist/4)/float(d)
    i = np.random.normal(min_dist/2, min_dist/4)/float(d)
    h = np.random.normal(min_dist/2, min_dist/4)/float(d)
    # Store these coefficients in a numpy array
    c = np.array([a, w, i, h])

    return c


# Function to calculate weight between two edges

# Higher weight means more probability of transmission
def calc_weight(x, y):
    d = dist(x, y)
    # If distance between 2 nodes is 0, then the weight is also 0
    if d == 0:
        return 0.0
    coeff = find_coeff(d)
    decay = find_decay()
    decay = decay.transpose()

    res = np.matmul(coeff, decay)
    # print(res)
    return float(res)


# The transmission graph contains info of the vector score, ie, weights
# between two nodes L1 and L2


class transmission_graph:

    # This function initialises the graph and stores weights between each node of the graph
    def initial(self):
        # Iterate over each node in the graph
        for i in regions.keys():
            # Iterate over all the neighbours
            for j in regions.keys():
                if i != j:
                    self.adj_matrix[i].append([j, calc_weight(i, j)])

        # Also assign the susceptibility value of each node
        # for i in regions.keys():
        #     susceptibility[i] = (i, calc_susceptibility(i))

    # Function to remove a region node. You have to just pass the name of the region(string) to be removed

    def remove_node(self, name):
        i = regions.pop(name)
        i = rain.pop(name)
        i = humid.pop(name)
        i = temp.pop(name)
        i = self.adj_matrix.pop(name)
        # i = susceptibility.pop(name)
        i = irradiance.pop(name)

        self.initial()

    def __init__(self):
        # n denotes the number of districts or nodes in the graph
        # which is 9 in our case
        self.n = 0
        # adj_matrix is the adjacency matrix
        # The ith row and jth column of the adjacency matrix
        # stores the weight of the edge between ith and jth node
        self.adj_matrix = {}

        self.E = [[np.random.random(), np.random.random(), np.random.random(), np.random.random()],
                  [np.random.random(), np.random.random(),
                   np.random.random(), np.random.random()],
                  [np.random.random(), np.random.random(),
                   np.random.random(), np.random.random()],
                  [np.random.random(), np.random.random(), np.random.random(), np.random.random()]]

        for place in regions:
            self.adj_matrix[place] = []

        self.initial()

        self.remove_node("A")
        self.remove_node("B")
        self.remove_node("C")
        self.remove_node("D")
        self.remove_node("E")
        self.remove_node("F")
        self.remove_node("G")
        self.remove_node("H")
        self.remove_node("I")

    # Function to add a new region node
    # You have to pass (region name(string), x coordinate(float), y coordinate(float), temperature, rainfall, humidity)

    def add_node(self, name, x, y, t1, r1, h1, i1):
        self.adj_matrix[name] = []

        regions[name] = [x, y]
        temp[name] = t1
        rain[name] = r1
        humid[name] = h1
        irradiance[name] = i1

        self.n = self.n + 1

        self.initial()

    # Function to modify the edge weight between two regions

    def change_weight(self, r1, r2, wt):
        for nbr in self.adj_matrix[r1]:
            if nbr[0] == r2:
                nbr[1] = wt

    def print_graph(self):
        for node in self.adj_matrix.keys():
            print(node, end=' : ')
            for weights in self.adj_matrix[node]:
                print(weights, end=',')
            print('\n')

    def calc_susceptibility_score(self, node):
        sn = [[temp[node], rain[node], humid[node], irradiance[node]]]
        E = np.array(self.E)
        Sn = np.array(sn)
        Sn = Sn.transpose()
        decay = find_decay()

        x = np.matmul(decay, E)
        y = np.matmul(x, Sn)

        return y
