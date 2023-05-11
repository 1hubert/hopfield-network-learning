"""
My first neural network.

An implementation of the Hopfield network,
which was introduced to me there: https://youtu.be/piF6D6CQxUw

TODO:
- If last 10/100/1000 iterations were the same, check if one of expected stable states was achieved and quit
- Experiment with micro-sized neural networks starting with 2 neurons to see how the weights influence states
"""

# There will be nodes
# Connections between nodes will be weights
# I think representing nodes as matrix (like we did at school) is the simplest approach
# the matrix will be just a python list of python lists
# As for the values assigned to every connection between nodes... Hmmmmmmmm
# Czy na node'ach są jakiekolwiek etykiety? Hmmmmmm
# Connections have weights and neurons have activity
# Activity - function of time, like v3(t)
# We assume that a neuron can only have two states: inactive (let's say, -1), and active (1)
# If our neural network had 16 nodes, that means we always have 16 ones or minus ones
# We call it the state of the network
# So the matrix I was talking about will be asigned to a var called "state" and won't have any extra info outside of -1's and 1's
# The second importance-wise variable will be called "weights" and it will be a matrix of size 16 (one row for all weights of one row)

# state = [
#     [1, -1, 1, 1],
#     [1, -1, 1, 1],
#     [1, -1, 1, 1],
#     [1, -1, 1, 1]
# ]




from copy import deepcopy
from random import randrange

def index_to_row_col(index, num_cols):
    row = index // num_cols  # Divide by number of columns
    col = index % num_cols  # Remainder is the column index
    return row, col


def row_col_to_index(row, col, num_cols):
    return row * num_cols + col


def print_state(state):
    for row in state:
        for neuron_activity in row:
            if neuron_activity == 1:
                print('⬛', end='')
            elif neuron_activity == -1:
                print('⬜', end='')
        else:
            print()

def print_timeline(t):
    for i, state in enumerate(t):
        print(f't = {i}')
        print_state(state)
        print('----------------')


def time_step(t, weights):
    """t -> t + 1"""
    state = t[-1]

    x_len = len(state)
    y_len = len(state[0])
    x_rand = randrange(x_len)
    y_rand = randrange(y_len)
    weights_index = row_col_to_index(x_rand, y_rand, y_len)
    # print(f'state[{x_rand}][{y_rand}] -> weights[{weights_index}]')

    result = 0
    for i, weight in enumerate(weights[weights_index]):
        vx, vy = index_to_row_col(i, y_len)
        result += weight * state[vx][vy]

    new_state = deepcopy(state)
    if result >= 0:
        new_state[x_rand][y_rand] = 1
    else:  # result < 0
        new_state[x_rand][y_rand] = -1

    return t + [new_state]


def train(img: list[list]) -> list[list]:
    """
    Input: Image (matrix with values -1 or 1)
    Output: Weights matrix
    """
    # Step 1: Flatten the matrix
    l = []
    for row in img:
        for elem in row:
            l.append(elem)

    # Step 2: Create outer product
    WeightMatrix = []
    for i in l:
        row = []
        for j in l:
            row +=[i*j]
        WeightMatrix.append(row)

    # Step 3: Set diagonal to 0
    for i in range(len(WeightMatrix)):
        WeightMatrix[i][i] = 0

    return WeightMatrix

# -------------
# Step 1: Define state and timeline
state = [
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
]

t = [state]

# Step 2: Train the neural network on a set of images
rice_field_img = [
    [1, 1, 1, 1, 1],
    [1, -1, 1, -1, 1],
    [1, 1, 1, 1, 1],
    [1, -1, 1, -1, 1],
    [1, 1, 1, 1, 1],
]

x_img = [
    [1, -1, -1, -1, 1],
    [-1, 1, -1, 1, -1],
    [-1, -1, 1, -1, -1],
    [-1, 1, -1, 1, -1],
    [1, -1, -1, -1, 1],
]

o_img = [
    [1, 1, 1, 1, 1],
    [1, -1, -1, -1, 1],
    [1, -1, -1, -1, 1],
    [1, -1, -1, -1, 1],
    [1, 1, 1, 1, 1],
]

weights_rice_field = train(rice_field_img)
weights_x = train(x_img)
weights_o = train(o_img)

weights = []
weights_row_count = 25
for w1, w2, w3 in zip(weights_rice_field, weights_x, weights_o):
    row = []
    for i in range(weights_row_count):
        avg = (w1[i] + w2[i] + w3[i]) / 3
        row.append(avg)

    weights.append(row)

# Step 3: Pass time x times
x = 1000
for _ in range(x):
    t = time_step(t, weights)

# Step 4: Print timeline
print_timeline(t)
