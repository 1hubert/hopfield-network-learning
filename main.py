"""
My first neural network.

An implementation of the Hopfield network,
which was introduced to me there: https://youtu.be/piF6D6CQxUw

Interesting comment from the video about "eyes":
Guest6265+:
That's really neat. I wonder what happens if you add "eyes" to the network, i.e. some subset of neurons whose activation is driven by an external signal instead of the update function.

Layerwise Lectures:
Good point. The original paper actually mentions an input current to the neurons. This would then be combined with the summed input of all other neurons, so it would modify the update function rather than replacing it. I left  this out for simplicity. However, it's actually an important detail once you start thinking about it. It sort of biases the network to a certain state and would explain where the starting state of the network comes from, if it's not random. The story would then go like this: the "eyes" of the network (love the metaphor btw) would cue the network with a partial state and the network would then autocomplete to its memory state. Kinda-sort-of-almost-like seeing a visual cue of some sort and then being reminded of a scene from your past - if we're being a little bit fanciful. Thanks for the comment.

TODO:
- If last 10/100/1000 iterations were the same, check if one of expected stable states was achieved and quit
- Experiment with micro-sized neural networks starting with 2 neurons to see how the weights influence states
- Add a function to create a unique timeline (where no two neighboring states are the same)
- refactor so that the state is a vector, not a matrix
- add "eyes" to the network
"""
from copy import deepcopy
from random import randrange, choice


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


# Step 1: Define dimensions, state and timeline
n = 5
state = [[choice([-1, 1]) for _ in range(n)] for _ in range(n)]

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
for w1, w2, w3 in zip(weights_rice_field, weights_x, weights_o):
    row = []
    for i in range(n * n):
        avg = (w1[i] + w2[i] + w3[i]) / 3
        row.append(avg)

    weights.append(row)

# Step 3: Pass time x times
x = 1000
for _ in range(x):
    t = time_step(t, weights)

# Step 4: Print timeline
print_timeline(t)
