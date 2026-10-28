import numpy as np

v=np.zeros(12)
v[8]=1

u = np.array([[0.812, 0.868, 0.918,   1.0,
                   0.762,   0.0, 0.660,  -1.0,
                   0.705, 0.655, 0.611, 0.388]])


T1 = np.load("T1.npy")

# Initialize transition matrix and constants
T = np.zeros((12, 12, 4))
UP = 0
LEFT = 1
DOWN = 2
RIGHT = 3
states = 12
actions = 4


def state_number(row, col):
    return row * 4 + col


def move(s, a):
    row = s // 4
    col = s % 4

    if a == 0:
        row -= 1
    elif a == 1:
        col -= 1
    elif a == 2:
        row += 1
    else:
        col += 1

    # Check grid boundaries (3x4 grid)
    if row < 0 or col < 0 or row >= 3 or col >= 4:
        return s

    new_state = state_number(row, col)

    # State 5 is a wall/obstacle
    if new_state == 5:
        return s

    return new_state


def left_action(a):
    return (a + 1) % 4


def right_action(a):
    return (a - 1) % 4


# Populate the transition matrix
for s in range(12):
    # State 3 is an absorbing/terminal state
    if s == 3 or s==7:
        continue

    # State 5 is an inaccessible obstacle
    if s == 5:
        continue

    for a in range(4):
        int_s = move(s, a)
        left_s = move(s, left_action(a))
        right_s = move(s, right_action(a))

        # Accumulate probabilities for intended, left, and right movements
        T[s, int_s, a] += 0.8
        T[s, left_s, a] += 0.1
        T[s, right_s, a] += 0.1



np.save("T.npy", T)
print(np.array_equal(T,T1))
# gamma= 1.0


# action_arr=np.zeros(4)

# for i in range(0,4):
#     action_arr[i]= np.sum(np.multiply(u,np.dot(v,T[:,:,i])))
    
# print(action_arr)

# print(v)
