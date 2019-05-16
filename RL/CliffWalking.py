import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.axes as ax


ALPHA = 0.1
EPSILON = 0.2
GAMMA = 1
LEN = 12
WID = 4
EPISODE = 1000

action_dest = []
for i in range(0, 12):
    action_dest.append([])
    for j in range(0, 4):
        destination = dict()
        destination[0] = [i, min(j + 1, 3)]
        destination[1] = [min(i + 1, 11), j]
        if 0 < i < 11 and j == 1:
            destination[2] = [0, 0]
        else:
            destination[2] = [i, max(j - 1, 0)]
        destination[3] = [max(i - 1, 0), j]
        action_dest[-1].append(destination)
action_dest[0][0][1] = [0, 0]

action_reward = np.zeros((LEN, WID, 4))
action_reward[:, :, :] = -1.0
action_reward[1:11, 1, 2] = -100.0
action_reward[0, 0, 1] = -100.0


def take_step(x, y, a):
    goal = 0
    if x == LEN - 1 and y == 0:
        goal = 1
    if a == 0:
        y += 1
    if a == 1:
        x += 1
    if a == 2:
        y -= 1
    if a == 3:
        x -= 1

    x = max(0, x)
    x = min(LEN - 1, x)
    y = max(0, y)
    y = min(WID - 1, y)

    if goal == 1:
        return x, y, -1
    if x > 0 and x < LEN - 1 and y == 0:
        return 0, 0, -100
    return x, y, -1


def epsGreedyPolicy(x, y, q, eps):
    t = random.randint(0, 3)
    if random.random() < eps:
        a = t
    else:
        q_max = q[x][y][0]
        a_max = 0
        for i in range(4):
            if q[x][y][i] >= q_max:
                q_max = q[x][y][i]
                a_max = i
        a = a_max
    return a


def findMaxQ(x, y, q):
    q_max = q[x][y][0]
    a_max = 0
    for i in range(4):
        if q[x][y][i] >= q_max:
            q_max = q[x][y][i]
            a_max = i
    a = a_max
    return a


#Change EPSI to the following values: 0.1, 0.2 and 0.2 with a decrease of 0.2/1000 e
def sarsa(q, EPSI, decreasing):
    runs = 1
    rewards = np.zeros([EPISODE])
    EPSI = EPSI
    path_lengths = []
    cum_rewards = []
    for j in range(runs):
        for i in range(EPISODE):
            if decreasing:
                EPSI -= 0.2/1000
            print(EPSI)
            reward_sum = 0
            path_length = 0
            x = 0
            y = 0
            a = epsGreedyPolicy(x, y, q, EPSI)
            while True:
                [x_next, y_next] = action_dest[x][y][a]
                reward = action_reward[x][y][a]
                path_length += 1
                reward_sum += reward
                a_next = epsGreedyPolicy(x_next, y_next, q, EPSI)
                q[x][y][a] += ALPHA * (reward + GAMMA * q[x_next][y_next][a_next] - q[x][y][a])
                if (x == LEN - 1 and y == 0) or reward_sum < -100:
                    break
                x = x_next
                y = y_next
                a = a_next
            rewards[i] += reward_sum
            cum_rewards.append(reward_sum)
            path_lengths.append(path_length)
            #print(reward_sum)
    print(path_lengths)
    rewards /= runs
    avg_rewards = []
    for i in range(9):
        avg_rewards.append(np.mean(rewards[:i + 1]))
    for i in range(10, len(rewards) + 1):
        avg_rewards.append(np.mean(rewards[i - 10:i]))
    return avg_rewards, path_lengths, cum_rewards


def qLearning(q, EPSI, decreasing):
    runs = 1
    EPSI = EPSI
    path_lengths = []
    cum_rewards = []
    rewards = np.zeros([EPISODE])
    for j in range(runs):
        for i in range(EPISODE):
            if decreasing:
                EPSI -= 0.2/1000
            reward_sum = 0
            path_length = 0
            x = 0
            y = 0
            while True:
                a = epsGreedyPolicy(x, y, q, EPSI)
                x_next, y_next, reward = take_step(x, y, a)
                path_length += 1
                a_next = findMaxQ(x_next, y_next, q)
                reward_sum += reward
                q[x][y][a] += ALPHA * (reward + GAMMA * q[x_next][y_next][a_next] - q[x][y][a])
                if (x == LEN - 1 and y == 0) or reward_sum < -100:
                    break
                x = x_next
                y = y_next
            rewards[i] += reward_sum
            #print(reward_sum)
            #print(path_length)
            path_lengths.append(path_length)
            cum_rewards.append(reward_sum)
    print(path_lengths)
    rewards /= runs
    avg_rewards = []
    for i in range(9):
        avg_rewards.append(np.mean(rewards[:i + 1]))
    for i in range(10, len(rewards) + 1):
        avg_rewards.append(np.mean(rewards[i - 10:i]))
    return avg_rewards, path_lengths, cum_rewards


def showOptimalPolicy(q):
    for j in range(WID - 1, -1, -1):
        for i in range(LEN):
            a = findMaxQ(i, j, q)
            if a == 0:
                print(" U ", end="")
            if a == 1:
                print(" R ", end="")
            if a == 2:
                print(" D ", end="")
            if a == 3:
                print(" L ", end="")
        print("")


def showOptimalPath(q):
    x = 0
    y = 0
    path = np.zeros([LEN, WID]) - 1
    end = 0
    exist = np.zeros([LEN, WID])
    path_length = 0
    while (x != LEN - 1 or y != 0) and end == 0:
        a = findMaxQ(x, y, q)
        path[x][y] = a
        if exist[x][y] == 1:
            end = 1
        exist[x][y] = 1
        x, y, r = take_step(x, y, a)
        path_length += 1
    for j in range(WID - 1, -1, -1):
        for i in range(LEN):
            if i == 0 and j == 0:
                print(" S ", end="")
                continue
            if i == LEN - 1 and j == 0:
                print(" G ", end="")
                continue
            a = path[i, j]
            if a == -1:

                print(" - ", end="")
            elif a == 0:

                print(" U ", end="")
            elif a == 1:

                print(" R ", end="")
            elif a == 2:

                print(" D ", end="")
            elif a == 3:

                print(" L ", end="")
        print("")
    print(path_length)

# plot method takes 1 q_learning and 1 sarsa, aswell as parameters for the plot.
def plot(q_learning, sarsa, subplotnum, xlabel, ylabel, ylim = (-100,0)):
    subPlotNumber = subplotnum
    for x in range(4):
        plt.subplot(subPlotNumber)
        plt.plot(range(len(sarsa[x])), sarsa[x], label="sarsa" if x == 0 else "" )
        plt.plot(range(len(sarsa[x])), q_learning[x], label="Qlearning" if x == 0 else "")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.gca().set_title(subPlotNumber)
        plt.ylim(ylim)
        subPlotNumber += 1
    plt.figlegend(loc="lower right")
    plt.tight_layout()
    plt.show()





#initialize lists
sarsa_rewards = []
sarsa_path = []
sarsa_cum = []

q_learning_rewards =[]
q_learning_path = []
q_learning_cum = []



# Create 4 plots for each of the needed variables (i.e. path lengths, cumulative reward and the learning)
# the method in there we are looking at is "sarsa(s_grid, 0.2, True)[0]"
# first parameter is the grid, we refresh the grid everytime to restart the learning
# 2nd parameter is the epsilon
# 3rd parameter is whether or not to decrease epsilon for each episode by epsilon/epsiode amount
for x in range(4):
    s_grid = np.zeros([12, 4, 4])
    q_grid = np.zeros([12, 4, 4])

    sarsa_rewards.append(sarsa(s_grid, 0.2, True)[0])
    sarsa_path.append(sarsa(s_grid, 0.2, True)[1])
    sarsa_cum.append(sarsa(s_grid, 0.2, True)[2])

    q_learning_rewards.append(qLearning(q_grid, 0.2, True)[0])
    q_learning_path.append(qLearning(q_grid,0.2, True)[1])
    q_learning_cum.append(qLearning(q_grid,0.2, True)[2])


#plots using subplots. check plot method
plot(q_learning_rewards, sarsa_rewards, 221, "episodes", "average reward")
plot(q_learning_path, sarsa_path, 221, "episodes", "path_length", ylim=(0 , 50))
plot(q_learning_cum, sarsa_cum, 221, "episodes", "path_length", ylim=(-200, 0))



print("Sarsa Optimal Policy")
showOptimalPolicy(s_grid)
print("Q-learning Optimal Policy")
showOptimalPolicy(q_grid)

print("Sarsa Optimal Path")
showOptimalPath(s_grid)
print("Q-learning Optimal Path")
showOptimalPath(q_grid)