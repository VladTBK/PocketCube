from __future__ import annotations


from moves import Move, MOVES

import numpy as np
from heapq import heappop, heappush
import time
import csv
from collections import deque
from math import sqrt, log
from random import choice, randint

N = "N"
Q = "Q"
PARENT = "parent"
ACTIONS = "actions"
STATE = "state"
EXPLORED = "explored"
BUDGET = [1000, 5000, 10000, 20000]
GODSNUMBER = 14
CP = [0.5, 1.0]
DISTANCE = 10
RUNS = 20

dbList = []
for i in range(DISTANCE + 1):
    dbList.append(f"patternDB_{i}.txt")


class Cube:
    def __init__(self, moves=None, scrambled: bool = True):
        self.goal_state = np.repeat(np.arange(6), 4)
        self.state = np.repeat(np.arange(6), 4)

        if moves or scrambled:
            self.scramble(moves)

    def scramble(self, moves=None):

        if moves is None:
            num_of_moves = np.random.randint(5, 11)
            moves = list(np.random.randint(len(MOVES), size=num_of_moves))

        self.state = Cube.move_state(self.state, moves)

    def move(self, move) -> Cube:
        cube = Cube()
        cube.state = Cube.move_state(self.clone_state(), move)
        return cube

    @staticmethod
    def move_state(state: np.ndarray, move) -> np.ndarray:
        move = Move.parse(move)

        if isinstance(move, list):
            for m in move:
                state = state[MOVES[m.value]]
        else:
            state = state[MOVES[move.value]]

        return state

    def clone_state(self) -> np.ndarray:
        return np.copy(self.state)

    def clone(self) -> Cube:
        cube = Cube()
        cube.state = self.clone_state()
        return cube

    def __lt__(self, other):
        pass


# Sum collum diffrences
def h1(scramble: np.ndarray, goal: np.ndarray):
    cost = 0
    for idx, val in enumerate(scramble):
        cost += abs(val - goal[idx])
    return cost


# Sum of steps of each element
def h2(scrambled: np.ndarray, goal: np.ndarray):
    indexCounter = {
        0: [0, 1, 2, 3],
        1: [4, 5, 6, 7],
        2: [8, 9, 10, 11],
        3: [12, 13, 14, 15],
        4: [16, 17, 18, 19],
        5: [20, 21, 22, 23],
    }
    cost = 0
    for idx, val in enumerate(scrambled):
        cost += abs(idx - indexCounter[val][0])
        indexCounter[val] = indexCounter[val][1:]
    return cost


def h3(scrambled: np.ndarray, goal: np.ndarry, db=dbList[7], h=h1):
    currDB = None
    cost = 0
    with open(db) as file:
        currDB = file.read()
    currDB = eval(currDB)
    if tuple(scrambled) in currDB:
        cost = currDB[tuple(scrambled)]
    else:
        cost = h(scrambled, goal)
    return cost


def applyAllMoves(scramble: Cube()):
    cubeList = []
    for i in range(6):
        tempCube = scramble.move(i)
        cubeList.append(tempCube)
    return cubeList


def initNode(parent=None, state=None):
    return {
        N: 0,
        Q: 0,
        EXPLORED: [],
        STATE: state,
        PARENT: parent,
        ACTIONS: {},
    }


def selectAction(node, c):
    N_node = node[N]
    maxVal = 0
    maxIdx = 0
    for idx, action in node[ACTIONS].items():
        temp = action[Q] / action[N] + c * sqrt(2 * log(N_node) / action[N])
        if maxVal < temp:
            maxVal = temp
            maxIdx = idx

    return maxIdx


def solvedMTCS(tree, goal: np.ndarray):
    node = tree
    count = 0
    while not np.all(np.equal(node[STATE], goal)):
        actionList = list(node[ACTIONS])
        if (
            all(explored in tree[EXPLORED] for explored in list(tree[ACTIONS]))
            and all(explored in node[EXPLORED] for explored in actionList)
            and node[PARENT] == tree
        ):
            return None
        if actionList and not all(
            explored in node[EXPLORED] for explored in actionList
        ):
            tempMin = float("inf")
            bestAction = None
            for action in actionList:
                if node[ACTIONS][action][Q] < tempMin and action not in node[EXPLORED]:
                    tempMin = node[ACTIONS][action][Q]
                    bestAction = action
            node[EXPLORED].append(bestAction)
            node = node[ACTIONS][bestAction]

        else:
            node = node[PARENT]

    path = []
    while node[PARENT]:
        path.append(node[STATE])
        node = node[PARENT]
    return path


def astar(scrambled: Cube(), goal: Cube(), h):
    frontier = []
    heappush(frontier, (0 + h(scrambled.state, goal.state), scrambled))
    discovered = {tuple(scrambled.state): (None, 0)}
    path = []
    while frontier:
        currCube = heappop(frontier)
        currCube = currCube[1]
        path.append(currCube)
        if np.all(np.equal(currCube.state, goal.state)):
            break
        cubeList = applyAllMoves(currCube)
        for cube in cubeList:
            tempCost = discovered[tuple(currCube.state)][1] + 1
            if tuple(cube.state) not in discovered:
                discovered[tuple(cube.state)] = (currCube, tempCost)
                heappush(frontier, (tempCost + h(cube.state, goal.state), cube))
    return path, len(discovered)


def bidirectionalbfs(scrambled: Cube(), goal: Cube()):
    forwardQueue = deque([scrambled])
    forwardVisited = {tuple(scrambled.state): None}

    backwardQueue = deque([goal])
    backwardVisited = {tuple(goal.state): None}

    commonCube = None
    while not commonCube:
        # Forward BFS
        currForward = forwardQueue.popleft()
        cubeForwardList = applyAllMoves(currForward)
        for cube in cubeForwardList:
            if tuple(cube.state) not in forwardVisited:
                forwardVisited[tuple(cube.state)] = currForward
                forwardQueue.append(cube)
                if tuple(cube.state) in backwardVisited:
                    commonCube = tuple(cube.state)
                    break
        # Backward BFS
        currBackward = backwardQueue.popleft()
        cubeBackwardList = applyAllMoves(currBackward)
        for cube in cubeBackwardList:
            if tuple(cube.state) not in backwardVisited:
                backwardVisited[tuple(cube.state)] = currBackward
                backwardQueue.append(cube)
                if tuple(cube.state) in forwardVisited:
                    commonCube = tuple(cube.state)
                    break

    tempState = commonCube
    path = []
    while forwardVisited[tuple(tempState)] is not None:
        path.append(forwardVisited[tuple(tempState)])
        tempState = forwardVisited[tuple(tempState)].state
    path.reverse()
    while backwardVisited[tuple(commonCube)] is not None:
        path.append(backwardVisited[tuple(commonCube)])
        commonCube = backwardVisited[tuple(commonCube)].state
    path.append(goal)

    return path, len(forwardVisited) + len(backwardVisited)


def MTCS(scrambled: Cube(), goal: Cube(), budget, cp, h):
    tree = initNode(state=scrambled.state)
    for _ in range(budget):
        currCube = scrambled
        node = tree

        # Selection
        while not np.all(np.equal(currCube.state, goal.state)) and all(
            action in node[ACTIONS] for action in range(6)
        ):
            newAction = selectAction(node, cp)
            currCube = currCube.move(newAction)
            node = node[ACTIONS][newAction]

        # Expansion
        if not np.all(np.equal(currCube.state, goal.state)):
            newAction = choice(
                list(filter(lambda a: a not in node[ACTIONS], list(range(6))))
            )
            currCube = currCube.move(newAction)
            node = initNode(node, currCube.state)
            node[PARENT][ACTIONS][newAction] = node

        # Simulation
        simulationStates = []
        simulationCount = 0
        tempCube = currCube.clone()
        while (
            not np.all(np.array_equal(tempCube.state, goal.state))
            and simulationCount < GODSNUMBER
        ):
            tempCube = tempCube.move(move=[randint(0, 5)])
            simulationStates.append(tempCube.state)
            simulationCount += 1

        # Rewarding
        heursticValues = [h(state, goal.state) for state in simulationStates]
        reward = np.min(heursticValues)
        # Backpropagation
        crt_node = node
        while crt_node:
            crt_node[N] += 1
            crt_node[Q] += reward
            crt_node = crt_node[PARENT]

    return tree


def generateDatabase(goal: Cube(), layers):
    layer = 0
    queue = deque([goal])
    database = {tuple(goal.state): layer}
    if layers >= DISTANCE:
        while queue:
            for _ in range(6**layer):
                currNode = queue.popleft()
                currNodeList = applyAllMoves(currNode)
                for node in currNodeList:
                    if tuple(node.state) not in database:
                        database[tuple(node.state)] = layer + 1
                        queue.append(node)
    else:
        while layer < layers:
            for _ in range(6**layer):
                currNode = queue.popleft()
                currNodeList = applyAllMoves(currNode)
                for node in currNodeList:
                    if tuple(node.state) not in database:
                        database[tuple(node.state)] = layer + 1
                        queue.append(node)
            layer += 1
    return database


# DB creation
def createDatabase():
    # Note: For a distance >= 10 will result over 6^10 states > 3.7 * 10^6
    for i in range(DISTANCE + 1 - 3):
        countLayerKeys = 0
        countKeys = 0
        startTime = time.time()
        database = generateDatabase(goalCube, i)
        stopTime = time.time()
        filename = f"patternDB_{i}.txt"
        with open(filename, "w") as file:
            file.write(str(database) + "\n")
        for key in database:
            countKeys += 1
            if database[key] == i:
                countLayerKeys += 1

        print(f"Distance={i} has {countKeys} unique states")
        print(f"Layer {i} states={countLayerKeys}")
        elapsedTime = stopTime - startTime
        print(f"Distance {i} took {elapsedTime}s \n")


case1 = "R U' R' F' U"
case2 = "F' R U R U F' U'"
case3 = "F U U F' U' R R F' R"
case4 = "U' R U' F' R F F U' F U U"
caseList = [case1, case2, case3, case4]
goalCube = Cube(scrambled=False)
testCube = Cube(moves=case1, scrambled=False)
# cost = h3(testCube.state, goalCube.state, dbList[4], h1)
# print(cost)
# createDatabase()
# test all cases

# open the file in the write mode
f1 = open("astarvsbfs.csv", "w", newline="")
f2 = open("astarvsbfsvsmcts.csv", "w", newline="")
f22 = open("mcts.csv", "w", newline="")
f3 = open("astarvsmcts.csv", "w", newline="")

# create the csv writer
writer1 = csv.writer(f1)
writer2 = csv.writer(f2)
writer22 = csv.writer(f22)
writer3 = csv.writer(f3)

header1 = ["Caz de test", "Timp 1", "Timp 2", "Stari 1", "Stari 2", "Cale 1", "Cale 2"]
writer1.writerow(header1)

header2 = ["Caz de test", "Timp", "Stari", "Cale"]
writer2.writerow(header2)

header22 = [
    "Caz de test",
    "Timp 1",
    "Timp 2",
    "Timp 3",
    "Stari 1",
    "Stari 2",
    "Stari 3",
    "Cale 1",
    "Cale 2",
    "Cale 3",
]
writer22.writerow(header22)

header3 = ["Caz de test", "Cale 1", "Cale 2"]
writer3.writerow(header3)


for case in caseList:
    tempCube = Cube(moves=case, scrambled=False)

    # Algoritm A*
    startTime = time.time()
    path1, states1 = astar(tempCube, goalCube, h1)
    # path = bidirectionalbfs(tempCube, goalCube)
    # for p in path:
    #     print(p.state)
    stopTime = time.time()
    print(len(path1))
    elapsedTime = stopTime - startTime
    formatedTime = "{:.5f}".format(elapsedTime)
    print(f"A* case {case} took {formatedTime}s")
    print(states1)

    # writer1.writerow([case, formatedTime, formatedTime2, states1, states2, len(path1), len(path2) ])

    # for run in range(RUNS):
    #     for budget in range(len(BUDGET)):
    #         for cp in range(len(CP)):
    #             startTimemcts = time.time()
    #             tree = MTCS(tempCube, goalCube, BUDGET[budget], CP[cp], h2)
    #             path = solvedMTCS(tree, goalCube.state)
    #             print(path)
    #             if path:
    #                 print(
    #                     f"Path found for run {run} using budget={BUDGET[budget]} and cp={CP[cp]} \n"
    #                 )
    #                 path.reverse()
    #                 for p in path:
    #                     print(tuple(p))
    #             else:
    #                 print(
    #                     f"No path found for run {run} using budget={BUDGET[budget]} and cp={CP[cp]}"
    #                 )
    #             stopTimemcts = time.time()
    #             elapsedTimemcts = stopTimemcts - startTimemcts
    #             formatedTimemcts = "{:.5f}".format(elapsedTimemcts)
    #             print(f"case {case} took {formatedTimemcts}s \n")
    #             if path is not None:
    #                 writer2.writerow([case, formatedTimemcts, budget, len(path)])
    #             else:
    #                 writer2.writerow([case, formatedTimemcts, budget, 0])

    # Algoritm BFS
    # startTime2 = time.time()
    # path2, states2 = bidirectionalbfs(tempCube, goalCube)
    # stopTime2 = time.time()
    # print(len(path2))
    # elapsedTime2 = stopTime2 - startTime2
    # formatedTime2 = "{:.5f}".format(elapsedTime2)
    # print(f"BFS case {case} took {formatedTime2}s")

    # writer2.writerow([case, states1, states2])
    # writer3.writerow([case, len(path1), len(path2)])

f1.close()
f2.close()
f3.close()


# test 1 case
# tempCube = Cube(moves=case1, scrambled=False)
# startTime = time.time()
# path = astar(tempCube, goalCube, h1)
# path = bidirectionalbfs(tempCube, goalCube)
# fig, ax = plt.subplots(figsize=(7, 5))
# for p in path:
#     ax.clear()
#     p.render(ax)
#     plt.pause(0.5)
# tree = mtcs(tempCube, goalCube, BUDGET[3], CP[1], h1)
# path = solvedStates(tree, goalCube.state)
# if path:
#     print(path)
# else:
#     print("No path found")
# print(path)
# stopTime = time.time()
# elapsedTime = stopTime - startTime
# print(f"case {case1} took {elapsedTime}s")

# test solvedMTCS for a random node
# tree = initNode(state=testCube.state)
# tree[Q] = 100
# for action in range(6):
#     tree[ACTIONS][action] = initNode(state=testCube.state, parent=tree)
#     tree[ACTIONS][action][Q] = randint(1, 40)

#     for child in range(3):
#         if action != 4:
#             tree[ACTIONS][action][ACTIONS][child] = initNode(
#                 state=testCube.state, parent=tree[ACTIONS][action]
#             )
#             tree[ACTIONS][action][ACTIONS][child][Q] = randint(1, 20)
#         elif child != 1:
#             tree[ACTIONS][action][ACTIONS][child] = initNode(
#                 state=testCube.state, parent=tree[ACTIONS][action]
#             )
#             tree[ACTIONS][action][ACTIONS][child][Q] = randint(1, 20)
#         else:
#             tree[ACTIONS][action][ACTIONS][child] = initNode(
#                 state=goalCube.state, parent=tree[ACTIONS][action]
#             )
#             tree[ACTIONS][action][ACTIONS][child][Q] = randint(1, 20)
# path = solvedMTCS(tree, goalCube.state)
# if path:
#     path.reverse()
#     pathList = []
#     for p in path:
#         pathList.append(tuple(p))
#     print(pathList)
# else:
#     print("No path found")
