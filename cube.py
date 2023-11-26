from __future__ import annotations

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import animation

from constants import MOVES, CORNERS, COLORS, LETTERS
from moves import Move, MoveInput, MoveSequence

import matplotlib.pyplot as plt
import numpy as np
from heapq import heappop, heappush
import time
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


class Cube:
    def __init__(self, moves: Moves | None = None, scrambled: bool = True):
        self.goal_state = np.repeat(np.arange(6), 4)
        self.state = np.repeat(np.arange(6), 4)

        if moves or scrambled:
            self.scramble(moves)

    def scramble(self, moves: Moves | None = None):

        if moves is None:
            num_of_moves = np.random.randint(5, 11)
            moves = list(np.random.randint(len(MOVES), size=num_of_moves))

        self.state = Cube.move_state(self.state, moves)

    def move(self, move: Moves) -> Cube:
        cube = Cube()
        cube.state = Cube.move_state(self.clone_state(), move)
        return cube

    @staticmethod
    def move_state(state: np.ndarray, move: Moves) -> np.ndarray:
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

    def hash(self) -> str:
        return Cube.hash_state(self.state)

    def __lt__(self, other):
        pass

    @staticmethod
    def hash_state(state: np.ndarray) -> str:
        return "".join(map(str, state))

    @staticmethod
    def _draw_corner(ax, position, colors):

        vertices = (
            np.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [1, 1, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 1],
                    [1, 1, 1],
                    [0, 1, 1],
                ]
            )
            + position
        )

        indices = [
            (0, 1, 2, 3),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (2, 3, 7, 6),
            (0, 3, 7, 4),
            (1, 2, 6, 5),
        ]

        faces = [[vertices[idx] for idx in face] for face in indices]

        ax.add_collection3d(
            Poly3DCollection(faces, facecolors=colors, linewidths=1, edgecolors="black")
        )

    @staticmethod
    def _draw_cube(state: np.ndarray, ax):

        for corner, (state_idxs, color_idxs) in CORNERS.items():
            colors = ["gray"] * 6

            for sticker_idx, color_idx in zip(state_idxs, color_idxs):
                colors[color_idx] = COLORS[state[sticker_idx]]

            Cube._draw_corner(ax, corner, colors)

    @staticmethod
    def render_state(state, ax):
        base_coords = np.array([(0, 1), (1, 1), (0, 0), (1, 0)])
        offsets = np.array([[0, 0], [1, 0], [2, 0], [-1, 0], [0, 1], [0, -1]]) * 2

        idx = 0

        for offset in offsets:
            for coords in base_coords:
                rect = plt.Rectangle(
                    coords + offset, 1, 1, edgecolor="black", linewidth=1
                )
                rect.set_facecolor(COLORS[state[idx]])
                ax.add_patch(rect)

                idx += 1

        ax.set_xlim(-2.1, 6.1)
        ax.set_ylim(-2.1, 4.1)
        ax.axis("off")
        # plt.show()

    def render(self, ax):
        Cube.render_state(self.state, ax)

    def render3D(self):

        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection="3d")

        Cube._draw_cube(self.state, ax)

        ax.axis("off")
        ax.set_xlim([0, 2])
        ax.set_ylim([0, 2])
        ax.set_zlim([0, 2])
        plt.show()

    @staticmethod
    def render3D_moves(
        initial_state: np.ndarray, moves: MoveSequence, save: bool = False
    ):
        moves = Move.parse(moves)

        original_state = np.copy(initial_state)
        state = initial_state

        fig = plt.figure(figsize=(4, 4), frameon=False)
        ax = fig.add_subplot(111, projection="3d")

        Cube._draw_cube(state, ax)

        ax.axis("off")
        ax.set_xlim([0, 2])
        ax.set_ylim([0, 2])
        ax.set_zlim([0, 2])

        move_index = 0

        def init():

            Cube._draw_cube(state, ax)
            return ax

        def animate(i):
            nonlocal move_index

            if i == 0:  # For the initial frame, show the original state
                state[:] = np.copy(original_state)
                Cube._draw_cube(state, ax)

            else:
                if move_index < len(moves):  # Check if there are more moves to perform
                    state[:] = Cube.move_state(state, moves[move_index])
                    ax.clear()

                    Cube._draw_cube(state, ax)
                    move_index += 1
                    ax.axis("off")
                    ax.set_xlim([0, 2])
                    ax.set_ylim([0, 2])
                    ax.set_zlim([0, 2])
                else:

                    move_index = 0
                    state[:] = np.copy(original_state)
                    Cube._draw_cube(state, ax)

        ani = animation.FuncAnimation(
            fig,
            animate,
            frames=len(moves) + 2,
            init_func=init,
            interval=1000,
            blit=False,
        )

        if save:
            ani.save("rubiks_cube_animation.gif", writer="pillow", fps=1)

        plt.show()
        return ani

    def render_text(self):
        lines = [
            [None, None, 16, 17],
            [None, None, 18, 19],
            [12, 13, 0, 1, 4, 5, 8, 9],
            [14, 15, 2, 3, 6, 7, 10, 11],
            [None, None, 20, 21],
            [None, None, 22, 23],
        ]

        for line in lines:
            print(
                "".join(
                    LETTERS[self.state[idx]] if idx is not None else " " for idx in line
                )
            )


# Sum collum diffrences
def h1(scramble: np.np.ndarray, goal: np.np.ndarray):
    cost = 0
    for idx, val in enumerate(scramble):
        cost += abs(val - goal[idx])
    return cost


def applyAllMoves(scramble: Cube()):
    cubeList = []
    for i in range(6):
        tempCube = scramble.move(i)
        cubeList.append(tempCube)
    return cubeList


def initNode(parent=None, state=None):
    return {N: 0, Q: 0, EXPLORED: [], STATE: state, PARENT: parent, ACTIONS: {}}


# Funcție ce alege o acțiune dintr-un nod
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


def solvedStates(tree, goal: np.ndarray):
    node = tree

    while not np.all(np.equal(node[STATE], goal)) and not all(
        explored in tree[EXPLORED] for explored in list(tree[ACTIONS])
    ):
        actionList = list(node[ACTIONS])
        if actionList:
            tempMin = 999
            bestAction = 0
            for action in actionList:
                if node[ACTIONS][action][Q] < tempMin and action not in node[EXPLORED]:
                    tempMin = node[ACTIONS][action][Q]
                    bestAction = action
            node[EXPLORED].append(bestAction)
            node = node[ACTIONS][bestAction]
        else:
            node = node[PARENT]

        # if bestAction in node[ACTIONS][bestAction]:
        #     node = node[ACTIONS][bestAction]

    return node


# def solvedStates(tree, goal: np.ndarray):
#     frontier = []
#     heappush(frontier, (tree[Q], tree))
#     node = tree
#     discovered = {tuple(node[STATE]): (None, node[Q])}
#     while frontier:
#         currNode = heappop(frontier)
#         currNode = currNode[1]
#         if np.all(np.equal(currNode[STATE], goal)):
#             break
#         actionList = list(node[ACTIONS])
#         for action in actionList:
#             tempCost = discovered[tuple(currNode[STATE])][1] + 1
#             if tuple(currNode[ACTIONS][action][STATE]) not in discovered:
#                 discovered[tuple(currNode[ACTIONS][action][STATE])] = (
#                     currNode,
#                     tempCost,
#                 )
#                 heappush(
#                     frontier,
#                     (
#                         tempCost + currNode[ACTIONS][action][Q],
#                         currNode[ACTIONS][action],
#                     ),
#                 )
#         break


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
    return path


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

    return path


def mtcs(scrambled: Cube(), goal: Cube(), budget, cp, h):
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


case1 = "R U' R' F' U"
case2 = "F' R U R U F' U'"
case3 = "F U U F' U' R R F' R"
case4 = "U' R U' F' R F F U' F U U"
caseList = [case1, case2, case3, case4]
goalCube = Cube(scrambled=False)
testCube = Cube(moves=[0], scrambled=False)
# test all cases
# for case in caseList:
#     tempCube = Cube(moves=case, scrambled=False)
#     startTime = time.time()
#     # path = astar(tempCube, goalCube, h1)
#     # path = bidirectionalbfs(tempCube, goalCube)
#     # fig, ax = plt.subplots(figsize=(7, 5))
#     # for p in path:
#     #     ax.clear()
#     #     p.render(ax)
#     #     plt.pause(0.5)
#     (action, tree) = mtcs(tempCube, goalCube, 10, CP[1], h1)
#     print(action)
#     if tree:
#         print_tree(tree[PARENT])
#     stopTime = time.time()
#     elapsedTime = stopTime - startTime
#     print(f"case {case} took {elapsedTime}s")

# test 1 case
tempCube = Cube(moves=case1, scrambled=False)
startTime = time.time()
# path = astar(tempCube, goalCube, h1)
# path = bidirectionalbfs(tempCube, goalCube)
# fig, ax = plt.subplots(figsize=(7, 5))
# for p in path:
#     ax.clear()
#     p.render(ax)
#     plt.pause(0.5)
tree = mtcs(tempCube, goalCube, 10, CP[1], h1)
path = solvedStates(tree, goalCube.state)
# print(path)
stopTime = time.time()
elapsedTime = stopTime - startTime
print(f"case {case1} took {elapsedTime}s")
