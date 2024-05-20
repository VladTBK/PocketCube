### Pocket Cube Solver
This project focuses on solving the Pocket Rubik's Cube (2x2x2) using various algorithms like A*, Bidirectional BFS, Monte Carlo Tree Search, and Pattern Database.
The primary goal is to understand these algorithms mechanics and determine which one yields the best results for solving this particular puzzle.

##### Overview
To facilitate testing of the algorithms in this project, a Cube class was created, represented by a matrix of 24 elements ranging from 1 to 5, each corresponding to the faces of the pocket cube. A cube is considered "completed" when the array is arranged as follows: [1, 1, 1, 1, 2, 2, 2, 2, ..., 5, 5, 5, 5]. While we know the end goal, we require a starting point. Therefore, two cubes are utilized: one serves as the general goal for the algorithms to aim for, while the other represents the starting configuration. The starting cube can either be randomly scrambled or a goal cube rotated by fixed positions. 
It's worth noting that the number of fixed rotations affects the difficulty of the solving process; the more rotations there are, the more challenging the solving becomes.

##### So witch one is better?
All the algorithms successfully solved the cube correctly. However, each algorithm has its strengths and weaknesses:
Certainly! Here are bullet points for each algorithm:

- **A***:
  - Strengths:
    - Guarantees an optimal solution if an admissible heuristic is used.
    - Had one of the worst solution times among the algorithms tested.
  - Weaknesses:
    - Can be computationally intensive for large search spaces.
    - Solution quality highly depends on the quality of the heuristic function witch for this particular project wasn't a perfect herustic. 2 pseudo-heuristics were used

- **Bidirectional BFS**:
  - Strengths:
    - Explores the search space from both the initial and goal states simultaneously, potentially reducing search time, witch was the lowest among the algoirthms tested.
    - Guarantees to find the shortest solution path.
  - Weaknesses:
    - Requires more memory compared to traditional BFS, witch was the second highest.

- **Monte Carlo Tree Search**:
  - Strengths:
    - Efficiently explores large search spaces through random sampling.
    - Adaptable to various problem domains without requiring domain-specific knowledge.
  - Weaknesses:
    - Solution quality highly depends on the number of simulations performed.
    - It's most used for 1 v 1 games, witch this wasn't, so was hard to implement and not optimized.

- **Pattern Database**:
- 
  - Strengths:
    - Utilizes precomputed pattern databases to guide the search process efficiently.
    - Guarantees to find an optimal solution in the shortest time.
  - Weaknesses:
    - Requires significant memory overhead for storing pattern databases. That's why only 2 * 6^7 states were stored. If a state wasn't found in the database, a heuristic was used. It should be noted that for a normal Rubik's Cube, this algorithm would not be efficient. 

# Contributing
