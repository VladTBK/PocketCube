import numpy as np


__all__ = ['MOVES', 'CORNERS', 'COLORS']

"""
Sticker indices:
          -------
         | 16 17 |
         |       |
         | 18 19 |
  ------- ------- ------- -------
 | 12 13 | 00 01 | 04 05 | 08 09 |
 |       |   U   |   R   |       |
 | 14 15 | 02 03 | 06 07 | 10 11 |
  ------- ------- ------- -------
         | 20 21 |
         |   F   |
         | 22 23 |
          -------
"""

MOVES = np.array([
        [0, 21, 2, 23, 6, 4, 7, 5, 19, 9, 17, 11, 12, 13, 14, 15, 16, 1, 18, 3, 20, 10, 22, 8], # R
        [0, 1, 14, 15, 4, 5, 2, 3, 8, 9, 6, 7, 12, 13, 10, 11, 16, 17, 18, 19, 22, 20, 23, 21], # F
        [2, 0, 3, 1, 18, 5, 19, 7, 8, 9, 10, 11, 12, 20, 14, 21, 16, 17, 15, 13, 6, 4, 22, 23], # U
        [0, 17, 2, 19, 5, 7, 4, 6, 23, 9, 21, 11, 12, 13, 14, 15, 16, 10, 18, 8, 20, 1, 22, 3], # R'
        [0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13, 2, 3, 16, 17, 18, 19, 21, 23, 20, 22], # F'
        [1, 3, 0, 2, 21, 5, 20, 7, 8, 9, 10, 11, 12, 19, 14, 18, 16, 17, 4, 6, 13, 15, 22, 23], # U'
    ])

"""
The sides of an individual cube (used when rendering the whole
2x2x2 rubik's cube) are indexed as follows:

               +--------+              +z
              /        /|              ^     (1, 1, 1)
             /    1   / | 3            |
            +--------+  |              +-------> +x
            |        |5 +             /
          4 |    2   | /             /
            |        |/             v (0, 0, 0)
            +--------+             +y
                0

The following constant maps the corners from the 3D rendering
to the actual indices of the corners in the internal representation
of the 2x2x2 cube (which is a 1D array). In addition, for each of
the 8 corners of the rubik's cube, information related to which sides
of the corner are visible.

CORNERS = {
    (3D Coords): ([indices of the corner], [visible sides]),
    ...
}
"""

CORNERS = {
    (0, 0, 0): ([11, 14, 22], [0, 4, 2]), # BOT_BL
    (1, 0, 0): ([10, 23, 7 ], [0, 2, 5]), # BOT_BR
    (0, 1, 0): ([9,  12, 16], [0, 4, 3]), # BOT_TL
    (1, 1, 0): ([8,  5,  17], [0, 5, 3]), # BOT_TR

    (0, 0, 1): ([2,  15, 20], [1, 4, 2]), # TOP_BL
    (1, 0, 1): ([3,  6,  21], [1, 5, 2]), # TOP_BR
    (0, 1, 1): ([0,  13, 18], [1, 4, 3]), # TOP_TL
    (1, 1, 1): ([1,  4,  19], [1, 5, 3]), # TOP_TR
}

"""
Face colors:
    ┌──┐
    │ 4│
 ┌──┼──┼──┬──┐
 │ 3│ 0│ 1│ 2│
 └──┼──┼──┴──┘
    │ 5│
    └──┘
"""
COLORS = {
    0: "#003DA5",  # blue
    1: "#BA0C2F",  # red
    2: "#009B48",  # green
    3: "#FE5000",  # orange
    4: "#FFD700",  # yellow
    5: "white"
}


LETTERS = {
    0: 'B',
    1: 'R',
    2: 'G',
    3: 'O',
    4: 'Y',
    5: 'W',
}