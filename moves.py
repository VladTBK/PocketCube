from __future__ import annotations

from typing import Union, List, TypeVar
from numbers import Number
from enum import Enum
import numpy as np

MOVES = np.array(
    [
        [
            0,
            21,
            2,
            23,
            6,
            4,
            7,
            5,
            19,
            9,
            17,
            11,
            12,
            13,
            14,
            15,
            16,
            1,
            18,
            3,
            20,
            10,
            22,
            8,
        ],  # R
        [
            0,
            1,
            14,
            15,
            4,
            5,
            2,
            3,
            8,
            9,
            6,
            7,
            12,
            13,
            10,
            11,
            16,
            17,
            18,
            19,
            22,
            20,
            23,
            21,
        ],  # F
        [
            2,
            0,
            3,
            1,
            18,
            5,
            19,
            7,
            8,
            9,
            10,
            11,
            12,
            20,
            14,
            21,
            16,
            17,
            15,
            13,
            6,
            4,
            22,
            23,
        ],  # U
        [
            0,
            17,
            2,
            19,
            5,
            7,
            4,
            6,
            23,
            9,
            21,
            11,
            12,
            13,
            14,
            15,
            16,
            10,
            18,
            8,
            20,
            1,
            22,
            3,
        ],  # R'
        [
            0,
            1,
            6,
            7,
            4,
            5,
            10,
            11,
            8,
            9,
            14,
            15,
            12,
            13,
            2,
            3,
            16,
            17,
            18,
            19,
            21,
            23,
            20,
            22,
        ],  # F'
        [
            1,
            3,
            0,
            2,
            21,
            5,
            20,
            7,
            8,
            9,
            10,
            11,
            12,
            19,
            14,
            18,
            16,
            17,
            4,
            6,
            13,
            15,
            22,
            23,
        ],  # U'
    ]
)

MoveInput = TypeVar("MoveInput", int, str, Enum)
MoveSequence = TypeVar("oveSequence", List[int], str, List[Enum])
Moves = TypeVar("Moves", MoveInput, MoveSequence)


class Move(Enum):

    R = 0
    F = 1
    U = 2
    Rp = 3
    Fp = 4
    Up = 5

    def opposite(self) -> Move:
        opposites = {
            Move.R: Move.Rp,
            Move.F: Move.Fp,
            Move.U: Move.Up,
            Move.Rp: Move.R,
            Move.Fp: Move.F,
            Move.Up: Move.U,
        }
        return opposites[self]

    @classmethod
    def from_str(cls, move_str: str) -> Move:
        move_dict = {
            "R": Move.R,
            "F": Move.F,
            "U": Move.U,
            "R'": Move.Rp,
            "F'": Move.Fp,
            "U'": Move.Up,
        }
        return move_dict[move_str]

    @classmethod
    def from_int(cls, move_int: Number) -> Move:
        if 0 <= move_int <= 5:
            return Move(move_int)
        else:
            raise ValueError(f"Invalid move {move_int}")

    @classmethod
    def parse(cls, move_input: Moves) -> Union[Move, List[Move]]:

        if isinstance(move_input, list):
            return [cls.parse(move) for move in move_input]

        if isinstance(move_input, Number):
            return cls.from_int(move_input)

        elif isinstance(move_input, str):
            if " " in move_input:
                return [cls.parse(move) for move in move_input.split(" ")]
            return cls.from_str(move_input)

    def __str__(self) -> str:
        move_str = {
            Move.R: "R",
            Move.F: "F",
            Move.U: "U",
            Move.Rp: "R'",
            Move.Fp: "F'",
            Move.Up: "U'",
        }
        return move_str[self]
