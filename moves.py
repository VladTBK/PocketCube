from __future__ import annotations

from typing import Union, List, TypeVar
from numbers import Number
from enum import Enum


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
            Move.Up: Move.U
        }
        return opposites[self]

    @classmethod
    def from_str(cls, move_str: str) -> Move:
        move_dict = {
            'R':  Move.R,
            'F':  Move.F,
            'U':  Move.U,
            "R'": Move.Rp,
            "F'": Move.Fp,
            "U'": Move.Up
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

        elif isinstance(move_input, Move):
            return move_input
        else:
            raise ValueError(f"Invalid move type: {move_input}")

    def __str__(self) -> str:
        move_str = {
            Move.R:  'R',
            Move.F:  'F',
            Move.U:  'U',
            Move.Rp: "R'",
            Move.Fp: "F'",
            Move.Up: "U'"
        }
        return move_str[self]
