"""Chromosome encoding for GA client-count search."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Chromosome:
    """Single GA candidate encoding client count K.

    Attributes:
        K: Number of clients selected per round (decision variable).
        fitness: Evaluated fitness score; None until evaluated.
    """

    K: int
    fitness: float | None = None
