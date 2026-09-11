"""BCR two-stage MIL models.

Thin re-export of the encoders used by the BCR pipeline:
  * stage 1: ``RegionEncoder``
  * stage 2: ``HierarchicalEncoder``, ``HierarchicalEncoderNuclei``

The encoder definitions live in ``common.layers``; this module re-exports them
for convenient import from the task package.
"""
from __future__ import annotations

from common import HierarchicalEncoder, HierarchicalEncoderNuclei, RegionEncoder

__all__ = ["RegionEncoder", "HierarchicalEncoder", "HierarchicalEncoderNuclei"]
