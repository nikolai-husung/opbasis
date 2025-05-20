import sys
MIN_PYTHON = (3,10)
assert sys.version_info >= MIN_PYTHON, "Requires Pyton %s.%s or"%MIN_PYTHON \
   + " higher due relying on __annotations__ and frozen order in dict."
del sys, MIN_PYTHON

from .basics import rotPlanes, Complex, CustomIndex, IntIndex, sptIdx, \
   indices, defaultIndices, massDim
from .blocks import Block, d, D0l, D0, Dl, D, DF, F, M, dM,\
   Colour, Multiplicative
from .dirac import Dirac, axisGammas, Gamma
from .eoms import GluonEOM, unmaskGluonEOMs, unmaskFermionEOMs
from .io import parseLinearCombs, CompressedBasis
from .minBasis import findMinBases
from .opBasis import Model, overcompleteBasis, symmetrise, parseAnsatz
from .ops import LinearComb, Commutative, Trace, Bilinear
from .pauli import Pauli, SU2
from .templates import TemplateRep, getBilinearTemplates,\
   getAlgebraTraceTemplates, getTemplates
