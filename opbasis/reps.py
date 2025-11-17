"""
This submodule adds convenient functions to extend the scope of the package to
representations, both reducible and irreducible ones. It is the user's
responsibility to provide the necessary group theory input such as characters
and (a mapping of the conjugacy classes into) the appropriate transformations
in a format suitable for the package.

The intended use is as follows:

0. Identify your group: Obtain the character table and the corresponding
   conjugacy classes including a collection of all their elements.

   .. hint::
      This step is not provided by the package.

1. Map the elements of your conjugacy classes into transformations acting on a
   `LinearComb`. Typically, this will involve an iteration over various
   transformations to match the specific element. Internally it is handled via
   a map of labels to some transformation that is expected to be callable with
   only a `LinearComb` as an argument. With help of that map, one may then
   represent each transformation as a collection of labels for the different
   transformations. For an example see `H4map`, which already implements such a
   map for the built-in transformations on the hypercubic group with
   reflections.

   .. tip::
      To make use of `H4map` (or any other map) one still has to work out the
      representation of the group elements in terms of those labels. For any
      subgroup of the hypercubic group with reflections this can be achieved
      via a call to `H4mapConjClasses` or `reconstructH4GroupElems`, which take
      as input integer-valued matrices (in the first case grouped into the
      specific conjugacy classes) and then iteratively reconstruct those
      matrices.

2. Obtain a projector into the various (irreducible) representations of your
   group via a call to `createProjector`, which expects as input the
   reconstructed conjugacy classes together with the chosen map.
   

Attributes
----------
H4map : dict[str,Callable[[LinearComb],LinearComb]]
   Labels built-in transformations of the hypercubic group with reflections
   to use them when reconstructing group elements in terms of those
   transformations.

   ::

      {
         "H4_R0" : lambda x: x.rotation(0),
         ...,
         "H4_P0" : lambda x: x.reflection(0),
         ...
      }

   where ``x`` is a `LinearComb`.
"""
from copy import deepcopy as _copy
from collections.abc import Callable

import numpy as np

from .basics import Complex, Fraction
from .ops import LinearComb


# Intentionally omit the identity!
_reverseMap = {
   "H4_P0" : np.diag([-1,1,1,1]),
   "H4_P1" : np.diag([1,-1,1,1]),
   "H4_P2" : np.diag([1,1,-1,1]),
   "H4_P3" : np.diag([1,1,1,-1]),
   "H4_R0" : np.array([[0,1,0,0],[-1,0,0,0],[0,0,1,0],[0,0,0,1]]),
   "H4_R1" : np.array([[0,0,1,0],[0,1,0,0],[-1,0,0,0],[0,0,0,1]]),
   "H4_R2" : np.array([[0,0,0,1],[0,1,0,0],[0,0,1,0],[-1,0,0,0]]),
   "H4_R3" : np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]]),
   "H4_R4" : np.array([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,-1,0,0]]),
   "H4_R5" : np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,-1,0]])
}
H4map = {
   "H4_P0" : lambda x: x.reflection(0),
   "H4_P1" : lambda x: x.reflection(1),
   "H4_P2" : lambda x: x.reflection(2),
   "H4_P3" : lambda x: x.reflection(3),
   "H4_R0" : lambda x: x.rotation(0),
   "H4_R1" : lambda x: x.rotation(1),
   "H4_R2" : lambda x: x.rotation(2),
   "H4_R3" : lambda x: x.rotation(3),
   "H4_R4" : lambda x: x.rotation(4),
   "H4_R5" : lambda x: x.rotation(5)
}


# NMAX is fixed by the full group!
NMAX = 10
def reconstructH4GroupElems(elems:list):
   """
   Reconstructs every element of *elems* in terms of the transformations of
   the hypercubic group with inversion.

   Returns
   -------
   list[list[str]]
      Reconstruction of each element in *elems* in terms of labels matching
      the keys of `H4map`.

   Raises
   ------
   ValueError
      Indicates that the matrices provided in *elems* are no elements of the
      hypercubic group with inversions and thus could not be reconstructed.
   """
   id_ = np.diag([1,1,1,1])
   idx = [None]*len(elems)
   for i,elem in enumerate(elems):
      if np.all(elem==id_): idx[i] = []
   refId,ref = zip(*_reverseMap.items())
   refId = list(map(lambda x: [x], refId))
   ref = list(ref)
   missing = sum(int(x is None) for x in idx)
   n = 0
   old = [id_]
   while n<NMAX and missing>0:
      for i,elem in enumerate(elems):
         if not idx[i] is None: continue
         for j,r in enumerate(ref):
            if np.all(elem==r):
               idx[i] = refId[j]
               missing -= 1
               break
      temp = []
      tempId = []
      old.extend(ref)
      for id_,r in _reverseMap.items():
         for _newId, _new in zip(map(lambda x: x + [id_], refId),
                                 map(lambda x: r@x, ref)):
            if not any(map(lambda x: np.all(x==_new), temp)) and \
               not any(map(lambda x: np.all(x==_new), old)):
               temp.append(_new)
               tempId.append(_newId)
      ref = temp
      refId = tempId
      n += 1
   if n<NMAX:
      return idx
   raise ValueError("Check your input group elements. They could not be "
                    "identified within %i iterations."%NMAX)


def H4mapConjClasses(cclasses:list[list]):
   """
   Turns a collection of group elements grouped into conjugacy classes into
   labels of the H4map to reconstruct them as transformations on H4.

   Returns
   -------
   list[list[str]]
      Reconstruction prescription in terms of transformations of H4 still
      grouped into conjugacy classes.
   """
   ranges = [len(x) for x in cclasses]
   allElems = sum(cclasses,start=[])
   idx = reconstructH4GroupElems(allElems)
   mapped = []
   for i in range(len(ranges)):
      mapped.append(idx[:ranges[i]])
      idx = idx[ranges[i]:]
   return mapped


def createProjector(cclasses:list[list[int|str]],
   mapToTransf:dict[int|str,Callable[[LinearComb],LinearComb]] = H4map):
   r"""Defines a new projector onto any representation of the chosen group.

   Each group element must be identified with a chain of transformations
   acting on the `LinearComb` to be projected. The user has to provide a map
   *mapToTransf* of identifiers to the actual transformations, which can then
   be used to describe the various group elements in each conjugacy class.

   Naming and providing the transformations is up to the user unless he relies
   on the built-in `H4map`, that can be used to map the full hypercubic group
   with reflections into the built-in transformations.

   The projector being returned uses the standard projection formula

   .. math::
      \mathop{\mathrm{proj}}_{\chi}\,O = \frac{1}{|G|}\sum_{C}\chi(C)\sum_{g\in C} [g\circ O]

   where :math:`[g\circ O]` denotes the transformation of :math:`O` and
   :math:`\chi` are the chosen characters for the different conjugacy classes.

   Parameters
   ----------
   cclasses : list[list[int|str]]
      Collections of chains of identifiers resolved in *map2transf* for each
      element of each conjugacy class.
   map2transf : dict[int|str,Callable[[LinearComb],LinearComb]], optional
      Translates the identifiers used to describe each group element in terms
      of accessible transformations. The details of the map must be provided
      by the user. Defaults to H4map.

   Returns
   -------
   Callable[[list[Complex|int|Fraction],LinearComb],LinearComb]
      The projector using *cclasses* in combination with *mapToTransf* when
      working out the projection to any representation chosen by the
      characters.
   """
   def projToRep(chars:list[Complex|int|Fraction], ansatz:LinearComb):
      r"""Projects the *ansatz* to the chosen representation.

      An operator transforming in a specific representation - not necessarily
      an irreducible one - is fully described by the characters *chars*, the
      conjugacy classes *cclasses*, and the *ansatz* on which to perform the
      projection according to the formula

      .. math::
         \mathop{\mathrm{proj}}_{\chi}O=\frac{1}{|G|}\sum_{C}\chi(C)\sum_{g\in C} [g\circ O]

      where :math:`[g\circ O]` denotes the transformation of :math:`O` and
      :math:`\chi` are the chosen characters for the different conjugacy
      classes.

      Parameters
      ----------
      chars : list[Complex|int|Fraction]
         Characters specifying the chosen representation.
      ansatz : LinearComb
         This can be any `LinearComb`. Typically, it amounts to a seed, when
         trying to find an operator transforming in a given representation. It
         may also be used to act with the projection onto a complete candidate
         to check if (parts) indeed transform in a specific representation.
   
      Returns
      -------
      LinearComb|list[LinearComb]
         Projection of input *ansatz*.
      """
      projected = LinearComb.ZERO()
      for ic in range(len(chars)):
         if chars[ic] == 0: continue
         for elem in cclasses[ic]:
            temp = _copy(ansatz)
            for it in elem:
               temp = mapToTransf[it](temp)
            projected += chars[ic]*temp
      projected.simplify()
      return projected * Fraction(1, sum(map(len, cclasses)))
   return projToRep
