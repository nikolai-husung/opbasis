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
      via a call to `reconstructH4GroupElems`, which takes as input
      integer-valued matrices and then iteratively reconstructs those matrices.

2. Obtain a `Projector` into any of the various (irreducible) representations
   of your group. This requires as input the characters as well as the
   reconstructed conjugacy classes together with the chosen map.

3. For irreducible representations (irrep) of dimension larger 1 there exists 
   the option to further break down the operators into different so called
   *rows*. To obtain such a `RowProjector`, a non-vanishing `LinearComb`
   **already projected** to the irrep must be provided to `constructDreps`.
   The resulting representation matrices can be reused for any operator already
   projected to the irrep to ensure the same conventions on the (otherwise
   equivalent) choices of *rows*.

   .. caution::
      For operators of higher canonical mass-dimension multiple copies of the
      same irrep may be realised leading to a failure of the procedure. A check
      has been build in to catch this case. Recommendation: Use the operator
      with lowest canonical mass-dimension as a *seed*.
   

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

from fractions import Fraction

from .calculus import Complex, Matrix, solve
from .ops import LinearComb
from .minBasis import _toRep, _mapToVectors, _mapCoefficients

         

# Intentionally omit the identity!
_reverseMap = {
   "H4_P0" : Matrix.diag([-1,1,1,1]),
   "H4_P1" : Matrix.diag([1,-1,1,1]),
   "H4_P2" : Matrix.diag([1,1,-1,1]),
   "H4_P3" : Matrix.diag([1,1,1,-1]),
   "H4_R0" : Matrix([[0,1,0,0],[-1,0,0,0],[0,0,1,0],[0,0,0,1]]),
   "H4_R1" : Matrix([[0,0,1,0],[0,1,0,0],[-1,0,0,0],[0,0,0,1]]),
   "H4_R2" : Matrix([[0,0,0,1],[0,1,0,0],[0,0,1,0],[-1,0,0,0]]),
   "H4_R3" : Matrix([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]]),
   "H4_R4" : Matrix([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,-1,0,0]]),
   "H4_R5" : Matrix([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,-1,0]])
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
def reconstructH4GroupElems(elems:list[Matrix]):
   r"""
   Reconstructs every element of *elems* in terms of the transformations of
   the hypercubic group including inversions as they are implemented in the
   package.

   .. tip::
      The general strategy employed here should be feasible for other groups
      by representing them in their matrix representation.

   Parameters
   ----------
   elems : list[Matrix]
      All group elements of the chosen (subgroup of the) hypercubic group with
      reflections, i.e., the literal :math:`4\times 4` matrices.

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
   id_ = Matrix.diag([1,1,1,1])
   idx = [None]*len(elems)
   for i,elem in enumerate(elems):
      if elem==id_: idx[i] = []
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
            if elem==r:
               idx[i] = refId[j]
               missing -= 1
               break
      temp = []
      tempId = []
      old.extend(ref)
      for id_,r in _reverseMap.items():
         for _newId, _new in zip(map(lambda x: x + [id_], refId),
                                 map(lambda x: r@x, ref)):
            if not _new in temp and not _new in old:
               temp.append(_new)
               tempId.append(_newId)
      ref = temp
      refId = tempId
      n += 1
   if n<NMAX:
      return idx
   raise ValueError("Check your input group elements. They could not be "
                    "identified within %i iterations."%NMAX)


class Projector:
   r"""Defines a new projector onto a representation specified by *chars* of
   the chosen group.

   Each group element must be identified with a chain of transformations acting
   on the `LinearComb` to be projected. The user has to provide a map
   *mapToTransf* of identifiers to the actual transformations, which can then
   be used to describe the various group elements in each conjugacy class.

   Naming and providing the transformations is up to the user unless he relies
   on the built-in `H4map`, that can be used to map the full hypercubic group
   with reflections into the built-in transformations.

   The projector uses the standard projection formula

   .. math::
      \mathop{\mathrm{proj}}_{\chi}\,O = \frac{d}{|G|}\sum_{C}\chi(C)\sum_{g\in C} [g\circ O]

   where :math:`[g\circ O]` denotes the transformation of :math:`O` and
   :math:`\chi` are the chosen characters for the different conjugacy classes.

   Parameters
   ----------
   d : int
      Dimension of the representation.
   chars : list[Complex|int|Fraction]
      Characters specifying the chosen representation.
   cclasses : list[list[int|str]]
      Collections of chains of identifiers resolved in *map2transf* for each
      element of each conjugacy class.
   map2transf : dict[int|str,Callable[[LinearComb],LinearComb]], optional
      Translates the identifiers used to describe each group element in terms
      of accessible transformations. The details of the map must be provided by
      the user. Defaults to H4map.
   """
   def __init__(self, d, chars, cclasses:list[list[int|str]],
      mapToTransf:dict[int|str,Callable[[LinearComb],LinearComb]] = H4map):
      self.cclasses = cclasses
      self.mapToTransf = mapToTransf
      self.chars = chars
      self.dim = d

   def __str__(self):
      """
      Returns a string containing the relevant information about the projector.
      """
      return """Projector onto irrep
====================

Dimension of group |G|: %i
Dimension of irrep:     %i
Characters:
%r
"""%(sum(map(len, self.cclasses)), self.dim, self.chars)

   def __call__(self, ansatz:LinearComb):
      r"""Projects the *ansatz* to the chosen representation.

      An operator transforming in a specific representation - not necessarily
      an irreducible one - is fully described by the characters *chars*, the
      conjugacy classes *cclasses*, and the *ansatz* on which to perform the
      projection according to the formula

      .. math::
         \mathop{\mathrm{proj}}_{\chi}\,O=\frac{d}{|G|}\sum_{C}\chi(C)\sum_{g\in C} [g\circ O]

      where :math:`[g\circ O]` denotes the transformation of :math:`O` and
      :math:`\chi` are the chosen characters for the different conjugacy
      classes.

      Parameters
      ----------
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
      for ic in range(len(self.chars)):
         if self.chars[ic] == 0: continue
         for elem in self.cclasses[ic]:
            temp = _copy(ansatz)
            for it in elem:
               temp = self.mapToTransf[it](temp)
            projected += self.chars[ic]*temp
      projected.simplify()
      return projected * Fraction(self.dim, sum(map(len, self.cclasses)))


def constructDreps(seed:LinearComb, d:int, groupElements:list[int|str],
   _map:dict[int|str,Callable] = H4map, ghat:list[int] = None):
   r"""
   Derives the :math:`d\times d`-dimensional representation matrices 
   :math:`D(h)` for the multi-dimensional irrep identified by the already
   projected `LinearComb` *seed* via the following strategy:
   
   1. Act separately with the elements of the group on the *seed* and identify
      *d* linearly independent variants. This fixes our basis :math:`\hat{g}`
      to derive :math:`D(h)` within the chosen irrep.
   2. From the identity
   
      .. math::
         [h\circ [\hat{g}_i \circ O]] = [D(h)]_{ji} [\hat{g}_j \circ O]\,\forall h\in G

      we obtain a linear system of equations. By solving that system we then
      obtain the representation matrices for each :math:`h\in G` for the
      selected irrep. These matrices only depend on the irrep and thus can be
      reused for other operators already projected to the same irrep.

   Parameters
   ----------
   seed : LinearComb
      A **non-vanishing** projection of an operator into the chosen irrep. Will
      be used for constructing a basis.

      .. hint::
         Use the `LinearComb` with lowest canonical mass-dimension available to
         avoid having various copies of the same irrep present in your operator
         leading to
         
         .. math:
            \mathop{\mathrm{rank}}(v_1,\ldots,v_n)=md

         with :math:`m>1` and where :math`v_i` are the vectors representing the
         partner functions.

   d : int
      Dimension of the chosen irrep.
   groupElements : list[int|str]
      All the group elements of the group expressed in terms of the labels used
      in *_map*.
   _map : dict[int|str,Callable], optional
      Maps the labels used to represent the group elements into the actual
      transformations acting on `LinearComb`. Defaults to `H4map`.
   ghat : list[int], optional
      Indices specifying the group elements to be used for building the basis
      if desired or needed for reproducibility. Defaults to `None`.

   Returns
   -------
   tuple[list[Matrix],list[int]]
      The full collection of the representation matrices and the indices of the
      specific group elements used to derive them (for reproducibility).

   Raises
   ------
   ValueError
      Indicating that there is a problem with the number of available
      linearly independent basis vectors.
   """
   # build in check that dim span(R(g) O)=d!
   _seeds = []
   for g in groupElements:
      temp = _copy(seed)
      for transf in g:
         temp = _map[transf](temp)
      temp.simplify()
      _seeds.append(temp)

   evecs,idxMap = _mapToVectors([_seeds])

   rank = Matrix(evecs[0]).transpose().rank()
   if rank > d:
      raise ValueError("Choose an operator of lower canonical mass-dimension "
         "to derive the representation matrices: dim(span(R(g)O)) = %i > %i."%(
         rank,d))
   elif rank < d:
      raise ValueError(
         "Could not find %i linearly independent basis vectors."%d)

   if ghat is None:
      evecsIndep = Matrix()
      V = Matrix()
      lastRank = 0
      _ghat = list()
      ghat = list()
      for i,evec in enumerate(evecs[0]):
         # This should never happen!
         if evec is None: continue
         evecsIndep.extend(evec, Matrix.ROW)
         newrank = evecsIndep.rank(True)
         if lastRank<newrank:
            lastRank += 1
            V.extend(evec, Matrix.ROW)
            _ghat.append(_seeds[i])
            ghat.append(i)
         if lastRank==d: break
   else:
      _ghat = [_seeds[i] for i in ghat]

      V = Matrix([evecs[0][i] for i in ghat])
      if V.rank()<d:
         raise ValueError("The proposed basis does not contain %i linearly "
            "independent basis vectors."%d)
   Vdag = V.conjugate()
   A = (Vdag @ V.transpose()).inverse()

   Dreps = list()
   for h in groupElements:
      # compute Drep(h)
      rhs = list()
      for p in _ghat:
         temp = _copy(p)
         for transf in h:
            temp = _map[transf](temp)
         rhs.append(_mapCoefficients(_toRep(temp), idxMap))
      Dreps.append((A @ (Vdag @ Matrix(rhs).transpose())).conjugate())
      # do we need the inverse here? Drep.inverse()
   return Dreps, ghat


class RowProjector:
   """Implements a projector onto different rows of a multidimensional irrep.

   For the derivation of the representation matrices see `constructDreps`.

   Conceptually, all choices for how to define these rows are equivalent. Here,
   an arbitrary choice is made for this definition according to the first *d*
   linearly independent basis vectors build from the *seed* operator.

   .. tip::
      For other choices of the basis vectors and thus rows one has to either
      change the order of *groupElements* or dictate the basis via handing a
      `list` of length *d* as *seed*.
   """
   def __init__(self, seed:LinearComb, d:int, groupElements:list[LinearComb],
      mapToTransf:dict[int|str,Callable[[LinearComb],LinearComb]] = H4map,
      unitary:bool = False, ghat:list[int] = None):
      r"""
      Derives all the representation matrices :math:`D(h)\forall h\in G` in
      the irrep chosen by *seed* and stores the details to be used when
      row-projecting any operators from this irrep.
      """
      self.groupElements = groupElements
      self.mapToTransf = mapToTransf
      self.dim = d
      self.seed = seed
      self.Dreps, self.ghat = constructDreps(seed, d, self.groupElements,
                                             mapToTransf, ghat)

   def __str__(self):
      return """RowProjector onto the %i rows of the irrep
=========================================

Dimension of group |G|:  %i
Irrep dimension:         %i
Chosen basis:            %s
Projected seed operator:
%s
"""%(self.dim, len(self.groupElements), self.dim,
       ", ".join("g_%i"%g for g in self.ghat), str(self.seed))

   def __call__(self, projected:list[LinearComb]):
      r"""
      Applies the projection formula

      .. math::
         O_i = \frac{d}{|G|} \sum_{g\in G} D_{ii}(g) [g\circ O]

      where :math:`D(g)` is a matrix derived in `constructDreps` and :math:`O`
      is **required** to be already projected into the chosen irrep.

      Parameters
      ----------
      projected : LinearComb
         An operator **already projected** into the chosen irrep.

      Returns
      -------
      LinearComb
         Projection into the *d* rows of the chosen irrep.
      """
      rows = [LinearComb.ZERO() for r in range(self.dim)]
      for ie, elem in enumerate(self.groupElements):
         #for ip,ipe in enumerate(self.ghat):
            temp = _copy(projected)
            for it in elem: 
               temp = self.mapToTransf[it](temp)
            for r in range(self.dim):
               rows[r] += self.Dreps[ie][r,r] * temp
      fac = Fraction(self.dim, len(self.groupElements))
      for r in range(self.dim):
         rows[r].simplify()
         rows[r] *= fac
      return rows
