"""
This submodule provides the basic implementaion of `Block` and its subclasses.
Insertions of (subclasses of) `Block` are used to implement covariant
derivatives, field-strength tensors, etc. including their default transformation
properties.

Custom implementations of `Block` then allow for extensions to models with non-
trivial transformation properties, e.g., due to more complicated flavour
symmetries.
"""

import re
from copy import deepcopy as _copy
from fractions import Fraction
from itertools import product

from abc import ABC, abstractmethod

from .basics import rotPlanes, Complex, dim, CustomIndex, IntIndex, sptIdx,\
   indices, defaultIndices, massDim

class Block:
   """
   Super class to derive from. Already implements default behaviour if
   sensible. Implements the transformations acting as the identity.
   => To be overwritten by appropriate non-trivial cases.
   
   If the instance of a Block carries indices (or a more general suffix), those
   details must be specified via a
   `decorator <https://docs.python.org/3/glossary.html#term-decorator>`_
   `@indices` or `@defaultIndices`, where the latter makes some implicit
   assumptions on how to read off the conventions used for the indices. This
   will set the class-attributes `__indexType__`, `__indexName__` and
   `__suffix__`  to some non-trivial values. For more details how to declare
   indices, see `indices`. **CAVEAT:** Do not modify those attributes directly!
   
   The name of the arguments passed to `@indices` has to match **exactly** the
   names of the class members used internally, i.e.,

   ::
   
      testIdx = CustomIndex("testIdx", ["a", "b"])
      @indices("(%s;%s)", mu = sptIdx, nu = testIdx)
      class MyBlock(Block):
         def __init__(self,x:sptIdx,y:testIdx,factor:int|Fraction|Complex=1):
            self.mu = x
            self.nu = y

   The **unique** types used in the keyword arguments matching the actual type
   of the index are mandatory as they are used to decide how the data is to be
   processed when parsing custom Block implementations from file. Currently
   there are two cases

   ::

      match = "123"
      if issubclass(f, CustomIndex):
         print(getattr(f,match))
      else:
         print(f(int(match)))

   **CAVEAT:** Deviating from this pattern will break the package! 

   Parameters
   ----------
   factor : int|Fraction|Complex, optional
      Overall factor, defaults to 1.
 
   Attributes
   ----------
   __indexName__ : tuple[str]
      Refers to the name for each of the indices used in the implementation of
      Block. **CAVEAT:** Do not modify!
   __indexType__ : tuple[type]
      Refers to the type of each of the indices used in the implementation of
      Block. Must be chosen accordingly to properly stringify and parse this
      Block. **CAVEAT:** Do not modify!
   __suffix__ : str
      String representation of all the indices. Expected to contain as many
      *%s* as there are indices present in the implementation of Block. This
      will be used for both calls to `__str__` as well as parsing Block from
      string via regular expressions. **CAVEAT:** Do not modify!
   """
   __suffix__    = ""
   __indexName__ = tuple()
   __indexType__ = tuple()
   __massDim__   = 0
   def __init__(self, factor:int|Fraction|Complex = 1):
      self.factor = factor
  
   def __neg__(self):
     """
     Changes the sign of the overall factor.

     Returns
     -------
     Block
        A copy of Block with negative overall factor.
     """
     cp = _copy(self)
     cp.factor = -cp.factor
     return cp

   def __str__(self):
      """
      Returns identifiable string representation of Block including the
      appropriate index structure for non-trivial indices.
      
      Returns
      -------
      str
         String representation of implementation of Block.
      
      Raises
      ------
      AssertionError
         If overall prefactor is non-trivial, i.e., call to `Block.simplify`
         in parent has been forgotten.
      """
      assert self.factor==1
      return self.__class__.__name__ +self.__suffix__%tuple(
         str(getattr(self,x)) for x in self.__indexName__)

   def charge(self):
      """
      Implements conventional charge conjugation.
      
      Returns
      -------
      Block
         A copy of itself transformed accordingly.
      """
      return _copy(self)

   def chiralSpurion(self):
      """
      Implements conventional spurionic chiral transformation.
      
      Returns
      -------
      int
         Anything other than masses or Dirac gamma matrices transforms
         trivially under chiral spurion.
      """
      return 1

   def reflection(self, mu:int):
      """
      Implements conventional Euclidean reflection in direction *mu*.
      
      Parameters
      ----------
      mu : int
         Spacetime direction.
      
      Returns
      -------
      Block
         A copy of itself transformed accordingly.
      """
      return _copy(self)

   def rotation(self, plane:int):
      """
      Implements conventional 90 degree rotations in the chosen *plane* as
      defined in `rotPlanes`.
      
      Parameters
      ----------
      plane : int
         Identifier of the current rotational plane [0,len(`rotPlanes`)[.
      
      Returns
      -------
      Block
         A copy of itself transformed accordingly.
      """
      return _copy(self)

   def simplify(self):
      """
      Is expected to take care of any relations among different index
      permutations. => Has to be adapted for non-trivial cases!
      """
      pass

   @classmethod
   def variants(cls):
      """
      Yields a generator over all declared (generalised) indices thus amounting
      to all the allowed index combinations. 

      => May be adapted for non-trivial cases!

      Returns
      -------
      Iterator[Block]
         Iterator over all possible variants of the particular implementation of
         Block.
      """
      for indexPerm in product(*cls._types()):
         yield cls(*indexPerm)

   @classmethod
   def _types(cls):
      """
      Determines the types for the indices relevant for the (custom) Block
      implementation from the function-signature of Block.__init__.
      **Type hints involving the exact types of the indices are mandatory!**

      Returns
      -------
      list[type]
         Type for each of the indices.
      """
      return cls.__indexType__

   @classmethod
   def _toRegex(cls):
      """
      Turns the specific Block implementation *cls* into a regex string to be
      used when parsing string representations of a LinearComb. The class name
      as well as any index or generalisation thereof are (expected to be) put
      into a regex group.

      Returns
      -------
      str
         Regular expression matching the particular implementation of Block
         starting with the class name followed by an implementation-specific
         suffix. Typically some indices.
      """
      return "(" + cls.__name__ + ")" + re.escape(cls.__suffix__)%tuple("(" +\
         ("|".join(re.escape(x.name) for x in t) if issubclass(t, CustomIndex)\
         else "|".join(str(x) for x in t)) + ")" for t in cls.__indexType__)


class Multiplicative(ABC,Block):
   """
   Abstract class to hint at implementation of __mul__ during call to simplify.

   Parameters
   ----------
   factor : int|Fraction|Complex, optional
      Overall factor, defaults to 1.
   """
   @abstractmethod
   def __mul__(self, factor):
      """
      Multiplication with instance of same type , i.e.,

      ::

         self.__class__ is factor.__class__ == True

      must be implemented for any subclass of Multiplicative.

      Parameters
      ----------
      factor : Multiplicative
         2nd factor of same type.
      
      Returns
      -------
      Multiplicative
         Product of this instance and *factor*.
      """
      pass


class Colour(Block):
   """
   Right now used as dummy to avoid explicit index contraction of Nc^2-1
   elements. -> Could be implemented for Nc=2 w/o loss of generality?
   
   Parameters
   ----------
   factor : int|Fraction|Complex, optional
      Overall factor, defaults to 1.
   """
   pass


@massDim(1)
class M(Block):
   """
   Represents the quark mass matrix when inserted into a singlet or a Trace.
   Otherwise it is the average sum of quark masses of the bilinear (which is
   equivalent to the quark mass matrix for the singlet).

   Parameters
   ----------
   factor : int|Fraction|Complex, optional
      Overall factor, defaults to 1.
   """
   def chiralSpurion(self):
      """
      Masses transform non-trivially under chiral spurion transformation.
      You could phrase it as if they anticommute with g5.

      Returns
      -------
      int
         Chiral spurion transformation behaviour, here -1.
      """
      return -1


@massDim(1)
class dM(M):
   """
   Represents half the difference of quark masses of the left and right flavour
   in a Bilinear,

      q.dM.Q = (m(q)-m(Q))/2 q.Q

   i.e., dM vanishes when inserted into a singlet. In combination with M this
   allows to work out mass-differences arising from `D0l`, `D0` acting in both
   directions within a `LinearComb`.
   
   Parameters
   ----------
   factor : int|Fraction|Complex, optional
      Overall factor, defaults to 1.
   """
   def charge(self):
      """
      The mass-difference changes sign under charge conjugations as the
      flavours in a Bilinear get exchanged.

      Returns
      -------
      dM
         A copy of this instance with opposite sign of the overal factor.
      """
      return self.__class__(-self.factor)


@massDim(1)
@defaultIndices
class d(Block):
   """
   Represents a total derivative.
   Due to its transformation properties (vector-like) it is also used as a
   parent class of covariant derivatives and the gluonic EOM.
   
   Parameters
   ----------
   mu : sptIdx
      Spacetime index [0,dim[.
   factor : int|Fraction|Complex, optional
      Overall factor, defaults to 1.
   """
   def __init__(self, mu:sptIdx, factor:int|Fraction|Complex = 1):
      self.mu = mu
      self.factor = factor

   def reflection(self, mu:int):
      if self.mu != mu:
         return self.__class__(self.mu, self.factor)
      return self.__class__(self.mu, -self.factor)

   def rotation(self, plane:int):
      rho,sigma = rotPlanes[plane]
      # understand signs!!!
      if self.mu == rho:
         return self.__class__(sigma, -self.factor)
      elif self.mu == sigma:
         return self.__class__(rho, self.factor)
      return self.__class__(self.mu, self.factor)


@massDim(1)
@defaultIndices
class Dl(d):
   def charge(self):
      return D(self.mu, self.factor)


@massDim(1)
@defaultIndices
class D(d):
   def charge(self):
      return Dl(self.mu, self.factor)


@massDim(1)
class D0(Block):
   """
   Used to mask fermion EOM acting to the right when creating all variants.
   
   Parameters
   ----------
   factor : int|Fraction|Complex, optional
      Overall factor, defaults to 1.
   """
   def charge(self):
      return D0l(-self.factor)
      
   def chiralSpurion(self):
      return -1


@massDim(1)
class D0l(Block):
   """
   Used to mask fermion EOM acting to the left when creating all variants.
   
   Parameters
   ----------
   factor : int|Fraction|Complex, optional
      Overall factor, defaults to 1.
   """
   def charge(self):
      return D0(-self.factor)
      
   def chiralSpurion(self):
      return -1


@massDim(3)
@defaultIndices
class DF(d):
   """
   Used to mask gluon EOM acting to the right when creating all variants.
   
   Parameters
   ----------
   mu : sptIdx
      Spacetime index [0,dim[.
   factor : int|Fraction|Complex, optional
      Overall factor, defaults to 1.
   """
   def charge(self):
      return _DFl(self.mu, -self.factor)

@massDim(3)
@defaultIndices
class _DFl(d):
   """
   Used to mask gluon EOM acting to the right when creating all variants.

   **CAVEAT:** For internal use only!
   
   Parameters
   ----------
   mu : sptIdx
      Spacetime index [0,dim[.
   factor : int|Fraction|Complex, optional
      Overall factor, defaults to 1.
   """
   def charge(self):
      return DF(self.mu, -self.factor)


@massDim(2)
@defaultIndices
class F(Block):
   """
   Represents the field strength tensor

      F(mu,nu) = [D(mu),D(nu)]

   with all the expected transformation properties. Current convention is to
   keep mu<nu and otherwise introduce an overall minus sign.

   Parameters
   ----------
   mu : sptIdx
      Spacetime index [0,dim[.
   nu : sptIdx
      Spacetime index [0,dim[.
   factor : int|Fraction|Complex, optional
      Overall factor, defaults to 1.
   """
   def __init__(self, mu:sptIdx, nu:sptIdx, factor:int|Fraction|Complex = 1):
      self.mu = mu
      self.nu = nu
      self.factor = factor

   def simplify(self):
      if self.mu>self.nu:
         self.factor = -self.factor
         self.mu,self.nu = self.nu,self.mu
      elif self.mu==self.nu:
         self.factor = 0

   def charge(self):
      return self.__class__(self.mu, self.nu, -self.factor)

   def reflection(self, mu:int):
      if self.mu==mu or self.nu==mu:
         return self.__class__(self.mu, self.nu, -self.factor)
      return self.__class__(self.mu, self.nu, self.factor)

   def rotation(self, plane:int):
      rho,sigma = rotPlanes[plane]
      # understand signs!!!
      if self.mu == rho and self.nu == sigma:
         return self.__class__(sigma, rho, -self.factor)
      if self.mu == sigma and self.nu == rho:
         return self.__class__(rho, sigma, -self.factor)
      if self.mu == rho:
         return self.__class__(sigma, self.nu, -self.factor)
      if self.mu == sigma:
         return self.__class__(rho, self.nu, self.factor)
      if self.nu == rho:
         return self.__class__(self.mu, sigma, -self.factor)
      if self.nu == sigma:
         return self.__class__(self.mu, rho, self.factor)
      return self.__class__(self.mu, self.nu, self.factor)

   @classmethod
   def variants(cls):
      """
      Returns all variants of F[mu,nu] with mu<nu.

      Returns
      -------
      Iterator[F]
         Iterator of all variants of F[mu,nu] with mu<nu.
      """
      for nu in sptIdx:
         for mu in sptIdx:
            if mu>=nu: continue
            yield cls(mu,nu)


class _AlgebraBlock(Block):
   """
   Container for covariant derivatives acting on an algebra-valued object like
   F[mu,nu], Colour, DF[mu] etc.
  
   **CAVEAT:** For internal use only!

   Parameters
   ----------
   blocks : list[Block]
      Ordered collection of covariant derivatives acting on the last element
      that is algebra valued.
   factor : int|Fraction|Complex, optional
      Overall factor, defaults to 1.
   """
   def __init__(self, blocks:list[Block], factor:int|Fraction|Complex = 1):
      self.blocks = blocks
      self.factor = factor

   def simplify(self):
      if self.blocks[0].__class__ is F:
         self.blocks = [D(b.mu,b.factor) for b in self.blocks[:0:-1]] +\
                       [self.blocks[0]]
      elif self.blocks[0].__class__ is _DFl:
         self.blocks = [D(b.mu,b.factor) for b in self.blocks[:0:-1]] +\
                       [DF(self.blocks[0].mu, self.blocks[0].factor)]
      for b in self.blocks:
         b.simplify()
         self.factor *= b.factor
         b.factor = 1

   def __str__(self):
      """
      Overwritten default behaviour of str(self) to handle with collections of
      `Block`.

      Returns
      -------
      str
         String representation of all instances of `Block` contained within the
         _AlgebraBlock.
      """
      assert self.factor==1
      return ".".join([str(b) for b in self.blocks])

   def charge(self):
      return _AlgebraBlock([b.charge() for b in self.blocks[::-1]], self.factor)

   def reflection(self, mu:int):
      return _AlgebraBlock([b.reflection(mu) for b in self.blocks], self.factor)

   def rotation(self, plane:int):
      return _AlgebraBlock([b.rotation(plane) for b in self.blocks],
                           self.factor)
