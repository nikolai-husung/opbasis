"""
This submodule implements the most basic objects used throughout the package,
ranging from `Union` and `Complex` numbers to customisable indices via
`CustomIndex` and `IntIndex`. The latter then have to be tracked for custom
`Block` implementations by use of the
`decorators <https://docs.python.org/3/glossary.html#term-decorator>`_
`@indices` or `@defaultIndices`.

Attributes
----------
dim : int
   Specifies the dimensions assumed in spacetime. Any change may require some
   adjustment of `rotPlanes` as well as any `Block` transformation properties.
rotPlanes : tuple[tuple[sptIdx,sptIdx]]
   Spacetime indices associated to the different rotational planes.
"""
from fractions import Fraction
from enum import Enum, IntEnum, unique
from copy import deepcopy as _copy
import math


whiteSpace = "\t"

def indices(suffix:str|None,**kwargs:dict[str,type]):
   """
   Sets the class attributes

   ::

      cls.__indexType__ # tuple[type] referring to the type of the index
      cls.__indexName__ # tuple[str]  referring to the index label
      cls.__suffix__    # str         representation of indices etc.

   Those are necessary to turn custom `Block` implementations into regular
   expressions. *suffix* is expected to carry as many *%s* as there are
   indices.

   Parameters
   ----------
   suffix : str|
      String representation of the trailing indices etc. used during calls
      to `Block.__str__` as well as parsing from string representation via
      regular expressions. If `None`, the default form is chosen for
      __suffix__.
   kwargs : dict[str,type]
      Collection of member-names of the current Block implementation referring
      to indices and their associated type. From Python 3.10 onwards
      dictionaries have a **frozen** order. Make sure to use an ordering of the
      indices that matches the arguments in the function signature.

   Returns
   -------
   function
      Handler acting on the actual class type.

   Raises
   ------
   AssertionError
      If the number of *%s* in *suffix* does not match the number of indices.
   """
   def inner(cls):
      cls.__indexType__ = tuple(kwargs.values())
      cls.__indexName__ = tuple(kwargs.keys())
      l = len(kwargs)
      if suffix is None:
         cls.__suffix__ = "" if l==0 else "[" + ",".join(["%s"]*l) + "]"
      else:
         assert l == len(suffix.split("%s"))-1,\
            "There must be as many %s in suffix as there are indices."
         cls.__suffix__ = suffix
      return cls
   return inner


def defaultIndices(cls):
   """
   Assumes the default behaviour for indices, i.e., the custom Block.__init__
   is expected to have as arguments all the indices plus one called *factor*
   referring to the overall factor. The arguments belonging to indices must be
   named according to the class-member used to store the indices and must be
   type-hinted using the explicit type for each index.

   Parameters
   ----------
   cls : type
      Block implementations to be assigned default index conventions.

   Returns
   -------
   type
      The initial Block implementation but with indices assigned.
   """
   kwargs = dict(cls.__init__.__annotations__)
   del kwargs["factor"]
   return indices(None, **kwargs)(cls)


def massDim(md:int|Fraction):
   """
   Allows to assign a non-zero mass dimension to an instance of `Block` through
   use of a `decorator <https://docs.python.org/3/glossary.html#term-decorator>`_
   @massDim(...) in front of the class implementation. Mass-dimensions are
   expected to be non-negative integers. If you want to deviate from this, one
   may rescale all other mass-dimensions to restore overall integer-valuedness.

   Parameters
   ----------
   md : int
      Mass-dimension that should be assigned to the current implementation of
      `Block`.

   Returns
   -------
   cls : type
      The initial Block implementation but with a mass-dimension set.

   Raises
   ------
   AssertionError
      If the chosen mass-dimension is not a non-negative integer.
   """
   assert isinstance(md, (int,Fraction)) and md >=0,\
      "Mass-dimension is expected to be a non-negative rational number."
   def setMassDim(cls):
      cls.__massDim__ = md
      return cls
   return setMassDim


@unique
class CustomIndex(Enum):
   """Custom-labelled indices.

   Allows the definition of indices with custom string-valued labels. Uses
   Python Enum as underlying object to assign integer-values to those labels.

   Parameters
   ----------
   value : str
      The name of the new CustomIndex (Enum) to create.
   names : list[str]
      The string-values to be used for labelling the custom indices.
   """
   def __str__(self):
      """
      Overrides the default `Enum.__str__` behaviour to stick to convention that
      a call to *str(...)* returns the string representation of the object.

      Returns
      -------
      str
         *name* of the current Enum element.
      """
      return self.name


@unique
class IntIndex(IntEnum):
   """
   Represents an integer-valued index, directly inherited from `IntEnum`
   without modification. Allows to define a range of values allowed for the
   indices.
   """
   pass


class Union:
   """
   Collection of elements *x* with an associated label *str(x)* assumed to be
   a unique identifier. Based on this label, adding a new element is vetoed if
   the label is already present.
   """
   def __init__(self):
      self.labels = list()
      self.content = list()

   def add(self, x):
      """
      Adds element *x* to the collection if not already present according to
      its string-representation.

      Parameters
      ----------
      x : object
         Element to be added.

      Returns
      -------
      bool
         Returns whether the element *x* has been added or vetoed due to already
         being present.
      """
      l = str(x)
      if not l in self.labels:
         self.labels.append(l)
         self.content.append(_copy(x))
         return True
      return False

   def __iter__(self):
      return iter(self.content)


class Complex:
   """
   Rudimentary implementation of a rational-number-valued complex number.

   Parameters
   ----------   
   re : int|Fraction
      Real part of the complex number.
   im : int|Fraction, optional
      Imaginary part of the complex number, defaults to 0.
   """
   def __init__(self, re:int|Fraction, im:int|Fraction=0):
      self.re = re
      self.im = im

   def __bool__(self):
      return self.re!=0 or self.im!=0

   def __eq__(self, cmp):
      if isinstance(cmp, Complex):
         return self.re==cmp.re and self.im==cmp.im
      if isinstance(cmp, (Fraction, int)):
         return self.re==cmp and self.im==0
      return NotImplemented
      
   def __repr__(self):
      return self.__class__.__name__ + "(%s,%s)"%(str(self.re),str(self.im))

   def __str__(self):
      if self.im==0:
         return ("" if self.re<0 else "+") + str(self.re)
      if self.re==0:
         return ("" if self.im<0 else "+") + str(self.im)
      if self.re<0:
         return "-(" + str(-self.re) + ("" if self.im>0 else "+") +\
            str(-self.im) + "j)"
      return "+(" + str(self.re) + ("" if self.im<0 else "+") +\
            str(self.im) + "j)"

   def __abs__(self):
      return math.isqrt(self.re*self.re + self.im*self.im)

   def __add__(self, add):
      if isinstance(add, Complex):
         return self.__class__(self.re+add.re, self.im+add.im)
      return self.__class__(self.re+add, self.im)

   def __radd__(self, add):
      return self.__class__(self.re+add, self.im)

   def __sub__(self, sub):
      if isinstance(sub, Complex):
         return self.__class__(self.re-sub.re, self.im-sub.im)
      return self.__class__(self.re-sub, self.im)

   def __rsub__(self, sub):
      return self.__class__(sub-self.re, -self.im)

   def __neg__(self):
      return self.__class__(-self.re, -self.im)

   def __mul__(self, mul):
      if isinstance(mul, Complex):
         return self.__class__(self.re*mul.re-self.im*mul.im,
                               self.re*mul.im+self.im*mul.re)
      if isinstance(mul, (Fraction, int)):
         return self.__class__(self.re*mul, self.im*mul)
      return NotImplemented
   
   def __rmul__(self, mul):
      return self.__class__(self.re*mul, self.im*mul)

   def __truediv__(self, div):
      if isinstance(div, (Fraction, int)):
         return self.__class__(Fraction(self.re, div), Fraction(self.im, div))
      if isinstance(div, Complex):
         temp = div.abs2()
         return self.__class__(
            Fraction(self.re*div.re + self.im*div.im, temp),
            Fraction(self.im*div.re - self.re*div.im, temp))
      return NotImplemented

   def __rtruediv__(self, lhs):
      return self.__class__(Fraction(lhs, self.re), Fraction(lhs, self.im))

   def __floordiv__(self, div):
      return self.__class__(Fraction(self.re, div), Fraction(self.im, div))

   def phase(self):
      """
      Assuming the complex number to be purely real or purely imaginary, this
      function returns the sign and (for a purely imaginary number) an
      imaginary unit.
      
      Returns
      -------
      Complex
         Sign and any overall imaginary unit.

      Raises
      ------
      AssertionError
         If number is not purely real or purely imaginary.
      """
      assert self.re==0 or self.im==0
      return self.__class__(
         0 if self.im!=0 else (-1 if self.re<0 else 1),
         -1 if self.im<0 else (1 if self.re==0 else 0))

   def abs2(self) -> int|Fraction:
      """
      Computes z*conj(z) of the complex number z.
      
      Returns
      -------
      int|Fraction
         Absolute value squared of the current complex number.
      """
      return self.re*self.re + self.im*self.im

   def conj(self):
      """
      Complex conjugation.
      
      Returns
      -------
      Complex
         Complex conjugate of the current complex number.
      """
      return Complex(self.re, -self.im)
 
dim = 4

sptIdx = IntIndex("sptIdx", ["w","x","y","z"], start=0)#: :meta hide-value:

rotPlanes = tuple((sptIdx(x[0]),sptIdx(x[1])) \
   for x in ((0,1), (0,2), (0,3), (1,2), (1,3), (2,3)))


