"""
This submodule implements the basic properties of 4x4 Dirac gamma matrices
including their transformations under the usual discrete spacetime symmetries.

Attributes
----------
axisGamma : tuple[Dirac]
   Specifies the indices from `Dirac` to be used for the different directions
   mu=0,1,2,3.
"""

from fractions import Fraction
from copy import deepcopy as _copy

from .blocks import Multiplicative
from .basics import rotPlanes, Complex, CustomIndex, indices, defaultIndices

Dirac = CustomIndex("Dirac", ["id_", "g5", "gw", "gx", "gy", "gz", "gw5",\
   "gx5", "gy5", "gz5", "gwx", "gwy", "gwz", "gxy", "gxz", "gyz"], start=1)

axisGammas = (Dirac.gw, Dirac.gx, Dirac.gy, Dirac.gz)

# Implement products of gamma matrices
MULTABLE = ((Dirac.id_.value, Dirac.g5.value, Dirac.gw.value, Dirac.gx.value, Dirac.gy.value, Dirac.gz.value, Dirac.gw5.value, Dirac.gx5.value, Dirac.gy5.value, Dirac.gz5.value, Dirac.gwx.value, Dirac.gwy.value, Dirac.gwz.value, Dirac.gxy.value, Dirac.gxz.value, Dirac.gyz.value), 
   (Dirac.g5.value, Dirac.id_.value, -Dirac.gw5.value, -Dirac.gx5.value, -Dirac.gy5.value, -Dirac.gz5.value, -Dirac.gw.value, -Dirac.gx.value, -Dirac.gy.value, -Dirac.gz.value, -Dirac.gyz.value, Dirac.gxz.value, -Dirac.gxy.value, -Dirac.gwz.value, Dirac.gwy.value, -Dirac.gwx.value), 
   (Dirac.gw.value, Dirac.gw5.value, Dirac.id_.value, Dirac.gwx.value, Dirac.gwy.value, Dirac.gwz.value, Dirac.g5.value, -Dirac.gyz.value, Dirac.gxz.value, -Dirac.gxy.value, Dirac.gx.value, Dirac.gy.value, Dirac.gz.value, -Dirac.gz5.value, Dirac.gy5.value, -Dirac.gx5.value), 
   (Dirac.gx.value, Dirac.gx5.value, -Dirac.gwx.value, Dirac.id_.value, Dirac.gxy.value, Dirac.gxz.value, Dirac.gyz.value, Dirac.g5.value, -Dirac.gwz.value, Dirac.gwy.value, -Dirac.gw.value, Dirac.gz5.value, -Dirac.gy5.value, Dirac.gy.value, Dirac.gz.value, Dirac.gw5.value), 
   (Dirac.gy.value, Dirac.gy5.value, -Dirac.gwy.value, -Dirac.gxy.value, Dirac.id_.value, Dirac.gyz.value, -Dirac.gxz.value, Dirac.gwz.value, Dirac.g5.value, -Dirac.gwx.value, -Dirac.gz5.value, -Dirac.gw.value, Dirac.gx5.value, -Dirac.gx.value, -Dirac.gw5.value, Dirac.gz.value), 
   (Dirac.gz.value, Dirac.gz5.value, -Dirac.gwz.value, -Dirac.gxz.value, -Dirac.gyz.value, Dirac.id_.value, Dirac.gxy.value, -Dirac.gwy.value, Dirac.gwx.value, Dirac.g5.value, Dirac.gy5.value, -Dirac.gx5.value, -Dirac.gw.value, Dirac.gw5.value, -Dirac.gx.value, -Dirac.gy.value), 
   (Dirac.gw5.value, Dirac.gw.value, -Dirac.g5.value, Dirac.gyz.value, -Dirac.gxz.value, Dirac.gxy.value, -Dirac.id_.value, -Dirac.gwx.value, -Dirac.gwy.value, -Dirac.gwz.value, Dirac.gx5.value, Dirac.gy5.value, Dirac.gz5.value, -Dirac.gz.value, Dirac.gy.value, -Dirac.gx.value), 
   (Dirac.gx5.value, Dirac.gx.value, -Dirac.gyz.value, -Dirac.g5.value, Dirac.gwz.value, -Dirac.gwy.value, Dirac.gwx.value, -Dirac.id_.value, -Dirac.gxy.value, -Dirac.gxz.value, -Dirac.gw5.value, Dirac.gz.value, -Dirac.gy.value, Dirac.gy5.value, Dirac.gz5.value, Dirac.gw.value), 
   (Dirac.gy5.value, Dirac.gy.value, Dirac.gxz.value, -Dirac.gwz.value, -Dirac.g5.value, Dirac.gwx.value, Dirac.gwy.value, Dirac.gxy.value, -Dirac.id_.value, -Dirac.gyz.value, -Dirac.gz.value, -Dirac.gw5.value, Dirac.gx.value, -Dirac.gx5.value, -Dirac.gw.value, Dirac.gz5.value), 
   (Dirac.gz5.value, Dirac.gz.value, -Dirac.gxy.value, Dirac.gwy.value, -Dirac.gwx.value, -Dirac.g5.value, Dirac.gwz.value, Dirac.gxz.value, Dirac.gyz.value, -Dirac.id_.value, Dirac.gy.value, -Dirac.gx.value, -Dirac.gw5.value, Dirac.gw.value, -Dirac.gx5.value, -Dirac.gy5.value), 
   (Dirac.gwx.value, -Dirac.gyz.value, -Dirac.gx.value, Dirac.gw.value, -Dirac.gz5.value, Dirac.gy5.value, -Dirac.gx5.value, Dirac.gw5.value, -Dirac.gz.value, Dirac.gy.value, -Dirac.id_.value, -Dirac.gxy.value, -Dirac.gxz.value, Dirac.gwy.value, Dirac.gwz.value, Dirac.g5.value), 
   (Dirac.gwy.value, Dirac.gxz.value, -Dirac.gy.value, Dirac.gz5.value, Dirac.gw.value, -Dirac.gx5.value, -Dirac.gy5.value, Dirac.gz.value, Dirac.gw5.value, -Dirac.gx.value, Dirac.gxy.value, -Dirac.id_.value, -Dirac.gyz.value, -Dirac.gwx.value, -Dirac.g5.value, Dirac.gwz.value), 
   (Dirac.gwz.value, -Dirac.gxy.value, -Dirac.gz.value, -Dirac.gy5.value, Dirac.gx5.value, Dirac.gw.value, -Dirac.gz5.value, -Dirac.gy.value, Dirac.gx.value, Dirac.gw5.value, Dirac.gxz.value, Dirac.gyz.value, -Dirac.id_.value, Dirac.g5.value, -Dirac.gwx.value, -Dirac.gwy.value), 
   (Dirac.gxy.value, -Dirac.gwz.value, -Dirac.gz5.value, -Dirac.gy.value, Dirac.gx.value, Dirac.gw5.value, -Dirac.gz.value, -Dirac.gy5.value, Dirac.gx5.value, Dirac.gw.value, -Dirac.gwy.value, Dirac.gwx.value, Dirac.g5.value, -Dirac.id_.value, -Dirac.gyz.value, Dirac.gxz.value), 
   (Dirac.gxz.value, Dirac.gwy.value, Dirac.gy5.value, -Dirac.gz.value, -Dirac.gw5.value, Dirac.gx.value, Dirac.gy.value, -Dirac.gz5.value, -Dirac.gw.value, Dirac.gx5.value, -Dirac.gwz.value, -Dirac.g5.value, Dirac.gwx.value, Dirac.gyz.value, -Dirac.id_.value, -Dirac.gxy.value), 
   (Dirac.gyz.value, -Dirac.gwx.value, -Dirac.gx5.value, Dirac.gw5.value, -Dirac.gz.value, Dirac.gy.value, -Dirac.gx.value, Dirac.gw.value, -Dirac.gz5.value, Dirac.gy5.value, Dirac.g5.value, -Dirac.gwz.value, Dirac.gwy.value, -Dirac.gxz.value, Dirac.gxy.value, -Dirac.id_.value))

CTABLE = (1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,-1,-1,-1,-1)


@defaultIndices
class Gamma(Multiplicative):
   """
   Implementation of 4d Dirac gamma matrices.

   Parameters
   ----------
   gamma : Dirac
      Identifier of the chosen gamma matrix.
   factor : int|Fraction|Complex, optional
      Overall factor, defaults to 1.
   """
   def __init__(self, gamma:Dirac, factor:int|Fraction|Complex = 1):
      self.factor = factor
      self.gamma = gamma

   def __eq__(self, cmp):
      return self.__class__ == cmp.__class__ and self.gamma == cmp.gamma and\
             self.factor == cmp.factor

   def __mul__(self, rhs):
      if rhs.__class__ is self.__class__:
         res = MULTABLE[self.gamma.value-1][rhs.gamma.value-1]
         return self.__class__(Dirac(abs(res)),
                               (-1 if res<0 else 1)*self.factor*rhs.factor)
      # ensures that in all unknown cases rhs.__rmul__ will be called.
      return NotImplemented

   def __rmul__(self, lhs):
      if isinstance(lhs, (Complex, Fraction, int, long)):
         return self.__class__(self.gamma, self.factor*lhs)
      raise NotImplementedError(
         "Right-handed multiplication not implemented for this type.")

   def charge(self):
      return self.__class__(self.gamma,
                            CTABLE[self.gamma.value-1]*self.factor)

   def chiralSpurion(self):
      _g5 = self.__class__(Dirac.g5)
      if _g5 * self * _g5 == self:
         return 1
      return -1

   def reflection(self, mu:int):
      reflGamma = self.__class__(axisGammas[mu]) * self.__class__(Dirac.g5)
      return -reflGamma * self * reflGamma

   def rotation(self, plane:int):
      rho,sigma = rotPlanes[plane]
      rotGamma = self.__class__(axisGammas[rho]) *\
                 self.__class__(axisGammas[sigma])
      test = rotGamma * self
      if test == self * rotGamma:
         return _copy(self)
      return self.__class__(test.gamma, test.factor)

   @classmethod
   def variants(cls):
      for gamma in Dirac:
         yield cls(gamma)

