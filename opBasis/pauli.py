"""
This submodule implements elements of SU(2), i.e., the identity and the Pauli
matrices and their multiplication rules.
"""

from fractions import Fraction
from copy import deepcopy as _copy

from .blocks import Multiplicative
from .basics import Complex, CustomIndex, indices, defaultIndices

Pauli = CustomIndex("Pauli", ["id_", "t1", "t2", "t3"], start=1)

# Implement products of gamma matrices
MULTABLE = (
   (Complex(Pauli.id_.value,0), Complex(Pauli.t1.value,0), Complex(Pauli.t2.value,0), Complex(Pauli.t3.value,0)),
   (Complex(Pauli.t1.value,0), Complex(Pauli.id_.value,0), Complex(0,Pauli.t3.value), Complex(0,-Pauli.t2.value)),
   (Complex(Pauli.t2.value,0), Complex(0,-Pauli.t3.value), Complex(Pauli.id_.value,0), Complex(0,Pauli.t1.value)),
   (Complex(Pauli.t3.value,0), Complex(0,Pauli.t2.value), Complex(0,-Pauli.t1.value), Complex(Pauli.id_.value,0)))

CTABLE = (1, 1 , -1, 1)

@defaultIndices
class SU2(Multiplicative):
   """
   Implements the hermitian 2x2 matrices (1, pauli_{1,2,3}).

   Parameters
   ----------
   tau : Pauli
      Identifier of the chosen pauli matrix / identity.
   factor : int|Fraction|Complex, optional
      Overall factor, defaults to 1.
   """
   def __init__(self, tau:Pauli, factor:int|Fraction|Complex = 1):
      self.factor = factor
      self.tau = tau

   def __eq__(self, cmp):
      return self.__class__ == cmp.__class__ and self.tau == cmp.tau and\
             self.factor == cmp.factor

   def __mul__(self, rhs):
      if rhs.__class__ is self.__class__:
         res = MULTABLE[self.tau.value-1][rhs.tau.value-1]
         return self.__class__(Pauli(abs(res)),
                               (res//abs(res))*self.factor*rhs.factor)
      if isinstance(rhs, (Complex, Fraction, int, long)):
         return self.__class__(self.tau, self.factor*lhs)
      # ensures that in all unknown cases rhs.__rmul__ will be called.
      return NotImplemented
      
   def __rmul__(self, lhs):
      if isinstance(lhs, (Complex, Fraction, int, long)):
         return self.__class__(self.tau, self.factor*lhs)
      raise NotImplementedError(
         "Right-handed multiplication not implemented for this type.")

   def charge(self):
      return self.__class__(self.tau, CTABLE[self.tau.value-1]*self.factor)

   @classmethod
   def variants(cls):
      for tau in Pauli:
         yield cls(tau)
