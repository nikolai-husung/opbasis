import opBasis as opb


# Implement custom taste space transformation for modified C, P, T, rotations
# and remnant chiral. As it turns out, MULTABLE stays exactly the same if you
# replace \gamma_\mu -> \tau_\mu in all the expressions.
@opb.defaultIndices
class Taste(opb.Gamma):
   def reflection(self, mu:int):
      _t5 = self.__class__(opb.Dirac.g5)
      return _t5 * self * _t5

   def rotation(self, plane:int):
      rho,sigma = opb.rotPlanes[plane]
      trho = self.__class__(opb.axisGammas[rho])
      tsigma = self.__class__(opb.axisGammas[sigma])
      _t5 = self.__class__(opb.Dirac.g5)
      test = trho*self*trho
      test2 = tsigma*self*tsigma
      if test == test2:
         return _t5*test*_t5
      return -_t5*trho*self*tsigma*_t5

   def tau5Spurion(self):
      _t5 = self.__class__(opb.Dirac.g5)
      if _t5 * self * _t5 == self:
         return 1
      return -1

# Specify discrete transformations as listed in model file.
# -> Staggered has some unconventional ones.
def modH4(op:opb.LinearComb, plane:int): return op.rotation(plane)

def modC(op:opb.LinearComb, _blank:int): return op.charge()

def modP(op:opb.LinearComb, mu:int): return op.reflection(mu)

# already taken care of by inheritance of Taste class
def remnantChiral(op:opb.LinearComb): return op.chiralSpurion()

def tau5(op:opb.LinearComb):
   for term in op.terms:
      test = 1
      for bilin in term.bilinears:
         for b in bilin.blocks:
            if not b.__class__ is Taste: continue
            test *= b.tau5Spurion()
      if test==-1: return False
   return True

def Shift(op:opb.LinearComb, mu:int):
   _tmu = Taste(opb.axisGammas[mu])
   for bilin in op.bilinears:
      bilin.blocks = [_tmu] + bilin.blocks + [_tmu]
   return op
