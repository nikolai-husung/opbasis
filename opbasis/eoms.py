"""
This submodule implements unmasking of EOMs, i.e., reexpressing `D0l`, `D0`, and
`DF` into user-provided expression. This is typically being done **after**
deriving an overcomplete basis. This way we have a handle on keeping track of
EOM-vanishing operators and impose those relations when deriving a minimal
basis.
"""

from dataclasses import dataclass
from copy import deepcopy as _copy
from fractions import Fraction

from .opBasis import Model
from .blocks import DF, D0, D0l, _AlgebraBlock, Colour, M, dM
from .ops import LinearComb, Trace, Bilinear
from .basics import dim

@dataclass
class GluonEOM:
   """
   The gluon EOM specifies what the short-hand `DF` actually looks like.
   """
   TF:Fraction
   Nc:Fraction
   gluonPart:list[LinearComb]
   fermionPart:list[LinearComb|None]


def _unmaskgeom(unmasked:LinearComb, gluonEOM:GluonEOM)->bool:
   """
   Internal routine that replaces a single gluon EOM by the appropriate linear
   combination on each call. Returns a boolean indicating whether an EOM was
   encountered.

   Parameters
   ----------
   unmasked : LinearComb
      Expression in which the gluon EOM is supposed to be written out
      explicitly.
   gluonEOM : GluonEOM
      Descriptor indicating how to unmask the gluon EOM.

   Returns
   -------
   bool
      Boolean indicating whether a masked gluon EOM has been encountered and
      unmasked.
   """
   for it,term in enumerate(unmasked.terms):
      for ip,p in enumerate(term.prod):
         for ib,b in enumerate(p.cyclic if p.__class__ is Trace else p.blocks):
            if b.__class__ is _AlgebraBlock and b.blocks[-1].__class__ is DF:
               mu = b.blocks[-1].mu
               for t in gluonEOM.gluonPart[mu].terms:
                  temp = _copy(term)
                  # check signs and normalisation!
                  temp.factor *= gluonEOM.gluonPart[mu].factor*t.factor
                  idx = [not c.blocks[-1].__class__ is Colour \
                         for c in t.prod[0].cyclic].index(True)
                  if p.__class__ is Trace:
                     temp.prod[ip].cyclic[ib].blocks = _copy(
                        list(b.blocks[:-1]) + t.prod[0].cyclic[idx].blocks)
                  else:
                     temp.factor *= -gluonEOM.TF
                     temp.prod[ip].blocks[ib].blocks = _copy(
                        list(b.blocks[:-1]) + t.prod[0].cyclic[idx].blocks)
                  unmasked.terms.append(temp)
               if not gluonEOM.fermionPart[mu] is None:
                  for t in gluonEOM.fermionPart[mu].terms:
                     temp = _copy(term)
                     temp.factor *= gluonEOM.fermionPart[mu].factor*t.factor
                     if p.__class__ is Trace:
                        temp.prod[ip] = _copy(t.prod[0])
                        idx = [c.__class__ is _AlgebraBlock and \
                               c.blocks[-1].__class__ is Colour \
                               for c in t.prod[0].blocks].index(True)
                        temp.prod[ip].blocks = _copy(t.prod[0].blocks[:idx] +\
                           p.cyclic[ib+1:] + p.cyclic[:ib] +\
                           t.prod[0].blocks[idx+1:])
                        unmasked.terms.append(temp)
                        # Dealing with Trace involving more than two
                        # _AlgebraBlock leads to sum of Traces with relative
                        # factor -1/Nc.
                        if len(p.cyclic)>2:
                           temp = _copy(term)
                           temp.factor /= -gluonEOM.Nc
                           del temp.prod[ip].blocks[ib]
                           temp.prod.append(_copy(t.prod[0]))
                           del temp.prod[-1].blocks[idx]
                           unmasked.terms.append(temp)
                     else:
                        # We purposefully do not write it out: Crossed
                        # contractions in spin, and colour space are not
                        # supported!
                        temp.prod[ip].blocks[ib] = _AlgebraBlock([Colour()])
                        temp.prod.append(_copy(t.prod[0]))
                        unmasked.terms.append(temp)
               del unmasked.terms[it]
               return True
   return False


def unmaskGluonEOMs(op:LinearComb, gluonEOM:GluonEOM):
   """
   Writes out all gluon EOMs contained in *op* explicitly.
   *gluonEOM* has to be given in the form

      1/TF*tr(DF[mu].Colour) :=
         1/TF*tr([D(nu),F(nu,mu)].Colour) + SUM_Psi Psi.Gamma[gmu].Colour.Psi

   Any deviation from this default form can be implemented by changing the
   linear combination.

      tr(Colour.Colour) = -TF*KroneckerDelta

   It is assumed that there will be at most one pair of Colour in any
   expression, i.e., at most 4-quark operators!

   **CAUTION:** *op* and *gluonEOM* will be changed!

   Parameters
   ----------
   op : LinearComb
      Expression in which all masked occurrences of the gluon EOM are supposed
      to be unmasked.
   gluonEOM : GluonEOM
      Descriptor indicating how to unmask the gluon EOM.
   """
   # Check for appropriate form of gluon EOM.
   assert all(all(len(term.prod)==1 and term.prod[0].__class__ is Trace and \
      len(term.prod[0].cyclic)==2 for term in gluonEOM.gluonPart[mu].terms) \
         for mu in range(dim))
   op.simplify()
   for mu in range(dim):
      gluonEOM.gluonPart[mu].simplify()
      if not gluonEOM.fermionPart[mu] is None:
         gluonEOM.fermionPart[mu].simplify()
   while _unmaskgeom(op, gluonEOM): continue
   op.simplify()


def _unmaskfeom(unmasked:LinearComb, fEOM:LinearComb, fEOMl:LinearComb,
                model:Model)->bool:
   """
   Internal routine that replaces a single (matching) fermion EOM by the
   appropriate linear combination on each call. Returns a boolean indicating
   whether an EOM was encountered.

   Parameters
   ----------
   unmasked : LinearComb
      Expression in which too look for masked fermion EOMs.
   fEOM : LinearComb
      Representation of the fermion EOM action on the right flavour for a specific
      flavour or set of flavours.
   fEOMl : LinearComb
      Representation of the fermion EOM acting on the left flavour for a specific
      flavour or set of flavours.
   model : Model
       Descriptor of the currently assumed theory.

   Returns
   -------
   bool
      Boolean indicating whether a masked fermion EOM has been encountered
      and unmasked.
   """
   fset = fEOM.terms[0].prod[0].flavours[1]
   if fset in model.flavourSets.keys():
      fset = [fset] + model.flavourSets[fset]
   for it,term in enumerate(unmasked.terms):
      for ip,p in enumerate(term.prod):
         if p.__class__ is Trace: continue
         lhsFlavour = p.flavours[0] in fset
         ofs = len(p.covl)-1
         for ib,b in enumerate(p.covl[::-1]):
            if b.__class__ is D0l and lhsFlavour:
               for t in fEOMl.terms:
                  temp = _copy(term)
                  temp.factor *= fEOMl.factor*t.factor
                  temp.prod[ip].blocks = _copy(t.prod[0].blocks) +\
                     temp.prod[ip].blocks
                  temp.prod[ip].covl = temp.prod[ip].covl[:ofs-ib] +\
                     _copy(t.prod[0].covl) + temp.prod[ip].covl[ofs-ib+1:]
                  unmasked.terms.append(temp)
            else:
               continue
            del unmasked.terms[it]
            return True
         rhsFlavour = p.flavours[1] in fset
         for ib,b in enumerate(p.covr):
            if b.__class__ is D0 and rhsFlavour:
               for t in fEOM.terms:
                  temp = _copy(term)
                  temp.factor *= fEOM.factor*t.factor
                  temp.prod[ip].blocks = temp.prod[ip].blocks +\
                     _copy(t.prod[0].blocks)
                  temp.prod[ip].covr = temp.prod[ip].covr[:ib] +\
                     _copy(t.prod[0].covr) + temp.prod[ip].covr[ib+1:]
                  unmasked.terms.append(temp)
            else:
               continue
            del unmasked.terms[it]
            return True
   return False


def unmaskFermionEOMs(op:LinearComb, fEOM:LinearComb, model:Model):
   """
   Writes out all fermionic EOM contained in *op* explicitly.
   Expects *fEOM* to be of the form (for a specific set of flavours or flavour)

      Psi.D0.Psi := Psi.Gamma[gmu].D[mu].Psi + Psi.M.Psi

   Any deviation from this default form can be implemented by changing the
   linear combination.

   **CAUTION:** *op* will be changed!

   Parameters
   ----------
   op : LinearComb
      Expression in which to replace all occurrences of the masked fermion EOM
      represented by `D0` and `D0l` with the precise expression from *fEOM*.
   fEOM : LinearComb
      Written out form of the fermion EOM acting on the right flavour.
   """
   assert all(len(t.prod)==1 and t.prod[0].__class__ is Bilinear\
              for t in fEOM.terms)
   op.simplify()
   # adjust fEOM to carry both M and dM
   fEOM = _copy(fEOM)
   for b in fEOM.bilinears:
      # mask left flavour to avoid simplify removing instances of dM
      b.flavours = ("#", b.flavours[1])
   fEOM.simplify()
   fEOMl = -fEOM.charge()
   fEOMl.simplify()
   while _unmaskfeom(op, fEOM, fEOMl, model): continue
   op.simplify()
