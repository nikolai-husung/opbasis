"""
This submodule contains the top-level Python representations for a `LinearComb`
of terms that form an operator compatible with the desired transformation
behaviour.
"""

from copy import deepcopy as _copy
from fractions import Fraction

import numpy as np


from .blocks import Block, _AlgebraBlock, Multiplicative,\
   d, D0l, D0, Dl, D, DF, F, M, dM, Colour
from .dirac import Dirac, axisGammas, Gamma
from .basics import dim, Complex, whiteSpace, massDim


@massDim(dim-1)
class Bilinear:
   """
   Bilinear of two flavours with an intermediate piece consisting of covariant
   derivatives (and fermion EOMs) acting to the left and right as well as any
   other implementation of Block in the middle.

   **CAVEAT:** Intended for internal use and checks against __class__.

   Parameters
   ----------
   blocks : list[Block]
      Collection of Blocks not acting directly on the two flavours.
   covl : list[Dl|D0l]
      Covariant derivatives (and fermion EOMs) acting to the left.
   covr : list[D|D0]
      Covariant derivatives (and fermion EOMs) acting to the right.
   flavours : tuple[str,str]
      Flavour content of the Bilinear.
   derivatives : list[d]|None, optional
      List of total derivatives acting on the Bilinear, defaults to None.
   factor : int|Fraction|Complex, optional
      Overall factor, defaults to 1.
   """
   __canonical__ = None
   def __init__(self, blocks:list[Block], covl:list[Dl|D0l], covr:list[D|D0],
                flavours:tuple[str,str], derivatives:list[d]|None=None,
                factor:int|Fraction|Complex = 1):
      self.blocks      = blocks
      self.covl        = covl
      self.covr        = covr
      self.factor      = factor
      self.flavours    = flavours
      self.derivatives = list() if derivatives is None else derivatives
      self._order()

   def _order(self):
      self.blocks = list(sorted(self.blocks,
         key=lambda x: self.__canonical__[type(x).__name__]))

   def simplify(self):
      """
      Track down overall factors from the different blocks and absorb them into
      an overall prefactor. Ensure unique ordering among different instances of
      Block and combine Multiplicative ones.
      """
      for der in self.derivatives:
         self.factor *= der.factor
         der.factor = 1
      self.derivatives = list(sorted(self.derivatives, key=lambda x: x.mu))
      self._order()
      if self.flavours[0] == self.flavours[1] and \
         any(b.__class__ is dM for b in self.blocks):
         self.factor = Complex(0)
      # Factor together any repeated Block instances that are not instances
      # of NonMultiplicative. => __mul__ must be implemented!
      while True:
         for ib,block in enumerate(self.blocks):
            if ib>0:
               if isinstance(block, Multiplicative) and \
                  block.__class__==self.blocks[ib-1].__class__:
                  self.blocks[ib-1] = self.blocks[ib-1] * block
                  del self.blocks[ib]
                  break
         else:
            break
      for block in self.covl+self.blocks+self.covr:
         block.simplify()
         self.factor *= block.factor
         block.factor = 1

   def __str__(self):
      assert self.factor==1
      # Deal with preceding total derivatives.
      rep = ".".join(str(der) for der in self.derivatives) +\
         ("." if len(self.derivatives)>0 else "") + self.flavours[0] + "."
      for block in self.covl + self.blocks + self.covr:
         rep += str(block) + "."
      return rep + self.flavours[1]

   def charge(self):
      return Bilinear([b.charge() for b in self.blocks[::-1]],
         [r.charge() for r in self.covr[::-1]],
         [l.charge() for l in self.covl[::-1]], self.flavours[::-1],
         _copy(self.derivatives), self.factor)

   def chiralSpurion(self):
      return np.prod([b.chiralSpurion() for b in self.covl+self.blocks+self.covr])

   def reflection(self, mu:int):
      return Bilinear([b.reflection(mu) for b in self.blocks], 
         [l.reflection(mu) for l in self.covl],
         [r.reflection(mu) for r in self.covr], self.flavours,
         [der.reflection(mu) for der in self.derivatives], self.factor)

   def rotation(self, plane:int):
      return Bilinear([b.rotation(plane) for b in self.blocks],
         [l.rotation(plane) for l in self.covl],
         [r.rotation(plane) for r in self.covr], self.flavours,
         [der.rotation(plane) for der in self.derivatives], self.factor)
         
   def swapFlavours(self):
      """Exchange left and right flavour of Bilinear.

      Should be used to implement non-trivial flavour dependence for Bilinear
      under charge conjugation, e.g.,

         u.Gamma[gx].s -> - s.Gamma[gx].u  =>   J[u,s] -> - J[s,u]

      under charge. Enforcing such a transformation requires additional work
      since it is NOT a symmetry of the current. This becomes important for
      higher dimensional variants, e.g.,

         (m[u]-m[s])/2 d.u.s := d.u.dM.s ->  - d.s.dM.u =: -(m[s]-m[u])/2 d.s.u

      Using here the conventional symmetrisiation will then ensure that dM
      survives iff u <-> s are swapped after the charge conjugation.

      Returns
      -------
      Bilinear
         A copy of the current Bilinear with flavours exchanged.
      """
      cp = _copy(self)
      cp.flavours = cp.flavours[::-1]
      return cp


@massDim(0)
class Trace:
   """Cyclic permuting grouping of instances of Block.

   Contains a collection of instances of _AlgebraBlock (or when used internally
   also _dA) or (custom) non-algebra Block. Non commutativity is assumed but cyclic
   permutations are taken into account during simplify to give the Trace a unique
   string representation.

   **CAVEAT:** Intended for internal use and checks against __class__.

   Parameters
   ----------
   cyclic : list[Block]
      List of Block instances over which to trace.
   derivatives : list[d]|None, optional
      List of total derivatives acting on the Trace, defaults to None.
   factor : int|Fraction|Complex, optional
      Overall factor, defaults to 1.
   """
   def __init__(self, cyclic:list[Block], derivatives:list[d]|None=None,
                factor:int|Fraction|Complex = 1):
      self.cyclic      = cyclic
      self.factor      = factor
      self.derivatives = list() if derivatives is None else derivatives
      self._labels     = None

   def simplify(self):
      if len(self.cyclic)==1 and self.cyclic[0].__class__ is _AlgebraBlock:
         # _AlgebraBlock is assumed to be traceless!
         self.cyclic[0].simplify()
         self.cyclic[0].factor = 1
         for der in self.derivatives:
            der.factor = 1
         self.factor = Complex(0)
      else:
         for der in self.derivatives:
            self.factor *= der.factor
            der.factor = 1
         self.derivatives = list(sorted(self.derivatives, key=lambda x: x.mu))
         for c in self.cyclic:
            c.simplify()
            self.factor *= c.factor
            c.factor = 1
         labels = list([str(c) for c in self.cyclic])
         order = ".".join(labels)
         if self._labels == order: return
         idx = 0
         for i in range(1,len(labels)):
            test = ".".join(labels[i:] + labels[:i])
            if test<order:
               order = test
               idx = i
         self.cyclic = self.cyclic[idx:] + self.cyclic[:idx]
         # use as a buffer to avoid redoing the entire simplification
         self._labels = order

   def __str__(self):
      assert self.factor==1
      return ".".join(str(der) for der in self.derivatives) +\
         ("." if len(self.derivatives)>0 else "") +\
         "tr(" + ".".join(str(c) for c in self.cyclic) + ")"

   def charge(self):
      return Trace([c.charge() for c in self.cyclic[::-1]],
         _copy(self.derivatives), self.factor)

   def chiralSpurion(self):
      # adjust for Tr(M^n)...
      if len(self.cyclic)>0 and not self.cyclic[0] is _AlgebraBlock:
         isChiral = 1
         for c in self.cyclic:
            isChiral *= c.chiralSpurion()
         return -isChiral
      return 1

   def reflection(self, mu:int):
      return Trace([c.reflection(mu) for c in self.cyclic],
         [der.reflection(mu) for der in self.derivatives], self.factor)

   def rotation(self, plane:int):
      return Trace([c.rotation(plane) for c in self.cyclic],
         [der.rotation(plane) for der in self.derivatives], self.factor)


class _Commutative:
   """Collection of Trace and Bilinear that commute among one another.

   Allows to have various traces and bilinears simultaneously and deals with the
   commutativity of those pieces. Enforces a unique ordering among those pieces.

   **CAVEAT:** Intended for internal use and type hints.

   Parameters
   ----------
   prod : list[Bilinear|Trace]
      Collection of instances of Trace and Bilinear that commute among one another.
   derivatives : list[d]|None, optional
      Total derivatives acting on all the elements in *prod*, defaults to None.
   factor : int|Fraction|Complex, optional
      Overall factor, defaults to 1.
   """
   def __init__(self, prod:list[Bilinear|Trace], derivatives:list[d]|None=None,
                factor:int|Fraction|Complex = 1):
      self.prod        = prod
      self.factor      = factor if isinstance(factor, Complex) else\
                            Complex(factor)
      self.derivatives = list() if derivatives is None else derivatives

   def __mul__(self, fac:Complex):
      cp = _copy(self)
      cp.factor *= fac
      return cp

   def __rmul__(self, fac:Complex):
      return self.__mul__(fac)

   def __neg__(self):
      return self.__mul__(-1)

   def __str__(self):
      return ".".join(str(der) for der in self.derivatives) +\
         "<" + " ".join(str(p) for p in self.prod) + ">"

   def simplify(self):
      """
      Sort commuting Bilinears and Trace by their unique sring representation
      and absorb overall factors.
      """
      for der in self.derivatives:
         self.factor *= der.factor
         der.factor = 1
      ids = list()
      for p in self.prod:
         p.simplify()
         self.factor *= p.factor
         p.factor = 1
         ids.append(str(p))
      if len(self.prod) == 1:
         self.derivatives += _copy(self.prod[0].derivatives)
         self.prod[0].derivatives = []
      self.derivatives = list(sorted(self.derivatives, key=lambda x: x.mu))
      # change how to sort here!
      self.prod = [self.prod[x[1]] for x in sorted(zip(ids,range(len(ids))))]

   def charge(self):
      return self.__class__([p.charge() for p in self.prod],
         _copy(self.derivatives), self.factor)

   def chiralSpurion(self):
      return all(p.chiralSpurion()==-1 for p in self.prod)

   def reflection(self, mu:int):
      return self.__class__([p.reflection(mu) for p in self.prod],
         [der.reflection(mu) for der in self.derivatives], self.factor)

   def rotation(self, plane:int):
      return self.__class__([p.rotation(plane) for p in self.prod],
         [der.rotation(plane) for der in self.derivatives], self.factor)

   @property
   def bilinears(self):
      """Iterator over all `Bilinear` in current term.

      Allows the user direct access to all instances of `Bilinear` present in
      the current instance of _Commutative, i.e., within the current term.

      Returns
      -------
      Iterator[Bilinear]
         Iterator over all instances of `Bilinear` present in current term.
      """
      for p in self.prod:
         if p.__class__ is Bilinear:
            yield p

   @property
   def traces(self):
      """Iterator over all `Trace` in current term.

      Allows the user direct access to all instances of `Trace` present in the
      current instance of _Commutative, i.e., within the current term.

      Returns
      -------
      Iterator[Trace]
         Iterator over all instances of `Trace` present in current term.
      """
      for p in self.prod:
         if p.__class__ is Trace:
            yield p


class LinearComb:
   """
   Main object dealing with linear combinations of monomials of the fields,
   masses, Gamma structures, (covariant) derivatives etc. It allows proper
   book-keeping and tracking of identical terms up to prefactors to turn them
   into the appropriate simplified linear combination.

   **CAVEAT:** Intended for internal use and type hints.

   Parameters
   ----------
   terms : list[_Commutative]
      Collection of terms in the linear combination.
   factor : int|Fraction|Complex, optional
      Overall factor, defaults to 1.
   """
   def __init__(self, terms:list[_Commutative], factor:int|Fraction|Complex = 1):
      self.terms  = terms
      self.factor = factor if isinstance(factor,Complex) else Complex(factor)

   def __bool__(self):
      return self.factor != 0

   def __eq__(self, cmp):
      if cmp.__class__ != self.__class__:
         return False
      return str(self) == str(cmp)

   def __add__(self, add):
      moreTerms = _copy(add.terms)
      for term in moreTerms:
         term.factor *= add.factor
      oldTerms = _copy(self.terms)
      for term in oldTerms:
         term.factor *= self.factor
      return self.__class__(oldTerms + moreTerms)

   def __sub__(self, sub):
      moreTerms = _copy(sub.terms)
      for term in moreTerms:
         term.factor *= -sub.factor
      oldTerms = _copy(self.terms)
      for term in oldTerms:
         term.factor *= self.factor
      return self.__class__(oldTerms + moreTerms)

   def __mul__(self, fac):
      """
      Implements the multiplication of `LinearComb` with an overall factor or
      another isntance of `LinearComb`. In the latter case all combinatorics
      are taken care of and overall derivatives present in any of the terms
      are written explicitly via the product rule. A call to
      `LinearComb.simplify` may be required to combine terms differing at most
      by their prefactors.

      Parameters
      ----------
      fac : int|Fraction|Complex|LinearComb
         Second factor to multiply with.

      Returns
      -------
      LinearComb
         Result of the multiplication. The initial `LinearComb` remains
         unchanged and independent of the returned value.
      """
      cp = _copy(self)
      if isinstance(fac, LinearComb):
         while _productRule(cp): continue
         fcp = _copy(fac)
         while _productRule(fcp): continue
         newTerms = list()
         for fterm in fcp.terms:
            for sterm in cp.terms:
               newTerms.append(
                  _Commutative(_copy(sterm.prod) + _copy(fterm.prod), None,
                              sterm.factor * fterm.factor))
         return LinearComb(newTerms, self.factor*fac.factor)
      if isinstance(fac, (int,Fraction,Complex)):
         cp.factor *= fac
         return cp
      return NotImplemented

   def __rmul__(self, fac:int|Fraction|Complex):
      return self.__mul__(fac)

   def __neg__(self):
      cp = _copy(self)
      cp.factor = -cp.factor
      return cp

   def __str__(self):
      """
      Assigns a unique string to the current instance of `LinearComb`.
      
      .. important::
         Make sure to always go through `LinearComb.__str__` and not the
         sub-elements!

      Returns
      -------
      str
         Unique string representation.
      """
      if not isinstance(self.factor, Complex):
         self.factor = Complex(self.factor)
      if self.factor.abs2():
         rep = "%s*{\n"%str(self.factor)
         for term in self.terms:
            rep += whiteSpace + "%s*"%str(term.factor) + str(term) + "\n"
         return rep + "}"
      return "+0"

   @staticmethod
   def ZERO():
      """
      Should be used to define a literal zero, e.g., when planning to sum up
      various instances of `LinearComb`.

      Returns
      -------
      LinearComb
         Representation of zero in terms of `LinearComb`.
      """
      return LinearComb(list(), 0)

   def simplify(self):
      """
      Combine identical terms differing at most in their prefactors. And impose
      a unique ordering.
      """
      for term in self.terms:
         term.simplify()
      ids = []
      factors = []
      for term in self.terms:
         ids.append(str(term))
         factors.append(term.factor)
      newFactors = []
      reorder = sorted(set(ids))
      for id_ in reorder:
         ofs = 0
         idx = list()
         # there must be a better way...
         for i in range(ids.count(id_)):
            ofs = ofs + ids[ofs:].index(id_)
            idx.append(ofs)
            ofs += 1
         newFactors.append(self.factor * sum(factors[x] for x in idx))
      temp = [x for x in newFactors if x!=0]
      if len(temp) == 0:
         self.factor = Complex(0)
         self.terms = list()
         return
      # By convention is the coefficient of the first term +1
      self.factor = temp[0]
      norm = Complex(1) / temp[0]
      pcs = list()
      for i,id_ in enumerate(reorder):
         idx = ids.index(id_)
         if newFactors[i]:
            pcs.append(_copy(self.terms[idx]))
            pcs[-1].factor = newFactors[i]*norm
      self.terms = pcs

   def charge(self):
      return self.__class__([t.charge() for t in self.terms], self.factor)

   def chiralSpurion(self):
      return all(t.chiralSpurion() for t in self.terms)

   def reflection(self, mu:int):
      return self.__class__([t.reflection(mu) for t in self.terms],
                            self.factor)

   def rotation(self, plane:int):
      return self.__class__([t.rotation(plane) for t in self.terms],
                            self.factor)

   @property
   def bilinears(self):
      """Iterator over all Bilinear in current expression.

      Allows the user direct access to all instances of `Bilinear` present in the
      current instance of LinearComb, i.e., agnostic to which term each
      Bilinear belongs to.

      Returns
      -------
      Iterator[Bilinear]
         Iterator over all instances of `Bilinear` present in LinearComb.
      """
      for term in self.terms:
         for p in term.prod:
            if p.__class__ is Bilinear:
               yield p

   @property
   def traces(self):
      """Iterator over all `Trace` in current expression.

      Allows the user direct access to all instances of Trace present in the
      current instance of LinearComb, i.e., agnostic to which term each `Trace`
      belongs to.

      Returns
      -------
      Iterator[Trace]
         Iterator over all instances of `Trace` present in LinearComb.
      """
      for term in self.terms:
         for p in term.prod:
            if p.__class__ is Trace:
               yield p


def _productRule(op:LinearComb)->bool:
   """
   Searches for an overall derivative in `_Commutative` to rewrite it via the
   product rule as an appropriate `LinearComb`.

   Parameters
   ----------
   op : LinearComb
      Expression (potentially) containing overall derivatives in some terms.
      **Will be modified!**

   Returns
   -------
   bool
      Indicates whether an overall deriative was encountered and replaced.
   """
   for it,term in enumerate(op.terms):
      # apply product rule
      if len(term.derivatives)>0:
         mu = term.derivatives[0].mu
         del term.derivatives[0]
         for ip,p in enumerate(term.prod):
            # traces carrying custom Block are assumed to be constants!
            if p.__class__ is Trace and\
               not p.cyclic[0].__class__ is _AlgebraBlock: continue
            temp = _copy(term)
            temp.prod[ip].derivatives.append(d(mu))
            op.terms.append(temp)
         del op.terms[it]
         return True
   return False
