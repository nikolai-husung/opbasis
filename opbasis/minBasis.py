"""
This submodule takes care of handling covariant derivatives, field strength
tensors and overall derivatives in a common way to be able to work out a basis
that is fully linearly independent and thus minimal.
"""

from copy import deepcopy as _copy
 
from .basics import indices, sptIdx
from .blocks import d, F, D, Dl, Colour, DF, D0, D0l
from .calculus import Complex, Matrix
from .ops import LinearComb, Block, _AlgebraBlock, Trace, Bilinear, _productRule
from .eoms import GluonEOM, unmaskGluonEOMs, unmaskFermionEOMs
from .opBasis import Model

@indices("[%s]", mu=sptIdx)
class _dA(Block):
   """
   Intended for internal use only to give gluon Trace a unique string
   representation in terms of monomials of the gluon field and its
   (higher order) derivatives.

   Parameters
   ----------
   mu : sptIdx
      Space-time index of the gluon.
   derivatives : list[d]|None, optional
      Collection (space-time indices) of derivatives acting on the gluon.
      Defaults to `None`.
   """
   def __init__(self, mu:sptIdx, derivatives:list[int]|None = None,
      factor:Complex = 1):
      self.factor = factor
      self.mu = mu
      self.derivatives = list() if derivatives is None else derivatives

   def __str__(self):
      assert self.factor==1
      rep = "].d[".join(str(x) for x in self.derivatives)
      if len(rep)>0: rep = "d[" + rep + "]"
      return rep + "_dA(%i)"%self.mu

   def simplify(self):
      self.derivatives = list(sorted(self.derivatives))


def _unmaskAlgebraBlock(op:LinearComb)->bool:
   """
   Looks for an instance of `_AlgebraBlock` and reexpresses it by the
   appropriate commutators of covariant derivatives acting on the right-most
   algebra block, which can be a field strength tensor or `Colour`.

   Parameters
   ----------
   op : LinearComb
      Expression in which `_AlgebraBlock` are supposed to be reexpressed.

      .. caution::
         The input argument will be changed!

   Returns
   -------
   bool
      Indicating whether an instance of `_AlgebraBlock` was encountered.
   """
   for it,term in enumerate(op.terms):
      for ip,p in enumerate(term.prod):
         # distinguish between Trace and Bilinear!
         if p.__class__ is Bilinear:
            # run backwards...
            ofs = len(p.blocks)-1
            for ib,b in enumerate(p.blocks[::-1]):
               if b.__class__ is _AlgebraBlock:
                  seed = [[Colour()]] if b.blocks[-1].__class__ is Colour else \
                         [[D(b.blocks[-1].mu),D(b.blocks[-1].nu)], \
                          [D(b.blocks[-1].nu),D(b.blocks[-1].mu,-1)]]
                  for ab in b.blocks[-2::-1]:
                     newSeed = list()
                     for s in seed:
                        newSeed.append([D(ab.mu)]+s)
                        newSeed.append(s+[D(ab.mu,-1)])
                     seed = newSeed
                  for s in seed:
                     temp = _copy(term)
                     del temp.prod[ip].blocks[ofs-ib]
                     temp.prod[ip].covr = _copy(s) + temp.prod[ip].covr
                     op.terms.append(temp)
                  del op.terms[it]
                  return True
         if p.__class__ is Trace:
            for ib,b in enumerate(p.cyclic):
               if b.__class__ is _AlgebraBlock:
                  seed = [[_dA(b.blocks[-1].nu, [b.blocks[-1].mu])], \
                          [_dA(b.blocks[-1].mu, [b.blocks[-1].nu], -1)],
                          [_dA(b.blocks[-1].mu), _dA(b.blocks[-1].nu)],
                          [_dA(b.blocks[-1].nu), _dA(b.blocks[-1].mu, None, -1)]]
                  for ab in b.blocks[-2::-1]:
                     newSeed = list()
                     for s in seed:
                        newSeed.append([_dA(ab.mu)]+s)
                        newSeed.append(s+[_dA(ab.mu, None, -1)])
                        for is_ in range(len(s)):
                           temp = _copy(s)
                           temp[is_].derivatives.append(ab.mu)
                           newSeed.append(temp)
                     seed = newSeed
                  for s in seed:
                     temp = _copy(term)
                     temp.prod[ip].cyclic = temp.prod[ip].cyclic[:ib] +\
                        _copy(s) + temp.prod[ip].cyclic[ib+1:]
                     op.terms.append(temp)
                  del op.terms[it]
                  return True
   return False


def _IBPbilinear(op:LinearComb)->bool:
   """
   Absorb one left-acting covariant derivative in a Bilinear into an overall
   derivative and a right-acting covariant derivative.
   
   Parameters
   ----------
   op : LinearComb
      Operator in which each Bilinear, carrying covariant derivatives acting to
      the left, is rewritten as total derivatives and right-acting covariant
      derivatives.
   
   Returns
   -------
   bool
      Indicates whether a left-acting covariant derivative was encountered and
      replaced.
   """
   for it,term in enumerate(op.terms):
      for ip,p in enumerate(term.prod):
         if p.__class__ is Bilinear and len(p.covl)>0:
            mu = p.covl[-1].mu
            del p.covl[-1]
            for ib,b in enumerate(p.blocks):
               if b.__class__ is _AlgebraBlock:
                  temp = _copy(term)
                  temp.prod[ip].blocks[ib].blocks = \
                     [D(mu)] + temp.prod[ip].blocks[ib].blocks
                  temp.factor = -temp.factor
                  op.terms.append(temp)
            temp = _copy(term)
            temp.prod[ip].covr = [D(mu)] + temp.prod[ip].covr
            temp.factor = -temp.factor
            op.terms.append(temp)
            p.derivatives.append(d(mu))
            return True
   return False


def _IBPtrace(op:LinearComb):
   """
   Searches for a total derivative acting on a algebra-valued `Trace` and
   replaces it by an appropriate combination of covariant derivatives.

   Parameters
   ----------
   op : LinearComb
      Operator to be reexpressed.

   Returns
   -------
   bool
      Indicates whether a total derivative was encountered and reexpressed.
   """
   for it,term in enumerate(op.terms):
      for ip,p in enumerate(term.prod):
         if p.__class__ is Trace and p.cyclic[0].__class__ is _AlgebraBlock \
            and len(p.derivatives)>0:
            mu = p.derivatives[-1].mu
            del p.derivatives[-1]
            for ic in range(len(p.cyclic)):
               temp = _copy(term)
               temp.prod[ip].cyclic[ic].blocks = [D(mu)] + \
                  temp.prod[ip].cyclic[ic].blocks
               op.terms.append(temp)
            del op.terms[it]
            return True
   return False


def _toRep(op:LinearComb):
   """
   Takes care of any total derivatives by applying the product rule and
   IBP relations. Afterwards, any `_AlgebraBlock` is written out explicitly
   and the resulting unique expression is prepared for being mapped in our
   vector-space.

   Parameters
   ----------
   op : LinearComb
      Operator to be processed.

   Returns
   -------
   list[tuple[str,Complex]] | None
      Returns the details needed to map the operator into our vector-space.
   """
   rep = _copy(op)
   while _productRule(rep): continue
   while _IBPbilinear(rep): continue
   while _IBPtrace(rep): continue
   rep.simplify()
   while _unmaskAlgebraBlock(rep): continue
   rep.simplify()
   return None if rep.factor==0 or len(rep.terms)==0 else\
      list((str(t),rep.factor*t.factor) for t in rep.terms) 


def _mapCoefficients(rep:list[tuple[str,Complex]], map_:dict[str,int]):
   """
   Each unique term in the overcomplete basis can be identified as an
   eigenvector in a vector space spanned by all the terms. Using the factors
   corresponding to the different terms in a `LinearComb`, we can thus assign
   each operator a vector in this vector-space to establish a truly linearly
   independent and thus minimal basis.

   Parameters
   ----------
   rep : list[tuple[str,Complex]]
      String-representations of all the terms of an operator together with the
      corresponding factor.
   map_ : dict[str,int]
      Table mapping each string-representation of a term into a component
      of the full #terms-dimensional vector.

   Returns
   -------
   list[Complex|int|Fraction]
      Eigenvector corresponding to `rep` in the full vector-space.
   """
   evec = [0]*len(map_)
   for key,factor in rep:
      evec[map_[key]] = factor
   return evec

def _mapToVectors(bases:list[LinearComb]):
   _bases = list(list(_toRep(op) for op in basis) for basis in bases)

   # Obtain unique identifiers to write down eigenvectors.
   # Enforce future ordering for 
   uniqueLabels = set()
   for basis in _bases:
      for rep in basis:
         if rep is None:
            continue
         l,e = list(zip(*rep))
         uniqueLabels |= set(l)
   uniqueLabels = tuple(uniqueLabels)
   idxMap = dict(zip(uniqueLabels, range(len(uniqueLabels))))
   
   evecs = list()
   for basis in _bases:
      tempBasis = list()
      for rep in basis:
         tempBasis.append(None if rep is None else \
            _mapCoefficients(rep, idxMap))
      evecs.append(tempBasis)
   return evecs, idxMap

def findMinBases(bases:list[list[LinearComb]], gEOM:GluonEOM|None=None,
                 fEOMs:list[LinearComb]|None=None, model:Model|None=None):
   """ 
   Takes the (various) overcomplete bases and identifies appropriate minimal
   bases while dropping linear dependent terms. The ordering of the sets of
   bases and within each basis determine which operators to keep. Operators
   with higher priority should come first.
   
   Optionally allows to start from EOMs still being masked by `D0`, `D0l`, and
   `DF`.

   In which case the appropriate EOMs must be provided:

   * *fEOMs* & *model* or both `None`
   * *gEOM* or `None`

   .. attention::
      If any EOMs are `None`, please make sure that all `D0`, `D0l`, and `DF`
      are unmasked!

   Parameters
   ----------
   bases : list[list[LinearComb]]
      Collection of operators that are still grouped to allow for keeping track
      of initially used templates.
   gEOM : GluonEOM|None, optional
      Description of the gluon EOM, defaults to `None`.
   fEOMs : list[LinearComb]|None, optional
      Collection of fermion EOMs, i.e., allows to have different EOMs for
      different flavours. Defaults to `None`.
   model : Model
      Description of the theory currently considered. Only required for use with
      the fermion EOMs to track sets of flavours. Defaults to `None`.

   Returns
   -------
   list[list[LinearComb]]
      Collection of linearly independent operators keeping the original grouping
      intact while ensuring that earlier groups take precedence.

   Raises
   ------
   AssertionError
      If fermion EOMs are provided but *model* is `None`.
   """
   _bases = _copy(bases)
   labels = tuple([str(op) for op in basis] for basis in bases)
   
   assert not ((not fEOMs is None) and model is None)

   if not fEOMs is None:
      for fEOM in fEOMs:
         for basis in _bases:
            for op in basis:
               unmaskFermionEOMs(op, fEOM, model)
   if not gEOM is None:
      for basis in _bases:
         for op in basis:
            unmaskGluonEOMs(op, gEOM)

   evecs,_ = _mapToVectors(_bases)
   
   evecsIndep = Matrix()
   minBases = list()
   lastRank = 0
   for ile,elems in enumerate(evecs):
      linIndep = list()
      for i,evec in enumerate(elems):
         # Probably highly inefficient but does the job.
         if evec is None: continue
         evecsIndep.extend(evec, Matrix.ROW)
         newrank = evecsIndep.rank(True)
         if lastRank<newrank:
            lastRank += 1
            linIndep.append(bases[ile][i])
      minBases.append(linIndep)
   return minBases
