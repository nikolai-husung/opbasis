"""
This submodule allows the user to input the desired `Model` specifying the
desired transformations and transformation-behaviour of the future operator
basis. Part of a `Model` are also declarations of any custom implementations
of `Block`. Finally, classes of operators can be defined as string-valued so
called *templates*, which are then parsed into ansätze for the desired operator
basis. After imposing the specified transformation-behaviour through a call
to `overcompleteBasis` this yields a yet overcomplete basis.
"""

from dataclasses import dataclass
from fractions import Fraction
from importlib import import_module as _import
from copy import deepcopy as _copy
from operator import and_
from collections.abc import Callable

import re

from itertools import product, combinations_with_replacement as cwr

from typing import Iterator

from .blocks import Block, d, D, Dl, D0, D0l,\
   _AlgebraBlock, F, DF, Colour,\
   M, dM
from .dirac import Gamma
from .pauli import SU2
from .ops import LinearComb, Commutative, Bilinear, Trace
from .basics import Complex, rotPlanes, Union

# end-points for _AlgebraBlock templates
_ALG = "|".join(["F","DF"])

_PROTECTED = ["Block", "d", "D", "Dl", "F", "DF", "D0", "D0l",
   "Colour", "M", "dM", "Gamma", "SU2", "Multiplicative",
   "LinearComb", "Commutative", "Bilinear", "Trace"]

def _checkProtected(name:str, type_:str, protected:list[str]):
   """
   Checks if a custom name has been used prior or is *protected*. The latter
   is by convention true for any names starting with an underscore or are names
   reserved for default implementations of `Block`. If no problems are found
   the name is added to the list of used names.

   Parameters
   ----------
   name : str
      Name proposed by the user.
   type_ : str
      What user-defined object carries that name.
   protected : list[str]
      Collection of names already being in use.

   Raises
   ------
   ValueError
      If `name` is already being used or otherwise *protected*.
   """
   if name.startswith("_") or name in protected:
      raise ValueError(
         "Chosen %s name '%s' is protected or already defined."%(type_,name))
   protected.append(name)

@dataclass
class Model:
   """
   Represents a theory with optional flavour contant and a given set of
   transformations that keep the action invariant. Any composite field
   may transform non-trivially as selected by the transformation behaviour
   accompanying each transformation.

   **CAUTION:** Not intended for direct use. The user should  go through the
   static method `read` or `parse` instead.
   """
   pyName: str
   blocks: list[str]
   transf: dict[str,str]
   spurion: list[str]
   flavourSets: dict[str,list[str]]
   templates: list[str]
   pyModule: object

   def toReference(self):
      """
      Turns model description into a string discarding any defined templates.
      May be used to store alongside any found operator bases to keep track of
      the model being used at the time, i.e., for reproducibility.

      Returns
      -------
      tuple[str,str]
         String-representation of the current model w/o any mentioning of the
         templates as well as the file-content of the user-provided Python
         module.
      """
      m = ""
      for b in sorted(self.blocks):
         m += "Block: %s\n"%b
      for f in sorted(self.flavourSets.keys()):
         m += "Flavours: %s = %s\n"%(" ".join(self.flavourSets[f]), f)
      for t in sorted(self.transf.keys()):
         m += "Discrete: %s %s\n"%(t,self.transf[t])
      for t in sorted(self.spurion):
         m += "Spurion: %s\n"%t
      with open("transformations/%s.py"%self.pyName, "r") as f:
         content = f.read()
      return (m,content)

   @staticmethod
   def read(fname:str):
      """Load model from file.

      Reads the description of a model, i.e., the underlying theory to be
      considered from a file with name *fname* and hands it over to
      `Model.parse`.

      Parameters
      ----------
      fname : str
         File name to read the Model from.

      Returns
      -------
      Model
         Python representation of the chosen theory.
      """
      fid = open(fname, "r")
      content = fid.read()
      fid.close()
      return Model.parse(content)

   @staticmethod
   def parse(content:str):
      """Parses model from string.

      Parses the description of a model, i.e., the underlying theory to be
      considered from the string *content*. Checks for consistency of chosen
      custom Block names.

      Parameters
      ----------
      content : str
         String to parse the Model from.

      Returns
      -------
      Model
         Python representation of the chosen theory.

      Raises
      ------
      AssertionError
         If some basic formatting is wrong or more than one *Transf* section
         is given.
      ValueError
         If a custom name coincides with internal naming or is being used more
         than once.
      NotImplementedError
         If an unknown option is encountered.
      """
      lines = content.split("\n")
      for i in range(len(lines)):
         lines[i] = lines[i].split("#")[0]
      pyName = None
      temps = list()
      blocks = list()
      eom = list()
      transfs = dict()
      spurions = list()
      fs = dict()
      temps = list()
      protected = _copy(_PROTECTED)
      for line in lines:
         if not ":" in line:
            assert line.replace(" ", "") == ""
            continue
         cat,opt = line.split(":", maxsplit=1)
         cat = cat.replace(" ", "")
         if cat == "Op":
            temps.append(opt.replace(" ", ""))
         elif cat == "Discrete":
            temp = [x for x in opt.split(" ") if x!=""]
            test = "".join(temp[1:])
            assert all(x in "x+-" for x in test)
            transfs[temp[0]] = test
         elif cat == "Spurion":
            temp = [x for x in opt.split(" ") if x!=""]
            assert len(temp) == 1
            spurions.append(temp[0])
         elif cat == "Block":
            b = opt.replace(" ", "")
            _checkProtected(b, "custom Block", protected)
            blocks.append(b)
         elif cat == "Flavours":
            f,s = opt.split("=")
            fset = s.replace(" ", "")
            _checkProtected(fset, "flavour vector", protected)
            fs[fset] = [x for x in f.split(" ") if x!=""]
            for f in fs[fset]:
               _checkProtected(f, "flavour", protected)
         elif cat == "Transf":
            assert pyName is None
            pyName = opt.replace(" ", "")
         else:
            raise NotImplementedError(
               "Option \"%s\" used in model file does not exist!"%cat)
      return Model(pyName, blocks, transfs, spurions, fs, temps, None)

   def apply(self):
      """Replaces the currently used Model.

      Enforces the appropriate sorting convention within a `Bilinear`. Since it
      depends on the current Model it has to be called at any time the Model
      is being changed. Loads the Python module corresponding to this Model
      thus making custom `Block` and transformations accessible.
      """
      if self.pyModule is None:
         self.pyModule = _import("transformations.%s"%self.pyName)
         self._checkBlockConsistency()
      hierarchy = dict()
      for counter,tb in enumerate(
         ["dM", "M", "Gamma", "SU2"] + self.blocks + ["_AlgebraBlock"]):
         hierarchy[tb] = counter
      Bilinear.__canonical__ = hierarchy

   def _checkBlockConsistency(self):
      """
      Checks that any indices provided for all custom Block implementations
      match our expectations and the naming of class attributes follows our
      conventions. For more details what those are, see `Block`.

      Raises
      ------
      AssertionError
         If types do not match our expectations.
      """
      for block in self.blocks:
         cls = getattr(self.pyModule, block)
         v = next(cls.variants())
         for n,t in zip(cls.__indexName__, cls.__indexType__):
            assert getattr(v, n).__class__ is t,\
               "Type mismatch for attribute %s: not %s is %s."%(n,
                  str(getattr(v, n)).__class__, str(t))
         t = v.factor.__class__
         assert t is int or t is Fraction or t is Complex,\
            "`factor` has unexpected type."
         del v


def _parsePart(template:str, model:Model)->Iterator[Trace|Bilinear]:
   """Parse string representation of Trace or Bilinear.

   Parses the given expression into either `Trace` or `Bilinear` with all
   possible variants. If the expression does not match any of the expected
   patterns an error is raised.

   Parameters
   ----------
   template : str
      String representation of a template for a Bilinear or Trace to be used.
   model : Model
      Python descriptor of chosen theory.

   Returns
   -------
   Iterator[Bilinear|Trace]
      Iterator over all possible variants matching the parsed *template*, e.g.,
      different indices.
   
   Raises
   ------
   ValueError
      If the given *template* does not match any regular expression, something
      is wrong with the formatting.
   """
   test = re.search("^(d\\.)+", template)
   if test:
      dparts = template[:test.end()].count("d")
      derivatives = list(cwr(d.variants(), dparts))
      template = template[test.end():]
   else:
      derivatives = [()]
   # Allow more general user-defined cases, e.g., involving M and Mu etc.
   mask = "(" + "|".join(model.blocks + ["M", "dM", "Gamma", "SU2"]) + ")"
   # Allow for different flavours which need to be declared!
   maskAlg = "(" + "|".join(model.blocks +\
      ["M", "dM", "Gamma", "SU2", "(D\\.)*(%s)"%_ALG]) + ")"
   flavourMask = "(" + "|".join("|".join([key]+flavours) \
      for key,flavours in model.flavourSets.items()) + ")"
   if re.match("^tr\\((%s\\.)*%s\\)$"%(mask,mask), template):
      # too general?
      for dd in derivatives:
         for x in product(*[getattr(model.pyModule, b).variants() \
            if b in model.blocks \
            else globals()[b].variants() for b in template[3:-1].split(".")]):
            yield Trace(_copy(list(x)), _copy(list(dd)))
   elif re.match("^tr\\(((D\\.)*(%s)\\.)+((D\\.)*(%s))\\)$"%(
                 _ALG,_ALG), template):
      ab = re.search("(D\\.)*(%s)"%_ALG, template)
      cv = list()
      while ab:
         cv.append([_AlgebraBlock(_copy(list(var))) for var in product(
            *[globals()[p].variants() for p in ab.group().split(".")])])
         template = template[ab.end()+1:]
         ab = re.search("(D\\.)*(%s)"%_ALG, template)
      for cyclic in product(*cv):
         for dd in derivatives:
            yield Trace(_copy(list(cyclic)), _copy(list(dd)))
   elif re.match("^tr\\(((D\\.)*(%s)\\.)*(D\\.)*Colour(\\.(D\\.)*(%s))+\\)$"%(
                 _ALG,_ALG),
        template)\
     or re.match("^tr\\(((D\\.)*(%s)\\.)+(D\\.)*Colour(\\.(D\\.)*(%s))*\\)$"%(
                 _ALG,_ALG),
      template):
      colour = re.search("(D\\.)*Colour")
      template = template[colour.end()+1:]+"."+template[3:colour.start()-1]
      ab = re.search("(D\\.)*(%s)"%_ALG, template)
      cv = list()
      while ab:
         cv.append([_AlgebraBlock(_copy(list(var))) for var in product(
            *[globals()[p].variants() for p in ab.group().split(".")])])
         template = template[ab.end()+1:]
         ab = re.search("(D\\.)*(%s)"%_ALG, template)
      for cyclic in product(*cv):
         for Dvar in product(*[globals()[b].variants() \
            for b in template[colour.start():colour.end()].split(".")]):
            for dd in derivatives:
               yield Trace(_copy(list(cyclic)) + \
                  [_AlgebraBlock(_copy(list(Dvar)))], _copy(list(dd)))
   elif re.match(("^%s\\.(D0l\\.)*(Dl\\.)*(%s\\.)*((D\\.)*" +
                  "(Colour\\.)){0,1}(%s\\.)*(D\\.)*(D0\\.)*%s$")%(
                 flavourMask,maskAlg,maskAlg,flavourMask), template):
      left = re.search("^%s\\.(D0l\\.)*(Dl\\.)*"%flavourMask, template)
      lp = left.group()[:-1].split(".")
      lpv = [list(globals()[p].variants()) for p in lp[1:]] if len(lp)>1 else []
      right = re.search("\\.(D\\.)*(D0\\.)*%s$"%flavourMask, template)
      rp = right.group()[1:].split(".")
      rpv = [list(globals()[p].variants()) for p in rp[:-1]] if len(rp)>1 else []
      fl = (lp[0],rp[-1])
      ab = re.search("\\.(D\\.)*(%s|Colour)\\."%_ALG, template)
      other = re.search("\\." + mask + "\\.", template)
      cv = list()
      while True:
         if ab and (not other or ab.start() < other.start()):
            cv.append([_AlgebraBlock(_copy(list(var))) for var in product(
              *[globals()[p].variants() for p in ab.group()[1:-1].split(".")])])
            template = template[ab.end()-1:]
         elif other:
            b = other.group()[1:-1]
            cv.append(getattr(model.pyModule, b).variants() \
               if b in model.blocks \
               else globals()[b].variants())
            template = template[other.end()-1:]
         else:
            break
         ab = re.search("\\.(D\\.)*(%s|Colour)\\."%_ALG, template)
         other = re.search("\\." + mask + "\\.", template)
      for blocks in product(*cv):
         for covl in product(*lpv):
            for covr in product(*rpv):
               for dd in derivatives:
                  yield Bilinear(_copy(list(blocks)), _copy(list(covl)),
                                 _copy(list(covr)), fl, _copy(list(dd)))
   else:
      raise ValueError("Something is wrong with the template *%s*."%template)


def parseAnsatz(template:str, model:Model)->Iterator[LinearComb]:
   """Turn *template* into all possible ansätze.

   Translates the given *template* into all variants matching that form. In
   turn those variants form all the possible ansätze used to build a basis of
   operators with appropriate transformation properties.

   Parameters
   ----------
   template : str
      String representation of the template describing ansätze to be used.
   model : Model
      Python descriptor of chosen theory.
   
   Returns
   -------
   Iterator[LinearComb]
      Iterator over all possible variants matching the parsed *template*, e.g.,
      different indices.
   
   Raises
   ------
   ValueError
      Indicates that the used *template* is badly formatted, e.g., due to
      missing custom Block definitions.
   """
   test = re.search("^(d\\.)*d(\\.){0,1}\\{[^{}]*\\}$", template)
   if not test and ("{" in template or "}" in template):
      raise ValueError("Badly formatted template *%s*."%template)
   elif test:
      td = re.search("^(d\\.)*d(\\.){0,1}\\{", template)
      dparts = template[:td.end()-1].count("d")
      derivatives = list(cwr(d.variants(), dparts))
      template = template[td.end():-1]
   else:
      derivatives = [()]
   for c in product(*[_parsePart(p, model) for p in template.\
      replace(" ", "").split("*")]):
      for dd in derivatives:
         yield LinearComb([Commutative(_copy(list(c)), _copy(list(dd)), 1)], 1)


def symmetrise(ansatz:LinearComb, model:Model)->LinearComb:
   """Iteratively enforce transformation properties on a given *ansatz*.

   Takes an initial ansatz, i.e., a linear combination (or more commonly a
   monomial) and performs the specified transformations. If the ansatz
   transforms already properly, nothing happens. Otherwise the new
   contribution is stored.

   Eventually all contributions are added up to enforce compliance with the
   specified transformations. If the result does not vanish, it is compatible.

   Parameters
   ----------
   ansatz : LinearComb
      Python representation of the (partially) symmetrised ansatz.
   model : Model
      Descriptor of the currently assumed theory including the desired
      transformation properties.
   
   Returns
   -------
   LinearComb
      Further symmetrised variant of the initial ansatz.
   """
   anyNew = True
   coll = Union()
   coll.add(ansatz)
   while anyNew:
      anyNew = False
      for c in coll:
         negc = -c
         for transf,behaviour in model.transf.items():
            for i,b in enumerate(behaviour):
               if b=="x": continue
               temp = _copy(c)
               while True:
                  temp = (1 if b=="+" else -1) * \
                         getattr(model.pyModule, transf)(temp, i)
                  temp.simplify()
                  if temp == c: break
                  # Any occurrence of the negative of an initial ansatz implies
                  # overall zero.
                  if temp == negc:
                     ansatz.factor = 0
                     return ansatz
                  anyNew |= coll.add(temp)
   coll = iter(coll)
   ansatz = next(coll)
   for c in coll:
      ansatz += c
   ansatz.simplify()
   if ansatz:
      ansatz.factor = Complex(1)
   return ansatz


def overcompleteBasis(template:str, model:Model)->list[LinearComb]:
   """Find all ansätze compatible with the desired transformation properties.

   Produces all possibilities of spacetime indices, gamma structures etc. in
   accordance with the given template *template*. The appropriate basis is then
   worked out using all symmetries listed in *model* and applying vetoing from
   Spurion symmetry checks.

   Parameters
   ----------
   template : str
      String representation of the currently considered ansatz.
   model : Model
      Descriptor of the currently assumed theory including the desired
      transformation properties.

   Returns
   -------
   list[LinearComb]
      All (unique) expressions surviving the transformation requirements and
      all Spurion symmetry checks.
   """
   basis = Union()
   for ansatz in parseAnsatz(template, model):
      ansatz.simplify()
      ansatz = symmetrise(ansatz, model)
      if ansatz.factor == 0: continue
      if not all(getattr(model.pyModule, sp)(ansatz) for sp in model.spurion):
         continue
      if ansatz != 0:
         basis.add(ansatz)
   return basis.content


def _prepareAlgebraBlockTemplates(nb:int, mdTarget:int):
   """
   Distributes covariant derivatives among all the `nb` blocks remaining.

   Parameters
   ----------
   nb : int
      Number of blocks (remaining).
   mdTarget : int
      Non-negative integer specifying the desired mass-dimension of all the
      covariant derivatives to be distributed.

   Returns
   -------
   Iterator[list[int]]
      Collection of all possible ways to distribute covariant derivatives among
      the `nb` blocks.
   """
   if nb>0:
      for i in range(mdTarget//D.__massDim__ + 1):
         for x in _prepareAlgebraBlockTemplates(nb-1,
            mdTarget - i*D.__massDim__):
            yield [i] + x
   elif mdTarget == 0:
      yield []
   

def getAlgebraTraceTemplates(mdTarget:int, sorting=None)->list[str]:
   """
   Returns all templates for algebra traces of mass-dimension `mdTarget` except
   for total derivatives. The latter can be obtained from their lower
   dimensional variants simply by adding an appropriate number of derivatives.
   Cyclicity of the `Trace` is already taken into account to reduce the overall
   number of templates.

   Parameters
   ----------
   mdTaget : int
      Non-negative integer specifying the desired mass dimension.
   sorting : Callable[[], int], optional
      Can be used to adjust the ordering within the templates. Defaults to None.

   Returns
   -------
   list[str]
      All the templates for alegbra traces relevant at this mass-dimension,
      which are not total derivatives.

   Raises
   ------
   AssertionError
      If `mdTarget` is negative.
   """
   assert mdTarget>=0, "mdTarget has to be a non-negative integer."
   templates = set()
   mdAlgMin = F.__massDim__ if F.__massDim__ < DF.__massDim__ else \
      DF.__massDim__
   if sorting is None:
      sorting = lambda x: (
         + (mdTarget+1) * x[1][DF]
         -                x[1][F]
         )
   for nb in range(2, mdTarget // mdAlgMin + 1):
      for blocks in map(list, product([F,DF], repeat=nb)):
         md = mdTarget - sum(map(lambda x: x.__massDim__, blocks))
         if md<0: continue
         for abt in _prepareAlgebraBlockTemplates(nb, md):
            temp = list(map(lambda x: (tuple(["D"]*x[0]), x[1]), zip(abt,blocks)))
            # drop duplicates by cyclicity
            template = [tuple(temp)]
            for i in range(1,nb):
               template.append(tuple(temp[i:]+temp[:i]))
            templates.add(tuple(sorted(template, key=str)))
   templates = [t[0] for t in templates]
   _sort = [{
      DF : sum(1 for x in t if x[1] is DF),
      F  : sum(1 for x in t if x[1] is F)
      } for t in templates]
   templates = [(
      "tr(" + ".".join(".".join(list(y[0])+[y[1].__name__]) for y in x[0])+ ")",
      x[1]) for x in sorted(zip(templates, _sort), key=sorting, reverse=True)]
   return (
      [t[0] for t in templates if t[1][DF]>0],
      [t[0] for t in templates if t[1][DF]==0]
      )


def _getCustomBlockTemplates(cblocks:list[type], mdTarget:int):
   """
   Generator returning all possible combinations of variants of `Block` in
   `cblocks` that have the appropriate mass-dimension.

   Parameters
   cblocks : list[type]
      Collection of custom implementations of `Block` (and masses) that to
      be used.
   mdTarget : int
      Desired mass-dimension.

   Returns
   -------
   Iterator[list[type]]
      Iterator over all the allowed combinations.
   """
   if mdTarget != 0 and len(cblocks)>0:
      for i in range(mdTarget // cblocks[0].__massDim__ + 1):
         md = mdTarget - i*cblocks[0].__massDim__
         temp = [cblocks[0]]*i
         for cbt in _getCustomBlockTemplates(cblocks[1:],
            mdTarget - i*cblocks[0].__massDim__):
            yield temp + cbt
   elif mdTarget == 0:
      yield []


def getBilinearTemplates(flavours:tuple[str,str], cblocks:list[type],
   mdTarget:int, sorting:Callable[[str,dict[type,int]],int]=None)\
   ->tuple[list[str],list[str]]:
   """
   Produce all templates of bilinears that have mass-dimension
   [Psi.Psi]+`mdTarget`, i.e., `mdTarget` is the relative mass-dimension of
   the "dressing" of a pure bilinear. No total derivatives are taken into
   account as those can be easily obtained by adding an appropriate overall
   power of derivatives.

   Parameters
   ----------
   flavours : tuple[str,str]
      Flavour content of the bilinear.
   cblocks : list[type]
      Implementations of `Block` to be used that are not derivatives,
      insertions of EOMs or field-strength tensors nor abstract variants like
      `_AlgebraBlock`.
   mdTarget : int
      Relative mass-dimension of the "dressing" of the pure bilinear.
   sorting : Callable[[tuple[str,dict[type,int]]],int], optional
      Prescription to enforce the desired ordering on the templates being
      generated. Higher numbers come first. Defaults to `None`.

   Returns
   -------
   tuple[list[str],list[str]]
      Collections of all the templates with the desired mass-dimension that
      contain EOMs or not respectively and which are not total derivaitves.

   Raises
   ------
   AssertionError
      If `mdTarget` is negative or `cblocks` contains unexpected types.
   """
   assert mdTarget>=0, "mdTarget has to be a non-negative integer."
   assert all(map(lambda x: not x in [d,D0,D0l,Dl,D,F,DF], cblocks)),\
   "Covariant derivatives, EOMs or field-strength tensors are not permitted."

   templates = list()
   _sort = list()
   middle = ""
   for m0 in filter(lambda x: x.__massDim__==0, cblocks):
      middle += m0.__name__ + "."
   cblocks = list(filter(lambda x: not x.__massDim__==0, cblocks))
   if sorting is None:
      sorting = lambda x: (
         + (mdTarget+1)**2 * (x[1][DF]+x[1][D0l]+x[1][D0])
         + (mdTarget+1)    * sum(x[1][y] for y in cblocks if issubclass(y, M))
         +                   x[1][F]
         )
   mdAlgMin = min((D.__massDim__,F.__massDim__,DF.__massDim__))
   # completely dumb approach
   for nD0l in range(mdTarget//D0l.__massDim__+1):
      s = dict()
      mD0l = mdTarget - nD0l*D0l.__massDim__
      s[D0l] = nD0l
      for nDl in range(mD0l//Dl.__massDim__+1):
         mDl = mD0l - nDl*Dl.__massDim__
         left = flavours[0] + "." + "D0l."*nD0l + "Dl."*nDl
         for nD0 in range(mDl//D0.__massDim__+1):
            mD0 = mDl - nD0*D0.__massDim__
            s[D0] = nD0
            for i in range(mD0//mdAlgMin+1):
               for alg in map(list, product([DF,F,D], repeat=i)):
                  mAlg = mD0 - sum(map(lambda x: x.__massDim__, alg))
                  # There should be a more efficient way!
                  if mAlg < 0: continue
                  s[F]  = alg.count(F)
                  s[DF] = alg.count(DF)
                  if mAlg==0:
                     templates.append(left + middle +\
                        "".join(map(lambda x: x.__name__ + ".", alg)) +\
                        "D0."*nD0 + flavours[1])
                     for block in cblocks:
                        s[block] = 0
                     _sort.append(_copy(s))
                  elif len(cblocks)>0:
                     for cbt in _getCustomBlockTemplates(cblocks, mAlg):
                        templates.append(left +\
                           "".join(map(lambda x: x.__name__ + ".", cbt)) +\
                           middle +\
                           "".join(map(lambda x: x.__name__ + ".", alg)) +\
                           "D0."*nD0 + flavours[1])
                        for block in cblocks:
                           s[block] = cbt.count(block)
                        _sort.append(_copy(s))
   templates = sorted(zip(templates,_sort), key=sorting, reverse=True)
   return (
      [x[0] for x in templates if x[1][D0]+x[1][D0l]+x[1][DF]>0],
      [x[0] for x in templates if x[1][D0]+x[1][D0l]+x[1][DF]==0])

