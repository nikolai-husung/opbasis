"""
This submodule allows the user to input the desired `Model` specifying the
desired transformations and transformation-behaviour of the future operator
basis. Part of a `Model` are also declarations of any custom implementations
of `Block`. Finally, classes of operators can be defined as string-valued so
called *templates*, which are then parsed into ansätze for the desired operator
basis. While proposing ansätze via templates can be performed manually, via
`getTemplates` one may get a complete set of templates without the danger of
overlooking any. After imposing the specified transformation-behaviour through
a call to `overcompleteBasis` this yields a yet overcomplete basis.
"""

from dataclasses import dataclass
from fractions import Fraction
from importlib import import_module as _import
from copy import deepcopy as _copy
from operator import and_

import re

from itertools import product, combinations_with_replacement as cwr

from typing import Iterator

from .blocks import Block, d, D, Dl, D0, D0l,\
   _AlgebraBlock, F, DF, Colour,\
   M, dM
from .dirac import Gamma
from .pauli import SU2
from .ops import LinearComb, _Commutative, Bilinear, Trace
from .basics import Complex, rotPlanes, Union

# end-points for _AlgebraBlock templates
_ALG = "|".join(["F","DF"])

_PROTECTED = ["Block", "d", "D", "Dl", "F", "DF", "D0", "D0l",
   "Colour", "M", "dM", "Gamma", "SU2", "Multiplicative",
   "LinearComb", "Bilinear", "Trace"]

MAX_ITER = 50

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
      protected = _copy(_PROTECTED)
      for line in lines:
         if not ":" in line:
            assert line.replace(" ", "") == ""
            continue
         cat,opt = line.split(":", maxsplit=1)
         cat = cat.replace(" ", "")
         if cat == "Discrete":
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
      return Model(pyName, blocks, transfs, spurions, fs, None)

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
      raise ValueError("Something is wrong with the template `%s`."%template)


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
         yield LinearComb([_Commutative(_copy(list(c)), _copy(list(dd)), 1)], 1)


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

   Raises
   ------
   ValueError
      If enforcing the desired transformation properties fails within MAX_ITER
      iterations.
   """
   anyNew = True
   coll = Union()
   # Ensure that ansatz is simplified such that it serves as a valid reference.
   ansatz.simplify()
   coll.add(ansatz)
   while anyNew:
      anyNew = False
      for c in coll:
         negc = -c
         for transf,behaviour in model.transf.items():
            for i,b in enumerate(behaviour):
               if b=="x": continue
               temp = _copy(c)
               niter = 0
               while True:
                  niter += 1
                  if niter > MAX_ITER:
                     raise ValueError("Enforcing the desired transformations "
                        "onto the current ansatz\n   %s\nfails within %i "
                        "iterations."%(str(ansatz),MAX_ITER))
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
   bblks = set()
   for ansatz in parseAnsatz(template, model):
      ansatz.simplify()
      if str(ansatz.terms[0]) in bblks: continue
      ansatz = symmetrise(ansatz, model)
      if ansatz.factor == 0: continue
      if not all(getattr(model.pyModule, sp)(ansatz) for sp in model.spurion):
         continue
      if ansatz != 0:
         basis.add(ansatz)
         bblks |= set(str(term) for term in ansatz.terms)
   return basis.content

