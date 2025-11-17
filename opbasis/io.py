"""
This submodule takes care of storing operator bases and reading them file.
Reading relies heavily on use of `regular expressions <re>`. As long as custom
`Block` implementations adhere to the common index-conventions, all of this
should work out of the box.
"""

import os
import re
import warnings
from fractions import Fraction
from zipfile import Path, ZipFile, ZIP_DEFLATED

from .basics import Complex, CustomIndex, dim
from .blocks import Block, d, D, Dl, \
   _AlgebraBlock, F, DF, \
   D0, D0l, Colour, M, dM
from .dirac import Gamma
from .pauli import SU2
from .ops import LinearComb, _Commutative, Bilinear, Trace
from .opBasis import Model
from .templates import TemplateRep

def _BlockWildcards(blocks):
   return "|".join(x._toRegex() for x in blocks)

# defaults to build regular expressions
_frac     = "([0-9]*|[0-9]*/[0-9]*)"
_coeff    = "[+-](\\([+-]?%s[+-]%sj\\)|%s|%sj)"%(_frac,_frac,_frac,_frac)
_d        = d._toRegex() + "\\."
_der      = "((%s)*%s)"%(_d,_d[:-2])
_D0l      = D0l._toRegex() + "\\."
_Dl       = Dl._toRegex() + "\\."
_covl     = "(" + _D0l + ")*(" + _Dl + ")*"
_D0       = D0._toRegex() + "\\."
_D        = D._toRegex() + "\\."
_covr     = "(" + _D + ")*(" + _D0 + ")*"
_algBlock = "(" + _D + ")*(" + _BlockWildcards([F,DF,Colour]) + ")"
_algTrace = "(%s\\.)*"%_d + "tr\\((" + _algBlock + "\\.){1,}" + \
            _algBlock + "\\)"



def _parseAlgBlock(expr:str)->_AlgebraBlock:
   Dfound = re.search("\\A" + _D, expr)
   blocks = list()
   while Dfound:
      blocks.append(D(D._types()[0](int(Dfound.groups()[1]))))
      expr = expr[Dfound.end():]
      Dfound = re.search("\\A" + _D, expr)
   ab = re.search("\\A(" + _BlockWildcards([F,DF,Colour]) + ")", expr)
   abg = [x for x in ab.groups()[1:] if not x is None]
   ab = globals()[abg[0]]
   blocks.append(ab(*[f(int(x)) for f,x in zip(ab._types(),abg[1:])]))
   return _AlgebraBlock(blocks)


def _parseAlgTrace(expr:str)->Trace:
   der = list()
   dfound = re.search("\\A" + _d, expr)
   while dfound:
      der.append(d(d._types()[0](int(dfound.groups()[1]))))
      expr = expr[dfound.end():]
      dfound = re.search("\\A" + _d, expr)
   expr = expr[3:]
   cyclic = list()
   abFound = re.search("\\A" + _algBlock, expr)
   while abFound:
      cyclic.append(_parseAlgBlock(abFound.group()))
      expr = expr[abFound.end()+1:]
      abFound = re.search("\\A" + _algBlock, expr)
   return Trace(cyclic, der)


def _parseOtherTrace(expr:str, oBlock:str, model:Model)->Trace:
   der = list()
   dfound = re.search("\\A" + _d, expr)
   while dfound:
      der.append(d(d._types()[0](int(dfound.groups()[1]))))
      expr = expr[dfound.end():]
      dfound = re.search("\\A" + _d, expr)
   expr = expr[3:]
   cyclic = list()
   obFound = re.search("\\A" + oBlock, expr)
   while obFound:
      cbg = [x for x in obFound.groups()[1:] if not x is None]
      cb = cbg[0]
      cb = getattr(model.pyModule, cb) if cb in model.blocks \
         else globals()[cb]
      cyclic.append(cb(*[getattr(f, x) if issubclass(f, CustomIndex) else\
                         f(int(x)) for f,x in zip(cb._types(),cbg[1:])]))
      expr = expr[obFound.end()+1:]
      obFound = re.search("\\A" + oBlock, expr)
   return Trace(cyclic, der)


def _parseBilinear(expr:str, flavourMask:str, oBlock:str, model:Model)\
   ->Bilinear:
   """
   Parses the string-representation of an instance of `Bilinear` using regular
   expressions, which are specified in *flavourMask* and *oBlock*.

   Parameters
   ----------
   expr : str
      String-representation of the `Bilinear`.
   flavourMask : str
      Regular expression to identify flavours.
   oBlock : str
      Regular expression to identify non-algebra-valued instances of `Block`.
   model : Model
      Descriptor of the considered theory.
   
   Returns
   -------
   Bilinear
      Instance of `Bilinear` as parsed from string.
   """
   der = list()
   dfound = re.search("\\A" + _d, expr)
   while dfound:
      der.append(d(d._types()[0](int(dfound.groups()[1]))))
      expr = expr[dfound.end():]
      dfound = re.search("\\A" + _d, expr)
   fl = re.search("\\A" + flavourMask + "\\.", expr)
   expr = expr[fl.end():]
   fl = fl.groups()[0]
   covl = list()
   D0lfound = re.search("\\A" + _D0l, expr)
   while D0lfound:
      covl.append(D0l())
      expr = expr[D0lfound.end():]
      D0lfound = re.search("\\A" + _D0l, expr)
   Dlfound = re.search("\\A" + _Dl, expr)
   while Dlfound:         
      covl.append(Dl(Dl._types()[0](int(Dlfound.groups()[1]))))
      expr = expr[Dlfound.end():]
      Dlfound = re.search("\\A" + _Dl, expr)
   blocks = list()
   anyFound = True
   while anyFound:
      anyFound=False
      obFound = re.search("\\A" + oBlock + "\\.", expr)
      while obFound:
         cbg = [x for x in obFound.groups()[1:] if not x is None]
         cb = cbg[0]
         cb = getattr(model.pyModule, cb) if cb in model.blocks \
            else globals()[cb]
         blocks.append(cb(*[getattr(f, x) if issubclass(f, CustomIndex) else \
                            f(int(x)) for f,x in zip(cb._types(),cbg[1:])]))
         expr = expr[obFound.end():]
         obFound = re.search("\\A" + oBlock + "\\.", expr)
         anyFound=True
      abFound = re.search("\\A" + _algBlock + "\\.", expr)
      while abFound:
         blocks.append(_parseAlgBlock(abFound.group()))
         expr = expr[abFound.end():]
         abFound = re.search("\\A" + _algBlock, expr)
         anyFound=True
   covr = list()
   Dfound = re.search("\\A" + _D, expr)
   while Dfound:
      covr.append(D(D._types()[0](int(Dfound.groups()[1]))))
      expr = expr[Dfound.end():]
      Dfound = re.search("\\A" + _D, expr)
   D0found = re.search("\\A" + _D0, expr)
   while D0found:
      covr.append(D0())
      expr = expr[D0found.end():]
      D0found = re.search("\\A" + _D0, expr)
   return Bilinear(blocks, covl, covr, (fl,expr), der)
  


def _parseLinearComb(expr:str, term:str, bilin:str, oTrace:str,
   flavourMask:str, oBlock:str, model:Model)->LinearComb:
   """
   Turns the string representation of `LinearComb` back into an instance of
   this type.

   Parameters
   ----------
   expr : str
      String expression of the full `LinearComb`.
   term : str
      Regular expression to identify one term.
   bilin : str
      Regular expression to identify a `Bilinear`.
   oTrace : str
      Regular expression to identify a non-algebra `Trace`.
   flavourMask : str
      Regular expression to identify flavours.
   oBlock : str
      Regular expression to identify of (also custom) instances `Block`.
   model : Model
      Descriptor of the considered theory.
   
   Returns
   -------
   LinearComb
      Parsed instance of `LinearComb`.
   """
   # figure out prefactor
   factor,expr = expr.split("*{")
   fac = re.search(_coeff+"\\Z", factor).groups()
   fre = fac[1] if not fac[1] is None else ("0" if fac[3] is None else fac[3])
   fim = fac[2] if not fac[2] is None else ("0" if fac[4] is None else fac[4])
   factor = (-1 if factor[0]=="-" else 1) * Complex(
      Fraction(*[int(x) for x in fre.split("/")]),
      Fraction(*[int(x) for x in fim.split("/")]))
   lcomb = LinearComb([], factor)
   tfound = re.search(term, expr)
   while tfound:
      factor,comm = tfound.group().split("*", 1)
      factor = factor.replace("\n","").replace("\t", "").replace(" ","")
      fac = re.search(_coeff+"\\Z", factor).groups()
      fre = fac[1] if not fac[1] is None else \
         ("0" if fac[3] is None else fac[3])
      fim = fac[2] if not fac[2] is None else \
         ("0" if fac[4] is None else fac[4])
      factor = (-1 if factor[0]=="-" else 1) * Complex(
         Fraction(*[int(x) for x in fre.split("/")]),
         Fraction(*[int(x) for x in fim.split("/")]))
      der,comm = comm.split("<")
      c = _Commutative([],
         [d(d._types()[0](int(d_.split("[")[1][:-1]))) \
            for d_ in der.split(".") if d_!=""],
         factor)
      oTrFound   = re.search(oTrace, comm)
      algTrFound = re.search(_algTrace, comm)
      bilinFound = re.search(bilin, comm)
      while oTrFound or algTrFound or bilinFound:
         if oTrFound and \
            (not algTrFound or oTrFound.start()<algTrFound.start())\
            and \
            (not bilinFound or oTrFound.start()<bilinFound.start()):
            c.prod.append(_parseOtherTrace(oTrFound.group(), oBlock, model))
            comm = comm[oTrFound.end():]
         elif algTrFound and \
            (not bilinFound or algTrFound.start()<bilinFound.start()):
            c.prod.append(_parseAlgTrace(algTrFound.group()))
            comm = comm[algTrFound.end():]
         else:
            c.prod.append(_parseBilinear(bilinFound.group(), flavourMask,
                                         oBlock, model))
            comm = comm[bilinFound.end():]
         oTrFound   = re.search(oTrace, comm)
         algTrFound = re.search(_algTrace, comm)
         bilinFound = re.search(bilin, comm)
      lcomb.terms.append(c)
      expr = expr[tfound.end():]
      tfound = re.search(term, expr)
   return lcomb


def parseLinearCombs(expr:str, model:Model)->list[LinearComb]:
   """
   Identifies all `LinearComb` separated by linebreaks and hands them to the
   function _parseLinearComb to be parsed.

   Parameters
   ----------
   expr : str
      String containing instances of `LinearComb` separated by linebreaks.
   model : Model
      Descriptor of the Model the various `LinearComb` correspond to.
   
   Returns
   -------
   list[LinearComb]
      Collection of all instances of `LinearComb` parsed from *expr*.
   
   Raises
   ------
   ValueError
      If the formatting encountered does not fit to the regular expression,
      e.g., due to use of the wrong Model or typos.
   """
   flavourMask = "(" + "|".join("|".join([key]+flavours) \
      for key,flavours in model.flavourSets.items()) + ")"
   oBlock = "(" + _BlockWildcards(
      [getattr(model.pyModule, b) for b in model.blocks] + [M,dM,SU2,Gamma])\
      + ")"
   oTrace = "tr\\((" + oBlock + "\\.)*" + oBlock + "\\)"
   bilin  = "(%s)*"%_d + flavourMask + "\\." + _covl + \
            "(" + oBlock + "\\.)*(" + _algBlock + "\\.)*" + _covr + flavourMask
   prod   = (bilin,_algTrace,oTrace)
   term   = "\\s*" + _coeff + "\\*" + _der + "?<((" +\
            "|".join(prod) + ") )*(" + "|".join(prod) + ")>"
   lcomb  = _coeff + "\\*\\{\\s*(" + term + "\\s*)*\\}"
   # Check for overall syntax.
   if not re.search(
      "\\A(" + lcomb + ")" + "((\n|\r|\r\n)+" + lcomb + ")*\\s*\\Z", expr):
      # Allow for empty string, i.e., no valid non-vanishing operator had
      # been found.
      if re.search("\\A( |\n|\r|\r\n)*\\Z", expr):
         return list()
      raise ValueError(
         "Wrong formatting for linear combination(s).")
   lfound = re.search(lcomb, expr)
   res = list()
   while lfound:
      res.append(_parseLinearComb(lfound.group(), term, bilin, oTrace,
         flavourMask, oBlock, model))
      expr = expr[lfound.end():]
      lfound = re.search(lcomb, expr)
   return res


class CompressedBasis:
   """
   Allows easy access to storing the overcomplete set of operators for various
   templates in separate files which are combined into a single zip-archive.
   If the file already exists, allows to extract stored `LinearComb`.

   .. important::
      Requires the Python module of the provided Model to be imported to ensure
      the availability of any custom implementations of `Block` defined within.

   Parameters
   ----------
   fname : str
      Name of the archive.
   model : Model
      Description of the Model the overcomplete bases found belong to.
   checkModule : bool, optional
      Compare model as stored in the archive with the currently used model.
      Defaults to True.

   Raises
   ------
   AssertionError
      If the Python module of the provided *model* is not already imported.
   ValueError
      If the provided *model* does not agree with the one stored in the
      zip-archive.
   """
   def __init__(self, fname:str, model:Model, checkModule:bool=True):
      assert not model.pyModule is None, "Requires the model provided to "+\
         "have its associated Python module imported."
      if os.path.exists(fname):
         with ZipFile(fname, 'r', ZIP_DEFLATED, compresslevel=9) as zf:
            if checkModule:
               check = Path(zf).joinpath(".MODEL").read_text()
               checkContent = Path(zf).joinpath(".PY").read_text()
               ref,refContent = model.toReference()
               if ref != check or refContent != checkContent:
                  raise ValueError("Models do not agree.")
            self.templates = [x.name[:-4] for x in Path(zf).iterdir() if\
               x.is_file() and x.name.endswith(".ops")]
      else:
         ref,refContent = model.toReference()
         with ZipFile(fname, 'a', ZIP_DEFLATED, compresslevel=9) as zf:
            zf.writestr(".MODEL", ref)
            zf.writestr(".PY", refContent)
         self.templates = []
      self.fname = fname

   def __contains__(self, template:str)->bool:
      """
      Allows simple use of the *in* statement to check if *template* has
      already been added previously either through a call to *add* or manually.

      Parameters
      ----------
      template : str
         String representation of the template for ansätze to be used.

      Returns
      -------
      bool
         Boolean indicating whether the *template* is contained within the
         archive.
      """
      return template in self.templates

   def add(self, template:TemplateRep, ops:list[LinearComb]):
      """
      Appends a file with the name *template*.ops containing the string-
      representation of the operators *ops*. Raises a ValueError in case
      *template*.ops already exists.

      Parameters
      ----------
      template : TemplateRep
         Carries details about the building blocks used in the template when
         deriving the associated operator basis as well as the actual
         mass-dimension.

      ops : list[LinearComb]
         Collection of operators found for this specific template that are
         compatible with the model-specific transformation properties.

      Raises
      ------
      ValueError
         If the *template* to be added is already present.
      """
      if template.rep in self.templates:
         raise ValueError(
            "The specified template *%s* already exists."%template.rep)
      with ZipFile(self.fname, 'a', ZIP_DEFLATED, compresslevel=9) as zf:
         zf.writestr(template.rep+".ops", "\n".join(str(op) for op in ops))
         zf.writestr(template.rep+".rep", "\n".join(
            [str(template.md),str(template.tder)] +\
            [key+" : "+str(item) for key,item in sorted(template.items())]))
      self.templates.append(template.rep)

   def get(self, template:str, model:Model)->list[LinearComb]:
      """
      Reads all operators contained in *template*.ops and parses them into
      `LinearComb`. The resulting list of operators is returned. 

      Parameters
      ----------
      template : str
         String representation of the template for ansätze to be used.

      Returns
      -------
      list[LinearComb]
         Collection of `LinearComb` stored for the given *template*.

      Raises
      ------
      ValueError
         If *template*.ops does not exist in the archive.
      """
      if not template in self.templates:
         raise ValueError("Specified template *%s* is not available."%template)
      with ZipFile(self.fname, 'r', ZIP_DEFLATED, compresslevel=9) as zf:
         ops = parseLinearCombs(
                  Path(zf).joinpath(template+".ops").read_text(), model)
      return ops

   def getTemplates(self):
      """
      Reads all templates currently available including their `TemplateRep` for
      customised ordering.

      Returns
      -------
      list[TemplateRep]
         Collection of `TemplateRep` representing all the operators stored
         and grouped according to their *template*.
      """
      res = list()
      with ZipFile(self.fname, 'r', ZIP_DEFLATED, compresslevel=9) as zf:
         for template in self.templates:
            parts = Path(zf).joinpath(template+".rep").read_text().split("\n")
            temp = TemplateRep(template, Fraction(parts[0]), bool(parts[1]))
            for part in parts[2:]:
               key,item = part.split(" : ")
               temp[key] = int(item)
            res.append(temp)
      return res
