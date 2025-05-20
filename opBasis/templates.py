from fractions import Fraction

from .blocks import d,DF,F,D,Dl,D0,D0l,Trace,Bilinear,Colour

class TemplateRep(dict):
   """
   Represents building blocks ued to create the overall templates potentially
   consisting of bilinears, algebra traces, and other traces. Allows to track
   details like the presence of EOM insertions, custom blocks, etc.

   Parameters
   ----------
   rep : str
      The actual string-representation of the template.
   md : int|Fraction
      Mass-dimension of the template.
   tder : bool
      Indicates whether the overall template is a total derivative.
   kwargs : dict[str,int]
      Other details of the template like the number of different EOM
      insertions.
   """
   def __init__(self, rep:str, md:int|Fraction, tder:bool,
      **kwargs:dict[str,int]):
      self.rep = rep
      self.md = md
      self.tder = tder
      dict.__init__(self, **kwargs)

   def __str__(self):
      return self.rep

   def __getitem__(self, key):
      if not key in self.keys():
         return 0
      return dict.__getitem__(self, key)

   def __add__(self, add):
      if self.rep == "": return _copy(add)
      if add.rep == "": return _copy(self)
      newSort = {"Bilinear" : 0, "AlgTrace" : 0}
      for k in set(self.keys())|set(add.keys()):
         newSort[k] = self[k] + add[k]
      return TemplateRep(self.rep + "*" + add.rep, self.md+add.md,
         self.tder+add.tder>0 and \
         newSort["Bilinear"] + newSort["AlgTrace"]==1, **newSort)


def _prepareAlgebraBlockTemplates(nb:int, mdTarget:int|Fraction):
   """
   Distributes covariant derivatives among all the `nb` blocks remaining.

   Parameters
   ----------
   nb : int
      Number of blocks (remaining).
   mdTarget : int|Fraction
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
         for x in _prepareAlgebraBlockTemplates(
            nb-1, mdTarget - i*D.__massDim__):
            yield [i] + x
   elif mdTarget == 0:
      yield []
   

def getAlgebraTraceTemplates(mdTarget:int|Fraction)->list[TemplateRep]:
   """
   Returns all templates for algebra traces of mass-dimension `mdTarget`
   (except for the identity). Those can be obtained from their lower
   dimensional variants simply by adding an appropriate number of derivatives.
   Cyclicity of the `Trace` is already taken into account to reduce the overall
   number of templates.

   Parameters
   ----------
   mdTarget : int|Fraction
      Non-negative integer specifying the desired mass dimension.

   Returns
   -------
   list[str]
      All the templates for alegbra traces relevant at this mass-dimension.
   """
   # Only added for consistency - this should never have an effect.
   mdTarget -= Trace.__massDim__
   templates = Union()
   mdAlgMin = F.__massDim__ if F.__massDim__ < DF.__massDim__ else \
      DF.__massDim__
   if mdTarget < 2*mdAlgMin: return list()
   for nd in range((mdTarget - 2*mdAlgMin)//d.__massDim__ + 1):
      md = mdTarget - nd*d.__massDim__
      for nb in range(2, md // mdAlgMin + 1):
         for blocks in map(list, product([F,DF], repeat=nb)):
            mdb = md - sum(map(lambda x: x.__massDim__, blocks))
            if mdb<0: continue
            for abt in _prepareAlgebraBlockTemplates(nb, mdb):
               temp = list(map(lambda x: ("D."*x[0], x[1]),
                  zip(abt,blocks)))
               # drop duplicates by cyclicity
               template = [tuple(temp)]
               for i in range(1,nb):
                  template.append(tuple(temp[i:]+temp[:i]))
               label = "d."*nd + "tr(" + ".".join(x[0]+x[1].__name__ \
                  for x in list(sorted(template, key=str))[0]) + ")"
               templates.add(TemplateRep(
                  label,
                  mdTarget + Trace.__massDim__,
                  nd>0,
                  **{F.__name__ : sum(1 for x in template[0] if x[1] is F),
                     DF.__name__ : sum(1 for x in template[0] if x[1] is DF),
                     d.__name__ : nd,
                     "AlgTrace" : 1}
                  ))
   return list(templates.content)

def _getCustomBlockTemplates(cblocks:list[type], mdTarget:int|Fraction):
   """
   Generator returning all possible combinations of variants of `Block` in
   `cblocks` that have the appropriate mass-dimension.

   Parameters
   ----------
   cblocks : list[type]
      Collection of custom implementations of `Block` (and masses) that to
      be used.
   mdTarget : int|Fraction
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
   mdTarget:int|Fraction)->list[TemplateRep]:
   """
   Produce all templates of bilinears that have mass-dimension `mdTarget`.

   Parameters
   ----------
   flavours : tuple[str,str]
      Flavour content of the bilinear.
   cblocks : list[type]
      Implementations of `Block` to be used that are not derivatives,
      insertions of EOMs or field-strength tensors nor abstract variants like
      `_AlgebraBlock`.
   mdTarget : int|Fraction
      Mass-dimension of the bilinear.

   Returns
   -------
   list[TemplateRep]
      Collection of all the templates with the desired mass-dimension.

   Raises
   ------
   AssertionError
      If `cblocks` contains unexpected types.
   """
   mdTarget -= Bilinear.__massDim__
   if mdTarget<0: return list()
   assert all(map(lambda x: not x in [d,D0,D0l,Dl,D,F,DF], cblocks)),\
   "Covariant derivatives, EOMs or field-strength tensors are not permitted."

   templates = list()
   middle = ""
   s0 = dict()
   s0["Bilinear"] = 1
   for m0 in filter(lambda x: x.__massDim__==0, cblocks):
      middle += m0.__name__ + "."
      s0[m0.__name__] = 1
   cblocks = list(filter(lambda x: not x.__massDim__==0, cblocks))
   mdAlgMin = min((D.__massDim__,F.__massDim__,DF.__massDim__))
   # completely dumb approach
   for nd in range(mdTarget//d.__massDim__ + 1):
      md = mdTarget - nd*d.__massDim__
      s = dict(s0)
      s[d.__name__] = nd
      for nD0l in range(md//D0l.__massDim__ + 1):
         mD0l = md - nD0l*D0l.__massDim__
         s[D0l.__name__] = nD0l
         for nDl in range(mD0l//Dl.__massDim__ + 1):
            mDl = mD0l - nDl*Dl.__massDim__
            left = "d."*nd + flavours[0] + "." + "D0l."*nD0l + "Dl."*nDl
            for nD0 in range(mDl//D0.__massDim__ + 1):
               mD0 = mDl - nD0*D0.__massDim__
               s[D0.__name__] = nD0
               for i in range(mD0//mdAlgMin + 1):
                  for alg in map(list, product([DF,F,D], repeat=i)):
                     mAlg = mD0 - sum(map(lambda x: x.__massDim__, alg))
                     # There should be a more efficient way!
                     if mAlg < 0: continue
                     s[F.__name__]  = alg.count(F)
                     s[DF.__name__] = alg.count(DF)
                     if mAlg==0:
                        for block in cblocks:
                           s[block.__name__] = 0
                        templates.append(TemplateRep(left + middle +\
                           "".join(map(lambda x: x.__name__ + ".", alg)) +\
                           "D0."*nD0 + flavours[1], mdTarget+Bilinear.__massDim__,
                           nd>0, **s))
                     elif len(cblocks)>0:
                        for cbt in _getCustomBlockTemplates(cblocks, mAlg):
                           for block in cblocks:
                              s[block.__name__] = cbt.count(block)
                           templates.append(TemplateRep(left +\
                              "".join(map(lambda x: x.__name__ + ".", cbt)) +\
                              middle +\
                              "".join(map(lambda x: x.__name__ + ".", alg)) +\
                              "D0."*nD0 + flavours[1],
                              mdTarget+Bilinear.__massDim__,
                              nd>0, **s))
   return templates

def _prepareCustomMassTemplates(cblocks:list[type], mdTarget:int|Fraction):
   """
   Builds a trace by combining different (custom) implementations of `M` that
   amount to the desired mass-dimension.

   Parameters
   ----------
   cblocks : list[type]
      All the (custom) implementations of `M` that are to be used.
   mdTarget : int|Fraction
      Desired mass-dimension.

   Returns
   -------
   Iterator[list[int]]
      Number each specific (custom) `M` is supposed to be used.
   """
   if len(cblocks) > 0:
      for i in range(mdTarget//cblocks[0].__massDim__ + 1):
         for cbt in _prepareCustomMassTemplates(
            cblocks[1:], mdTarget - i*cblocks[0].__massDim__):
            yield [i] + cbt
   if mdTarget==0:
      yield []


def getCustomMassTemplates(cblocks:list[type], mdTarget:int|Fraction):
   """
   Given a collection of "mass-like" implementations of `Block`, i.e.,
   subclasses of `M` with `@massDim(n)` at `n>0`, all templates for single
   traces with appropriate mass-dimension `mdTarget` are being derived.

   Parameters
   ----------
   cblocks : list[type]
      Mass-like implementations of `Block`.
   mdTarget : int|Fraction
      The desired mass-dimension.

   Returns
   -------
   list[TemplateRep]
      All templates matching the requested mass-dimension.
   """
   if mdTarget<1: return list()
   templates = list()
   for cbt in _prepareCustomMassTemplates(cblocks, mdTarget):
      s = dict()
      rep = "tr("
      for k,n in zip(cblocks, cbt):
         s[k.__name__] = n
         rep += (k.__name__ + ".")*n
      templates.append(TemplateRep(rep[:-1] + ")", mdTarget, False, **s))
   return templates


def _buildCombinations(active:list[TemplateRep], excl:list[TemplateRep], 
   mdTarget:int|Fraction, minExcl:int):
   """
   Finds all combinations of templates for traces, bilinears, and derivatives
   thereof, that have the desired mass-dimension `mdTarget`.

   Parameters
   ----------
   active : list[TemplateRep]
      Collection of templates to be used exclusively until `minExcl==0`.
   excl : list[TemplateRep]
      Collection of templates to be included once `minExcl==0`.
   mdTarget : int|Fraction
      The desired mass-dimension of the full template.
   minExcl : int
      The minimal number of bilinears or algebra traces. Needed to avoid
      redundancy between total derivatives and derivatives acting only on
      sub-parts of the template.

   Returns
   -------
   Iterator[TemplateRep]
      All possible templates with mass-dimension `mdTarget`.
   """
   if mdTarget>0:
      if minExcl<=0 and len(excl)>0:
         active = active+excl
         excl = list()
      for ia,a in enumerate(active):
         for comb in _buildCombinations(active[ia:], excl, mdTarget - a.md,
            minExcl-1):
            yield comb+a
   elif mdTarget==0 and minExcl<1:
      yield TemplateRep("", 0, False)


def getTemplates(mdTarget:int|Fraction, flavours:list[tuple[str,str]]=None,
   cblocks:list[type]=None, algTrace:bool=True,
   customFilter:Callable[[TemplateRep],bool]=None):
   """
   Produces all templates with a given mass-dimension `mdTarget` including
   product of traces, bilinears etc.

   Parameters
   ----------
   mdTarget : int|Fraction
      Desired mass-dimension.
   flavours : list[tuple[str,str]], optional
      Collection of flavours to be used in templates for a `Bilinear`. Defaults
      to `None` if omitted, implying to not create such templates.
   cblocks : list[type], optional
      Collection of (custom) blocks to be used as masses, Dirac gamma matrices
      etc. Defaults to `None`.
   algTrace : bool, optional
      Indicates whether templates involving algebra traces are allowed.
      Defaults to `True`.
   customFilter : Callable[[TemplateRep],bool]
      Additional filter to drop any undesired templates, e.g., keep only
      even powers of the twisted-mass parameter in templates for `Trace`.

   Returns
   -------
   list[TemplateRep]
      Collection of all the allowed templates.

   Raises
   ------
   AssertionError
      If `mdTarget` is negative.
   ValueError
      If `mdTarget` is too large and would allow for 6-quark operators.
   """
   assert mdTarget>=0, "mdTarget has to be a non-negative rational number."
   if not flavours is None and mdTarget >= 3*Bilinear.__massDim__:
      raise ValueError("6-quark operators and beyond are not accessible due "+
         "to implicit use of Fierz identities.")
   cblocks = [] if cblocks is None else cblocks
   if customFilter is None: customFilter = lambda x: True
   temp = list()
   custom = list()
   for i in range(mdTarget+1):
      # prepare custom Block traces and store them for higher mass-dimensions!
      if not flavours is None:
         for fl in flavours:
            temp.extend([y for y in getBilinearTemplates(fl, cblocks, i) \
               if customFilter(y))
            temp.extend([y for y in getBilinearTemplates(fl,
               cblocks+[Colour], i) if customFilter(y)])
      if algTrace:
         temp.extend([y for y in getAlgebraTraceTemplates(i) if customFilter(y)])
      custom.extend(y for y in getCustomMassTemplates(
         [x for x in cblocks if issubclass(x, M)], i) if customFilter(y))
   templates = list()
   for nd in range(mdTarget//d.__massDim__ + 1):
      md = mdTarget - nd*d.__massDim__
      for c in _buildCombinations(temp, custom, md, 2 if nd>0 else 1):
         if c.rep=="" or not c[Colour.__name__] in (0,2): continue
         # We only care about overall derivatives..!
         if nd>0:
            c[d.__name__] += nd
            c.rep = "d."*nd + "{" + c.rep + "}"
            c.md += nd*d.__massDim__
            c.tder = True
         templates.append(c)
   return templates

