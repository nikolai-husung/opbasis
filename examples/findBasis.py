# This script is a rudimentary example on how to extract the minimal operator
# basis for pure gauge theory or Wilson quarks.
import sys
from fractions import Fraction

import opBasis as opb


name = sys.argv[1]
md   = int(sys.argv[2])

model = opb.Model.read("models/%s.in"%name)
model.apply()
print(model)

fset = list(model.flavourSets.keys())[0]
# Default behaviour: Use already existing templates and associated basis.
#                    -> Adapt at will.
storage = opb.CompressedBasis(name+"_%i.basis"%md, model)
if len(storage.templates) == 0:
   templates = opb.getTemplates(md, [(fset,fset),],
      [getattr(model.pyModule,x) for x in model.blocks] + [opb.Gamma,opb.M])
else:
   templates = storage.getTemplates()
# Sort templates according to the following criteria:
# is(total derivative) > EOM-vanishing > massive > #(of Bilinear)
#                      > #F(in Bilinear) > others > #F(in Trace)
# -> Last three inequalities are our choice to get, e.g., the SW-term and
#    discard tr(F.F.F) in favour of other operators.
templates = [x for x in sorted(templates, reverse=True,
   key=lambda x: (
      + 7**5 * (1 if x.tder else 0)
      + 7**4 * (x["DF"]+x["D0"]+x["D0l"])
      + 7**3 * x["M"]
      + 7**2 * x["Bilinear"]
      + 7    * (1 if "D.F.D.F" in x.rep else 0)
      +        x["F"]*(1 if x["Bilinear"]>0 else -1)
      ))]

bases = list()
for template in templates:
   if str(template) in storage:
      basis = storage.get(str(template), model)
   else:
      basis = opb.overcompleteBasis(str(template), model)
      storage.add(template, basis)
      for b in basis:
         print(b)
   if len(basis)>0:
      bases.append(basis)

# Need to specify EOMs.
gEOM = opb.GluonEOM(Fraction(1,2),Fraction(3),[],[])
gluonPart = "\t-2*<tr(D[#nu#].F[#nu#,#mu#].Colour)>"
gluonPart = "\n".join(gluonPart.replace("#nu#", str(nu)) for nu in range(4))
fermionPart = ""
for fset in model.flavourSets.keys():
   fermionPart += "\t-1*<%s.Gamma[#gmu#].Colour.%s>\n"%(fset,fset)

for mu in range(4):
   gEOM.gluonPart.append(opb.parseLinearCombs("+1*{\n" + gluonPart.\
      replace("#mu#", str(mu)) + "\n}", model)[0])
   if len(model.flavourSets):
      gEOM.fermionPart.append(opb.parseLinearCombs("+1*{\n" + fermionPart.\
         replace("#gmu#", opb.axisGammas[mu].name) + "}", model)[0])
   else:
      gEOM.fermionPart.append(None)

fEOMs = list()
if len(model.flavourSets):
   for fset in model.flavourSets.keys():
      fEOM = "\t+1*<%s.Gamma[#gnu#].D[#nu#].%s>"%(fset,fset)
      fEOM = "\n".join(fEOM.replace("#gnu#", opb.axisGammas[nu].name).\
         replace("#nu#", str(nu)) for nu in range(4))
      fEOM += "\n\t+1*<%s.Gamma[id_].M.%s>"%(fset,fset)
      fEOMs.append(opb.parseLinearCombs("+1*{\n" + fEOM + "\n}", model)[0])

for basis in opb.findMinBases(bases, gEOM, fEOMs, model):
   print("############################################")
   for b in basis:
      print(b)
