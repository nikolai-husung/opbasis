import opbasis as opb
# Specify discrete transformations as listed in model file.
def H4(op:opb.LinearComb, plane:int):
   return op.rotation(plane)
def C(op:opb.LinearComb, _blank:int):
   return op.charge()
def P(op:opb.LinearComb, mu:int):
   return op.reflection(mu)
def chiral(op:opb.LinearComb):
   return op.chiralSpurion()
