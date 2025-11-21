"""
This submodule collects all the basic functionality for calculus contained
within this package. The central ingredient is the `Complex` type, which allows
to work with rational-number-valued complex numbers. While not perfect, this is
sufficient for many scenarios and much simpler to implement and maintain than
irrational numbers and avoids (most) reliance on rounding.
"""
from fractions import Fraction
import math
from copy import deepcopy as _copy

class Complex:
   """Rudimentary implementation of a rational-number-valued complex number.

   Parameters
   ----------   
   re : int|Fraction
      Real part of the complex number.
   im : int|Fraction, optional
      Imaginary part of the complex number, defaults to 0.
   """
   def __init__(self, re:int|Fraction, im:int|Fraction=0):
      self.re = Fraction(re)
      self.im = Fraction(im)

   def __bool__(self):
      return self.re!=0 or self.im!=0

   def __eq__(self, cmp):
      if isinstance(cmp, Complex):
         return self.re==cmp.real and self.im==cmp.imag
      if isinstance(cmp, (Fraction, int)):
         return self.re==cmp and self.im==0
      return NotImplemented
      
   def __repr__(self):
      return self.__class__.__name__ + "(%s,%s)"%(str(self.re),str(self.im))

   def __str__(self):
      if self.im==0:
         return ("" if self.re<0 else "+") + str(self.re)
      if self.re==0:
         return ("" if self.im<0 else "+") + str(self.im) +"j"
      if self.re<0:
         return "-(" + str(-self.re) + ("" if self.im>0 else "+") +\
            str(-self.im) + "j)"
      return "+(" + str(self.re) + ("" if self.im<0 else "+") +\
            str(self.im) + "j)"

   def __abs__(self):
      temp = self.re*self.re + self.im*self.im
      if isinstance(temp, Fraction):
         if temp.is_integer():
            temp = temp.numerator
         else:
            return Complex(Fraction(math.isqrt(temp.numerator),
                                    math.isqrt(temp.denominator)))
      return Complex(math.isqrt(temp))

   def __add__(self, add):
      if isinstance(add, (int,Fraction,Complex)):
         return self.__class__(self.re+add.real, self.im+add.imag)
      return NotImplemented

   def __iadd__(self, add):
      if isinstance(add, (int,Fraction,Complex)):
         self.re += add.real
         self.im += add.imag
         return self
      return NotImplemented

   def __radd__(self, add):
      return self.__add__(add)

   def __sub__(self, sub):
      if isinstance(sub, (int,Fraction,Complex)):
         return self.__class__(self.re-sub.real, self.im-sub.imag)
      return NotImplemented

   def __isub__(self, sub):
      if isinstance(sub, (int,Fraction,Complex)):
         self.re -= sub.real
         self.im -= sub.imag
         return self
      return NotImplemented

   def __rsub__(self, sub):
      if isinstance(sub, (int,Fraction,Complex)):
         return self.__class__(sub.real-self.re, sub.imag-self.im)
      return NotImplemented

   def __neg__(self):
      return self.__class__(-self.re, -self.im)

   def __mul__(self, mul):
      if isinstance(mul, (int,Fraction,Complex)):
         return self.__class__(self.re*mul.real-self.im*mul.imag,
                               self.re*mul.imag+self.im*mul.real)
      return NotImplemented

   def __imul__(self, mul):
      if isinstance(mul, (int,Fraction,Complex)):
         temp = self.re*mul.real-self.im*mul.imag
         self.im = self.re*mul.imag+self.im*mul.real
         self.re = temp
         return self
      return NotImplemented
   
   def __rmul__(self, mul):
      return self.__mul__(mul)

   def __truediv__(self, div):
      if isinstance(div, (Fraction, int)):
         return self.__class__(Fraction(self.re, div), Fraction(self.im, div))
      if isinstance(div, Complex):
         temp = div.abs2()
         return self.__class__(
            Fraction(self.re*div.real + self.im*div.imag, temp),
            Fraction(self.im*div.real - self.re*div.imag, temp))
      return NotImplemented

   def __rtruediv__(self, lhs):
      temp = self.abs2()
      return self.__class__(Fraction(lhs*self.re, temp),
                            Fraction(-lhs*self.im, temp))

   def __pow__(self, n:int):
      if not isinstance(n, int): return NotImplemented
      temp = 1
      for i in range(abs(n)):
         temp *= self
      return 1/temp if n<0 else temp

   def phase(self):
      """
      Assuming the complex number to be purely real or purely imaginary, this
      function returns the sign and (for a purely imaginary number) an
      imaginary unit.
      
      Returns
      -------
      Complex
         Sign and any overall imaginary unit.

      Raises
      ------
      AssertionError
         If number is not purely real or purely imaginary.
      """
      assert self.re==0 or self.im==0
      return self.__class__(
         0 if self.im!=0 else (-1 if self.re<0 else 1),
         -1 if self.im<0 else (1 if self.re==0 else 0))

   def abs2(self) -> int|Fraction:
      """Computes z*conjugate(z) of the complex number z.
      
      Returns
      -------
      int|Fraction
         Absolute value squared of the current complex number.
      """
      return self.re*self.re + self.im*self.im

   def conjugate(self):
      """Complex conjugation.
      
      Returns
      -------
      Complex
         Complex conjugate of the current complex number.
      """
      return Complex(self.re, -self.im)

   @property
   def real(self):
      """
      Returns the real part of the complex number to comply with standard 
      Python number interface.
      """
      return self.re

   @property
   def imag(self):
      """
      Returns the imaginary part of the complex number to comply with standard
      Python number interface.
      """
      return self.im


class Matrix:
   """Basic implementation of a matrix with complex rational coefficients.

   Parameters
   ----------
   components : list[list[Complex]], optional
      The rows and columns of the matrix. If `None` are given, yields an empty
      matrix which is a valid starting point to build your matrix via 
      `Matrix.extend`.
   """
   ROW = 0
   COL = 1
   def __init__(self, components:list[list[Complex]] = None):
      # Make sure that we always have complex numbers with fractional entries.
      self.components = list() if components is None \
         else [[Complex(x.real, x.imag) for x in row] for row in components]

   @property
   def M(self):
      """Returns the number of rows."""
      return len(self.components)

   @property
   def N(self):
      """Returns the number of columns."""
      if self.M==0: return 0
      return len(self.components[0])

   @staticmethod
   def ZERO(M:int,N:int):
      r"""Creates an :math:`M\times N` matrix filled with zeros.

      Parameters
      ----------
      M : int
         Number of rows.
      N : int
         Number of columns.
      
      Returns
      -------
      Matrix
         An all zero matrix with the requested dimensions.
      """
      return Matrix([[0]*M]*N)

   @staticmethod
   def diag(delems:list[Complex], M=None, N=None):
      """Generalisation of a standard diagonal square matrix.

      The user can choose the dimensions freely as long as the diagonal entries
      fit into the matrix.

      Parameters
      ----------
      diag : list[Complex]
         Diagonal elements to fill the matrix with.
      M : int, optional
         Requested number of rows. Defaults to `None`. Otherwise, must be at
         least the number of *delems* given.
      N : int, optional
         Requested number of columns. Defaults to `None`. Otherwise, must be
         at least the number of *delems* given.

      Returns
      -------
      Matrix
         New matrix with diagonal entries set to *delems* and otherwise zero.
      """
      l = len(delems)
      M = l if M is None else M
      N = l if N is None else N
      assert l<=M and l<=N, ("The number of diagonal entries (%i) does not "
         "fit into the requested %ix%i matrix.")%(l,M,N)
      comp = list()
      for r in range(M):
         row = [0]*N
         if r<=l:
            row[r] = delems[r]
         comp.append(row)
      return Matrix(comp)

   def __str__(self):
      s = "[\n"
      for row in self.components:
         s += "   " + str(row) + "\n"
      return s + "]"

   def __add__(self, mat):
      if isinstance(mat, Matrix):
         comp = list()
         for r in range(self.M):
            row = list()
            for c in range(self.N):
               row.append(self.components[r][c]+mat.components[r][c])
            comp.append(row)
         return Matrix(comp)
      return NotImplemented

   def __iadd__(self, mat):
      if isinstance(mat, Matrix):
         comp = list()
         for r in range(self.M):
            for c in range(self.N):
               self.components[r][c] += mat.components[r][c]
         return self
      return NotImplemented

   def __sub__(self, mat):
      if isinstance(mat, Matrix):
         comp = list()
         for r in range(self.M):
            row = list()
            for c in range(self.N):
               row.append(self.components[r][c]-mat.components[r][c])
            comp.append(row)
         return Matrix(comp)
      return NotImplemented

   def __isub__(self, mat):
      if isinstance(mat, Matrix):
         for r in range(self.M):
            for c in range(self.N):
               self.components[r][c] -= mat.components[r][c]
         return self
      return NotImplemented

   def __rmul__(self, fac):
      if isinstance(fac, (Complex,Fraction,int)):
         return Matrix([[fac * c for c in r] for r in self.components])
      return NotImplemented

   def __mul__(self, fac):
      if isinstance(fac, (Complex,Fraction,int)):
         return Matrix([[fac * c for c in r] for r in self.components])
      return NotImplemented

   def __matmul__(self, mat):
      if not isinstance(mat, Matrix): return NotImplemented
      comp = list()
      for r in range(self.M):
         row = list()
         for c in range(mat.N):
            temp = 0
            for cprime in range(self.N):
               temp += self.components[r][cprime]*mat.components[cprime][c]
            row.append(temp)
         comp.append(row)
      return Matrix(comp)

   def __imatmul__(self, mat):
      if not isinstance(mat, Matrix): return NotImplemented
      for r in range(self.M):
         temp = [0]*mat.N
         for c in range(mat.N):
            for cprime in range(self.N):
               temp[c] += self.components[r][cprime]*mat.components[cprime][c]
         self.components[r] = temp
      return self

   def __truediv__(self, div):
      if isinstance(div, (int,Fraction,Complex)):
         return Matrix([[c / div for c in r] for r in self.components])

   def __neg__(self):
      return Matrix([[-c for c in row] for row in self.components])

   def __eq__(self, mat):
      for r in range(self.M):
         if any(map(lambda x,y: x!=y, self.components[r], mat.components[r])):
            return False
      return True

   def __getitem__(self, idx):
      if isinstance(idx, tuple) and len(idx)==2:
         return self.components[idx[0]][idx[1]]
      raise TypeError(
         "Matrix only supports two indices specifying row and column.")

   def __setitem__(self, idx:tuple[int,int], value:Complex):
      if isinstance(idx, tuple) and len(idx)==2:
         self.components[idx[0]][idx[1]] = value
      raise TypeError(
         "Matrix only supports two indices specifying row and column.")

   def extend(self, elems:list[Complex], dim=COL):
      """Enlarge the matrix by adding a column or row to the existing matrix.

      Parameters
      ----------
      elems : list[Complex] | Matrix
         Vector to be added to the matrix.
      dim : int, optional
         Selects whether to add the vector as a `ROW` or `COL`.
         Defaults to `COL`.
      
      Raises
      ------
      AssertionError
         Indicates that the dimensions do not match.
      NotImplementedError
         Indicates that the chosen *dim* is neither 0 nor 1.
      """
      if not isinstance(elems, Matrix):
         elems = Matrix([elems])
      if dim==Matrix.ROW:
         assert self.N==0 or elems.N==self.N,\
            "Wrong dimensions: Expected %i columns but got %i."%(
            self.N,elems.N)
         self.components.extend(_copy(elems.components))
      elif dim==Matrix.COL:
         if self.N==0:
            self.components = [[_copy(c)] for c in elems]
         else:
            assert elems.M==self.M,\
               "Wrong dimensions: Expected %i rows but got %i."%(self.M,elems.M)
            for r,c in enumerate(elems.components):
               self.components[r].extend(_copy(c))
      else:
         raise NotImplementedError(
            "There are no other variants available for now (?)")

   def transpose(self):
      """Exchanges rows and columns.

      Returns
      -------
      Matrix
         Transposed matrix.
      """
      comp = list()
      for c in range(self.N):
         row = list()
         for r in range(self.M):
            row.append(self.components[r][c])
         comp.append(row)
      return Matrix(comp)

   def conjugate(self):
      """Charge conjugation.

      Returns
      -------
      Matrix
         Charge conjugated matrix.
      """
      return Matrix([[c.conjugate() for c in row] for row in self.components])

   def trace(self):
      """Computes the generalised trace.

      Returns
      -------
      Complex
         Sum of all the diagonal elements.
      """
      return sum(self.components[i][i] for i in range(min([self.M,self.N])))

   def inverse(self):
      """Inversion of the matrix.

      Uses basic implementation of Gauss elimination for inverse of complex
      rational square matrix.

      Returns
      -------
      Matrix
         The inverse of this matrix.

      Raises
      ------
      AssertionError
         Indicates that the matrix is not square and thus non-invertible.
      """
      assert self.M==self.N, "Only square matrices are invertible."
      return solve(self, Matrix.diag([1]*self.N))

   def rank(self, persistent=False):
      """
      This is an adaptation of `solve` omitting any RHS input but allowing for
      all-zero columns. If *persistent* is `True` the `Matrix` is changed into
      a generalised row-echelon form and any totally zero rows are being
      dropped. This way only the rows relevant for checks of
      linear-independence are being kept.

      Parameters
      ----------
      persistent : bool, optional
         Indicates whether to keep the generalised row-echelon form of the
         matrix as the new matrix at the end of the calculation. Defaults to
         `False`.

      Returns
      -------
      int
         Number of linearly independent rows.
      """
      if not persistent:
         return _copy(self).rank(True)
      mat = self.components

      rank = 0
      for col in range(self.N):
         rnew = rank
         while rnew<self.M and mat[rnew][col] == 0:
            rnew += 1
         # No suitable row found: skip.
         if rnew==self.M:
            continue
         if rnew!=rank:
            temp = mat[rank]
            mat[rank] = mat[rnew]
            mat[rnew] = temp
         temp = mat[rank][col]
         if temp != 1:
            mat[rank] = [c/temp for c in mat[rank]]
         rank += 1
         for row in range(rank,self.M):
            if mat[row][col] != 0:
               ratio = mat[row][col]
               mat[row][col:] = [c - ratio * sub for c,sub in\
                  zip(mat[row][col:],mat[rank-1][col:])]
      self.components = mat[:rank]
      return rank


def solve(A:Matrix, y:Matrix):
   """Solves the general linear system of equations via Gauss elimination

   .. math::
      Ax = y

   Parameters
   ----------
   A : Matrix
     Matrix acting on the :math:`x` that we are looking for.
   y : Matrix
     Right-hand side answer.

   Returns
   -------
   Matrix
      Result found for :math:`x`.
   """
   A = _copy(A)
   A.extend(y, Matrix.COL)
   mat = A.components
   N = len(mat)

   for col in range(N):
      rnew = col
      while rnew < N and mat[rnew][col] == 0:
         rnew += 1
      if rnew == N:
         raise ValueError("System of equations is not (uniquely?) solvable.")
      if rnew!=col:
         temp = mat[col]
         mat[col] = mat[rnew]
         mat[rnew] = temp
      temp = mat[col][col]
      mat[col] = [c/temp for c in mat[col]]
      for row in range(col+1,N):
         if mat[row][col] != 0:
            ratio = mat[row][col]
            mat[row][col:] = [c - ratio * sub for c,sub in\
               zip(mat[row][col:],mat[col][col:])]
   for col in range(N-1,0,-1):
      for row in range(col-1,-1,-1):
         if mat[row][col] != 0:
            ratio = mat[row][col]
            mat[row][col:] = [c - ratio * sub for c,sub in\
               zip(mat[row][col:],mat[col][col:])]
   return Matrix([row[N:] for row in mat])

