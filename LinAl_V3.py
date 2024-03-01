import pandas as pd
import numpy as np
import copy
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#turn matrix into python readable
def get_size(myFileName):
  with open(myFileName, 'r') as myFile:
    data = myFile.read()
    msize = int(data[0])
    nsize = int(data[2])
    matrix = np.loadtxt(myFileName, skiprows=1)
    return matrix, msize, nsize

def trace(matrix, m_dummy, output_dummy):
  trace = 0.0
  rows, columns = matrix.shape
  for index in range(0, rows):
    diag = matrix[index][index]
    trace += float(diag)
  print("Trace of matrix: " + str(trace))
  print("-----------")
  return trace

def col_norms(matrix, dimension_dummy, output_dummy):
  rows, columns = matrix.shape
  for index in range(columns):
    col_sqrt = 0
    for row in matrix:
      col_sqrt += row[index]**2
    norm = col_sqrt**(1/2)
    print("Norm of column " + str(index) + ": " + str(norm))
  print("-----------")

def print_matrix(matrix, mn_dummy):
  rows, columns = matrix.shape
  print("Matrix of size " + str(rows) + "x" + str(columns) + ":")
  for row in matrix:
    #print(row)
    index = 0
    for cell in row:
      index += 1
      if index != columns:
        print(cell, end = " , ")
      if index == columns:
        print(cell,  "\n")  

def error_matrix(A_s, B_s, x):
  print("Producing error matrix...\n")
  #x = np.reshape(x, (4, 1))
  error_matrix = A_s @ x - B_s
  print("Error matrix:")
  print_matrix(error_matrix, "Error Matrix")
  return error_matrix

'''Now write a driver (or calling) program that takes a filename as an argument,
reads the matrix A from the Amat.dat file, and performs the following tests:'''
def driver_1(filename):
  print("Getting matrix from " + filename + "...")
  matrix, A_rows, A_cols = get_size(filename)
  print_matrix(matrix, dummy)
  trace(matrix, dummy, dummy)
  col_norms(matrix, dummy, dummy)
  return matrix

'''Gaussian elimination with partial pivoting'''
def gaussian_elimination_partial_piv(A, B, dummy1, dummy2, dummy3):
  print("Performing Gaussian elimination with partial pivoting...\n")
  print("Original Matrices:\n")
  print("A =", end = " ")
  print_matrix(A, "A")
  print("B =", end = " ")
  print_matrix(B, "B")
  is_singular = False
  A_B = np.concatenate((A,B), axis = 1)
  rows, columns = A.shape
  i = 0 
  while i < rows:
    for p in range(i+1,rows): #element under diagonals 
      diag = A_B[i,i]
      under_diag = A_B[p,i]
      if abs(diag) < abs(under_diag): 
        #print("Swapping rows " + str(i) + " with "+ str(p) + "...")
        tmp_1 = list(A_B[i, :])
        tmp_2 = list(A_B[p, :])
        A_B[i, :] = np.array(tmp_2)
        A_B[p, :] = np.array(tmp_1)
        #print(A_B)
    #Checking if it's singular
    if abs(diag) <= emach :
      is_singular = True
      print("Singular...")
      A = A_B[:, :columns]
      B = A_B[:, columns:]
      return A, B, is_singular
    #Elimination
    for j in range(i+1,rows): #element under diagonals 
      diag = A_B[i][i]
      scale = A_B[j][i]/diag
      A_B[j] = A_B[j] - (scale*A_B[i])
    i += 1
  A = A_B[:, :columns]
  B = A_B[:, columns:]
  print("---")
  print("Resulting matrices from Gauss Elimination:\n")
  print("A =", end = " ")
  print_matrix(A, "A")
  print("---")
  print("B =", end = " ")
  print_matrix(B, "B")
  return A,B,is_singular
'''Back substitution for gaussian elimination'''
def back_sub_GE(A, B):
  print("Performing back substitution...\n")
  A_B = np.concatenate((A,B), axis = 1)
  rows, columns = A.shape
  b_cols = B.shape[1]
  x = np.zeros_like(B)
  for col in range(b_cols):
    x[rows-1, col] = A_B[rows-1, columns+col] / A_B[rows-1, rows-1]
    for i in range(rows - 2, -1, -1):
      x[i, col] = A_B[i, columns+col]
      for j in range(i+1, rows):
        x[i, col] = x[i, col] - A_B[i, j] * x[j, col] 
      x[i, col] = x[i, col] / A_B[i, i]
  print("Solution Matrix X from Gaussian Elimination:")
  print_matrix(x, dummy)
  print("\n")
  return x
'''Lu decomposition'''
def LU_decomp(Ainput, dummym, dummysingular, dummys):
  A = np.copy(Ainput)
  print("Performing LU Decomposition...")
  print("Original Matrix before LU Decomposition:")
  print_matrix(A, dummy)
  singular = False
  rows, columns = A.shape
  #filling permutation vector
  s = [i for i in range(rows)]
  #loop through columns
  for j in range(0, columns):
    diag_abs = abs(A[j,j])
    k = j
    for i in range(j+1,columns):
        if abs(A[i,j]) > diag_abs:
            diag_abs = abs(A[i,j])
            k = i # index
    #swap
    if k != j:
      #row swap  
      k_row = np.copy(A[k, :])
      j_row = np.copy(A[j, :])
      A[j, :] = np.copy(k_row)
      A[k, :] = np.copy(j_row)
      #permutation swap 
      s_j = s[j]
      s[j] = s[k]
      s[k] = s_j
    #check for singularity 
    if (abs(A[j, j]) <= emach):
      singular = True
      return A, s, singular
    #zero lower diags 
    for i in range(j+1, rows):
        A[i, j] /= A[j, j]
        for t in range(j+1, rows):
            A[i,t] -=  A[i,j]*A[j,t]
    #Get L U:
    U = np.zeros_like(A)
    L = np.zeros_like(A)
    for i in range(0, rows):
      for j in range(0, rows):
        if j < i:
          U[i,j] = 0 
          L[i,j] = A[i,j]
        if j > i:
          U[i,j] = A[i,j]
        if j == i:
          U[i,j] = A[i,j]
          L[i,j] = 1.0
  print("Matrix after LU decomposition:")
  print_matrix(A, dummy)
  print("L Matrix:")
  print_matrix(L, dummy)
  print("U Matrix:")
  print_matrix(U, dummy)
  return A, s, singular

'''Lu backsubstitution'''
def LU_backsub(A, dummydimension, B, s):
  print("Performing LU Backsubstitution...")
  A_rows, A_cols = A.shape
  B_rows, B_cols = B.shape
  X = np.zeros_like(B)
  Y = np.zeros_like(B)
  m = B_rows
  n = B_cols
  #initialize y 
  m = A_rows #n = cols
  for j in range(0,m):
    Y[j, :] = B[s[j], :]
  # loop through B columns
  for j in range(0, m - 1):
    for i in range(j + 1, m):
      Y[i, :] = Y[i, : ] - Y[j, :]*A[i, j]
  #backsubstitution
  for j in range(0, n):
    for i in range(m - 1, -1, -1):
      if abs(A[i,i]) <= emach:
        print("Is singular")
        return 
      summation = 0.0
      for k in range(i+1, m):
        summation += A[i,k]*X[k, j]
      X[i, j] = (Y[i, j] - summation)/A[i,i]
  return X
def plot(X_5, D, Coefs):
  a = X_5[0]
  b = X_5[1]
  c = X_5[2]
  point1 = Coefs[0]
  point2 = Coefs[1]
  point3 = Coefs[2]

  xs = np.linspace(-5, 6, 10)
  ys = np.linspace(-5, 6, 10)
  #original ax + by + cz = d
  #z  = (d - ax - by)/c
  X, Y = np.meshgrid(xs, ys)
  Z = plane(X, Y, a, b, c)
  fig = plt.figure()
  ax = fig.add_subplot(projection = '3d')
  surf = ax.plot_surface(X, Y, Z, alpha =.5)

  scatter1 = ax.scatter(point1[0], point1[1], point1[2], label = "Point A")
  scatter2 = ax.scatter(point2[0], point2[1], point2[2], label = "Point B")
  scatter3 = ax.scatter(point3[0], point3[1], point3[2], label = "Point C")
  
  plt.title("Plane Coefficients with LU Decomposition")

  plt.xlabel("X Axis")
  plt.ylabel("Y Axis")

  plt.show()
def plane(x, y, a, b, c):
  d = 1
  z = (d - a*x - b*y)/c
  return z 
if __name__ == "__main__":
  global dummy, emach 
  dummy = " " #used to make input easier for grading, necessary inputs are calculated by the function
  emach = 10**(-16)

  #Driver to print matrix, get trace, and get column norm, the matrix is returned
  A = driver_1("Amat.dat")
  B = driver_1("Bmat.dat")

###----------Question 3----------
  #Gaussian elimination reduction
  A_GE, B_GE, is_singular = gaussian_elimination_partial_piv(A, B, dummy, dummy, dummy)   
  #Gaussian backsubstitution
  X_GE = back_sub_GE(A_GE,B_GE)    
  #Gaussian Error
  A_s = A 
  B_s = B
  error_GE = error_matrix(A_s, B_s, X_GE)
  error_norms_GE = col_norms(error_GE, dummy, dummy)

###----------Question 4----------
  #LU decomposition
  U, swaps, singular = LU_decomp(A, dummy, dummy, dummy)
  #LU backsubstitution
  X_LU = LU_backsub(U, dummy, B, swaps)
  #LU error
  A_s = A
  B_s = B
  error_LU = error_matrix(A_s, B_s, X_LU)
  error_norms_LU = col_norms(error_LU, dummy, dummy)
  
###----------Question 5----------
  d = 1.0
  D = np.array([[d], [d], [d]])
  Coefs = np.zeros((3, 3))

  Coefs[:, 0] = [1, 2, 3]
  Coefs[:, 1] = [-3, 2, 5]
  Coefs[:, 2] = [np.pi, np.e, -(2**(1/2))]

  U_5, swaps_5, singular_5 = LU_decomp(Coefs, dummy, dummy, dummy)
  X_5 = LU_backsub(U_5, dummy, D, swaps_5)
  print_matrix(X_5, dummy)

  plot(X_5, D, Coefs) #couldnt get to work

  #Check :) 
  #print(scipy.linalg.lu(A))
  # I accidentally erased my reference, I think it was this one:
  # https://www.youtube.com/watch?v=eDb6iugi6Uk