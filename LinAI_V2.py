import pandas as pd
import numpy as np
import copy

Amat = "Amat.dat"
Bmat = "Bmat.dat"

def get_size(myFileName):
  with open(myFileName, 'r') as myFile:
    data = myFile.read()
    msize = int(data[0])
    nsize = int(data[2])
    matrix = np.loadtxt(myFileName, skiprows=1)
    return matrix, msize, nsize
def print_matrix(matrix, name):
  rows, columns = matrix.shape
  print("Printing "+ name)
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
def trace(matrix):
  trace = 0.0
  rows, columns = matrix.shape
  for index in range(0, rows):
    diag = matrix[index][index]
    trace += float(diag)
  print("Trace of matrix: " + str(trace))
  print("-----------")
  return trace
def col_norms(matrix):
  rows, columns = matrix.shape
  for index in range(columns):
    col_sqrt = 0
    for row in matrix:
      col_sqrt += row[index]**2
    norm = col_sqrt**(1/2)
    print("Norm of column " + str(index) + ": " + str(norm))
  print("-----------")
def error_matrix(A_s, x, B_s):
  print("Producing error matrix...\n")
  #x = np.reshape(x, (4, 1))
  error_matrix = A_s @ x - B_s
  print("Error matrix:")
  print_matrix(error_matrix, "Error Matrix")
  return error_matrix

#------Question 3------
def gaussian_elimination_partial_piv(A, B):
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
    if diag == 0.0:
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
def back_sub(A, B):
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
    print("Solution vector x:")
    print_matrix(x, "X")
    print("\n")
  return x

#------Question 4------
def LU_decomp(A): 
  print("Performing LU Decomposition...")
  singular = False
  rows, columns = A.shape
  s = list(range(rows)) #python indexing!
  for col in range(0, columns):
    j = col
    column = A[:, col]
    k = np.argmax(np.abs(column)) #index
    p = np.max(column) # value
    if k != j:
      k_row = np.copy(A[k, :])
      j_row = np.copy(A[j, :])
      A[j, :] = k_row
      A[k, :] = j_row
      s_j = np.copy(s[j])
      s_k = np.copy(s[k])
      s[j] = s_k
      s[k] = s_j
    if A[j][j] == 0:
      singular = True
      return A, s, singular
    for i in range(j+1, rows):
      A[i][j] = A[i][j]/A[j][j]
      for k in range(j+1, rows):
        A[i][k] = A[i][k] - A[i][j]*A[j][k]
  print_matrix(A, "A after LU")
  return A, s, singular

def LU_backsub(A, B, s):
  print("Performing LU Backsubstitution...")
  A_rows, A_cols = A.shape
  B_rows, B_cols = B.shape
  X = np.zeros_like(B)
  # loop through B columns
  for col in range(B_cols):
    x = np.zeros(B_cols)
    b = B[:,col]
    y = list(range(len(s)))
    m = A_rows
    #initialize y 
    for j in range(1,m-1):
      y[j] = b[s[j]]
    #forward substitution
    for j in range(1,m-1):
      for i in range(j+1, m):
        y[i]= y[i]-y[j]*A[i][j]
    #backsub
    for i in range(m, 1, -1): #indexing? 
      if A[i][i] == 0 :
        print("Singular")
        return X
      sum = 0.0
      for k in range(i+1, m):
        sum += A[i][k]*x[k]
      x[i] = (y[i]-sum)/A[i][i]
    X[:, col]=x
  print_matrix(X, "X from LU Decomposition")
  return X


if __name__ == "__main__":
  # A_matrix, A_rows, A_cols = get_size(Amat)
  # B_matrix, B_rows, B_cols = get_size(Bmat)
  # A_s = A_matrix
  # B_s = B_matrix

  # #Question 3
  # A, B, is_singular = gaussian_elimination_partial_piv(A_matrix, B_matrix)
  # if is_singular == True:
  #   print("A is singular \n -----------")
  # if is_singular == False:
  #   print("A is not singular \n -----------")
  # x = back_sub(A,B)
  # print("-----------")
  # error = error_matrix(A_s, x, B_s)
  # error_norms = col_norms(error)

  #Question 4
  # print_matrix(A_matrix, "A before LU")
  # U_A, swaps, singular = LU_decomp(A_matrix)
  # X = LU_backsub(A_LU, B_matrix, swaps)

  #LU Test
  A_t = np.array([[1, 2], [3, 4]])
  B_t = np.array([[6, 9], [0, 0]])
  U_t, swaps_t, singular_t = LU_decomp(A_t)
  print("test swaps:" + str(swaps_t))
  X = LU_backsub(U_t, B_t, swaps_t)

  #Question 5
  # Q5 = np.array([[1,2,3],[-3,2,5],[np.pi, np.e, -(2**(1/2))]])
  # U_5, swaps_5, singular_5 = LU_decomp(Q5)
  # X_5 = LU_backsub(U_5, B) what to put for B? 

