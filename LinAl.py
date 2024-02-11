import pandas as pd
import numpy as np
import copy

Amat = "Amat.dat"
Bmat = "Bmat.dat"
np.set_printoptions(precision=16)


#-----Question 1.2-----
def get_msize_nsize(myFileName):
  with open(myFileName, 'r') as myFile:
    data = myFile.read()
    msize = int(data[0])
    nsize = int(data[2])
    matrix = np.loadtxt(myFileName, skiprows=1)
    return matrix, msize, nsize
    

#------Question 2------

def print_matrix(matrix):
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
    
def calculate_trace(matrix, nsize):
  trace = 0.0
  for index in range(0,nsize):
    diag = matrix[index][index]
    trace += float(diag)   
  print("Trace of matrix: " + str(trace))
  print("-----------")
  return trace

def calculate_column_norms(matrix, nsize):
    for index in range(0,nsize):
      column_sum_sqr = 0
      for row in matrix:
        column_sum_sqr += row[index]**2
      norm = column_sum_sqr**(1/2)
      print("Norm of column " + str(index) + ": " + str(norm))
    print("-----------")

#------Question 3------
def gaussian_elimination_partial_piv(A, B):
  print("Performing Gaussian elimination with partial pivoting...\n")
  print("Original Matrices:\n")
  print("A =", end = " ")
  print_matrix(A)
  print("---")
  print("B =", end = " ")
  print_matrix(B)
  is_singular = False
  #concatenate A and B
  A_B = np.concatenate((A, B), axis = 1)
  rows, columns = A.shape
  i = 0
  #print(A_B)
  #Swapping
  while i < rows: #
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
      #print("Eliminating from row" + str(j) +"...")
      diag = A_B[i][i]
      scale = A_B[j][i]/diag
      A_B[j] = A_B[j] - (scale*A_B[i])
      #print(A_B)
    i += 1
  A = A_B[:, :columns]
  B = A_B[:, columns:]
  print("---")
  print("Resulting matrices from Gauss Elimination:\n")
  print("A =", end = " ")
  print_matrix(A)
  print("---")
  print("B =", end = " ")
  print_matrix(B)

  return A, B, is_singular
def back_substitution(A, B):
  print("Performing back substitution...\n")
  A_B = np.concatenate((A, B), axis = 1)
  rows, columns = A.shape
  x = np.zeros_like(B)
  den = A_B[rows-1][rows-1]
  num = A_B[rows-1][rows]
  x[rows - 1] = num/den
  for i in range(rows - 2, -1, -1): #start at end and work backwards
    x[i] = A_B[i][rows]
    for j in range(i +1, rows):
      x[i] = x[i] - A_B[i][j] * x[j]
    x[i] = x[i]/A_B[i][i]
  print("Solution vector x:")
  print_matrix(x)
  print("\n")
  return x
#def back_substitution(A,B):
  

def error_matrix(A_s, x, B_s):
  #print(x.shape)
  print("Producing error matrix...\n")
  #x = np.reshape(x, (4, 1))
  error_matrix = A_s @ x - B_s
  print("Error matrix:")
  print_matrix(error_matrix)
  return error_matrix
  



if __name__ == "__main__":
  #Question 2
  A_matrix, A_msize, A_nsize = get_msize_nsize(Amat)
  A_s = A_matrix
  #print_matrix(A_matrix)
  #trace = calculate_trace(A_matrix, A_nsize)
  #calculate_column_norms(A_matrix, A_nsize)

  #Printing matrix b with dimensions
  B_matrix, B_msize, B_nsize = get_msize_nsize(Bmat)
  B_s = B_matrix
  #print_matrix(B_matrix)

  #Question 3
  A, B, is_singular = gaussian_elimination_partial_piv(A_matrix, B_matrix)
  if is_singular == True:
    print("A is a singular matrix!!!")
  if is_singular == False:
    print("A is not singular")
  print("-----------")
  x = back_substitution(A,B)
  print("-----------")
  error = error_matrix(A_s, x, B_s)
  error_norms = calculate_column_norms(error, A_nsize)


#ref https://www.youtube.com/watch?v=DiZ0zSzZj1g
x = np.linalg.solve(A, B)
print("Actual solution:")
print_matrix(x)
 #SO CLOSE NEED TO ITERATE X ACTUAL X AND MY X HAVE SAME FIRST COLUMN