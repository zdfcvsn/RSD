import numpy as np

def rowtomat(row,lines,visual_filed):
    mat = np.zeros((lines,160))
    for i in range(160):
        mat[visual_filed-1:visual_filed+len(row)-1,i]=row
    return mat*255