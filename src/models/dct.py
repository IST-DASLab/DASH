import torch
import math

def _init_transform_square(n):
    lin = torch.arange(n)
    I = lin.repeat(n, 1).to(torch.float)
    Q = math.sqrt(2 / n) * torch.cos(torch.pi * (I.t() * (2. * I + 1.)) / (2. * n))
    Q[0, :] *= math.sqrt(0.5)
    return Q

class DCT2d:
    def __init__(self, R, C, dtype):
        """
        Initializes the DCT-II transformation matrix or matrices for a 2-dimensional input `x`.
        If `x` is a square matrix, this class only initializes `Q`.
        If `x` is a rectangular matrix, this class initializes `Q_R` and `Q_C`.
        Args:
            R: number of rows of the input matrix `x` to be transformed
            C: number of columns of the input matrix `x` to be transformed
            dtype: data type for the transformation matrices
        """
        self.is_square = (R == C)
        if self.is_square:
            self.Q = _init_transform_square(R).to(dtype).cuda()
            self.Q.requires_grad = False
        else:
            self.Q_R = _init_transform_square(R).to(dtype).cuda()
            self.Q_C = _init_transform_square(C).to(dtype).cuda()
            self.Q_R.requires_grad = False
            self.Q_C.requires_grad = False

    def transform(self, x, inverse=False):
        if self.is_square:
            if inverse:
                return self.Q.T @ x @ self.Q
            else:
                return self.Q @ x @ self.Q.T
        else:
            if inverse:
                return self.Q_R.T @ x @ self.Q_C
            else:
                return self.Q_R @ x @ self.Q_C.T

class DCT1d:
    def __init__(self, n, dtype):
        """
        Initializes the 1d DCT-II transformation matrix for a 2-dimensional input `x` (to be applied per row, acros all columns)
        Multiplies the input to the right by Q^T.
        Args:
            n: number of columns of the input matrix `x` to be transformed (Q^T is multiplied to the right)
            dtype: data type for the transformation matrices
        """
        self.Q = _init_transform_square(n).to(dtype).cuda()
        self.Q.requires_grad = False

    def transform(self, x, inverse=False):
        if inverse:
            return x @ self.Q
        else:
            return x @ self.Q.T
