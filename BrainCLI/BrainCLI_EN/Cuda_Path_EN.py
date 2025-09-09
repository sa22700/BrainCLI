"""
Copyright [2025] [Pirkka Toivakka]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
# This project uses model weights licensed under CC BY 4.0 (see /Models/LICENSE)


from array import array
import ctypes
import os

try:
    lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "../Models/libmatrix.so"))
    cuda_available = True

except OSError:
    lib = None
    cuda_available = False

def flatten(matrix):
    return [float(item) for row in matrix for item in row]

def gpu_dot(a, b, n, m, k):
    a_flat = array('f', flatten([list(row) for row in a]))
    b_flat = array('f', flatten([list(row) for row in b]))
    c_flat = array('f', [0.0] * (n * k))
    ptr_f = ctypes.POINTER(ctypes.c_float)
    lib.matmul.argtypes = [ptr_f, ptr_f, ptr_f, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.matmul.restype = None
    lib.matmul(
        ctypes.cast(a_flat, ptr_f),
        ctypes.cast(b_flat, ptr_f),
        ctypes.cast(c_flat, ptr_f),
        n, m, k
    )
    return [[float(c_flat[i * k + j]) for j in range(k)] for i in range(n)]

def gpu_add(a, b, n, m):
    a_flat = array('f', flatten([list(row) for row in a]))
    b_flat = array('f', flatten([list(row) for row in b]))
    c_flat = array('f', [0.0] * (n * m))
    ptr_f = ctypes.POINTER(ctypes.c_float)
    lib.addmat.argtypes = [ptr_f, ptr_f, ptr_f, ctypes.c_int, ctypes.c_int]
    lib.addmat.restype = None
    lib.addmat(
        ctypes.cast(a_flat, ptr_f),
        ctypes.cast(b_flat, ptr_f),
        ctypes.cast(c_flat, ptr_f),
        n, m
    )
    return [[float(c_flat[i * m + j]) for j in range(m)] for i in range(n)]

def gpu_subtract(a, b, n, m):
    a_flat = array('f', flatten([list(row) for row in a]))
    b_flat = array('f', flatten([list(row) for row in b]))
    c_flat = array('f', [0.0] * (n * m))
    ptr_f = ctypes.POINTER(ctypes.c_float)
    lib.matsub.argtypes = [ptr_f, ptr_f, ptr_f, ctypes.c_int, ctypes.c_int]
    lib.matsub.restype = None
    lib.matsub(
        ctypes.cast(a_flat, ptr_f),
        ctypes.cast(b_flat, ptr_f),
        ctypes.cast(c_flat, ptr_f),
        n, m
    )
    return [[float(c_flat[i * m + j]) for j in range(m)] for i in range(n)]

def gpu_elementwise_multiply(a, b, n, m):
    a_flat = array('f', flatten([list(row) for row in a]))
    b_flat = array('f', flatten([list(row) for row in b]))
    c_flat = array('f', [0.0] * (n * m))
    ptr_f = ctypes.POINTER(ctypes.c_float)
    lib.matmply.argtypes = [ptr_f, ptr_f, ptr_f, ctypes.c_int, ctypes.c_int]
    lib.matmply.restype = None
    lib.matmply(
        ctypes.cast(a_flat, ptr_f),
        ctypes.cast(b_flat, ptr_f),
        ctypes.cast(c_flat, ptr_f),
        n, m
    )
    return [[float(c_flat[i * m + j]) for j in range(m)] for i in range(n)]



def clip_matrix(mat, min_val, max_val):
    for row in mat.rows:
        for i in range(len(row)):
            if row[i] > max_val:
                row[i] = max_val
            elif row[i] < min_val:
                row[i] = min_val