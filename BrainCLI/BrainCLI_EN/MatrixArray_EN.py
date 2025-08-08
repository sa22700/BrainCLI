'''
Copyright [2025] [Pirkka Toivakka]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
# This project uses model weights licensed under CC BY 4.0 (see /Models/LICENSE)


from array import array
from operator import mul
from BrainCLI.BrainCLI_EN.Debug_Log_EN import log_error
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

class BrainRandom:
    def __init__(self, seed=12345):
        self.state = seed

    def set_seed(self, new_seed):
        self.state = new_seed

    def rand(self):
        self.state = (1664525 * self.state + 1013904223) % (2 ** 32)
        return self.state / (2 ** 32)

    def uniform(self, min_val, max_val):
        return min_val + (max_val - min_val) * self.rand()

    def random_matrix(self, shape, min_val=-0.01, max_val=0.01):
        return [[self.uniform(min_val, max_val)
                 for _ in range(shape[1])]
                for _ in range(shape[0])]

brain_random = BrainRandom()

class BrainMatrix:
    def __init__(self, data, use_gpu=False):
        self._rows = [array('d', row) for row in data]
        self.shape = (len(self._rows), len(self._rows[0]) if self._rows else 0)
        self._cols = None
        self.use_gpu = use_gpu and cuda_available

    def __repr__(self):
        return f"BrainMatrix(shape={self.shape})"

    @property
    def cols(self):
        if self._cols is None:
            self._cols = [array('d', col) for col in zip(*self._rows)]
        return self._cols

    def to_list(self):
        return [list(row) for row in self._rows]

    def array_dot(self, other: 'BrainMatrix') -> 'BrainMatrix':
        try:
            if self.shape[1] != other.shape[0]:
                raise ValueError("Matrix dimensions do not match.")

        except Exception as e:
            print(f"Error: {e}")
            log_error(f"Error: {e}")
            raise

        if self.use_gpu or getattr(other, "use_gpu", False):
            if lib is not None:
                n, m = self.shape
                _, k = other.shape
                result = gpu_dot(self._rows, other._rows, n, m, k)
                return BrainMatrix(result, use_gpu=True)
        cols = other.cols
        result = [[sum(map(mul, r, c)) for c in cols] for r in self._rows]
        return BrainMatrix(result)

    def array_add(self, other: 'BrainMatrix') -> 'BrainMatrix':
        try:
            if self.shape != other.shape:
                raise ValueError("Matrix dimensions do not match.")

        except Exception as e:
            print(f"Error: {e}")
            log_error(f"Error: {e}")
            raise

        if (self.use_gpu or getattr(other, "use_gpu", False)) and lib is not None:
            n, m = self.shape
            result = gpu_add(self._rows, other._rows, n, m)
            return BrainMatrix(result, use_gpu=True)
        result = [[r[i] + o[i]
                   for i in range(self.shape[1])]
                  for r, o in zip(self._rows, other._rows)]
        return BrainMatrix(result)

    def array_subtract(self, other: 'BrainMatrix') -> 'BrainMatrix':
        try:
            if self.shape != other.shape:
                raise ValueError("Matrix dimensions do not match.")

        except Exception as e:
            print(f"Error: {e}")
            log_error(f"Error: {e}")
            raise

        if (self.use_gpu or getattr(other, "use_gpu", False)) and lib is not None:
            n, m = self.shape
            result = gpu_subtract(self._rows, other._rows, n, m)
            return BrainMatrix(result, use_gpu=True)
        result = [[r[i] - o[i]
                   for i in range(self.shape[1])]
                  for r, o in zip(self._rows, other._rows)]
        return BrainMatrix(result)

    def array_scale(self, scalar: float) -> 'BrainMatrix':
        result = [[r[i] * scalar for i in range(self.shape[1])]
                  for r in self._rows]
        return BrainMatrix(result)

    def elementwise_multiply(self, other: 'BrainMatrix') -> 'BrainMatrix':
        try:
            if self.shape != other.shape:
                raise ValueError("Matrix dimensions do not match.")
        except Exception as e:
            print(f"Error: {e}")
            log_error(f"Error: {e}")
            raise

        if (self.use_gpu or getattr(other, "use_gpu", False)) and lib is not None and hasattr(lib, "elemul"):
            n, m = self.shape
            result = gpu_elementwise_multiply(self._rows, other._rows, n, m)
            return BrainMatrix(result, use_gpu=True)

        result = [[r[i] * o[i]
                   for i in range(self.shape[1])]
                  for r, o in zip(self._rows, other._rows)]
        return BrainMatrix(result)

    def transpose(self) -> 'BrainMatrix':
        return BrainMatrix([list(col) for col in zip(*self._rows)])

    @staticmethod
    def array_random(shape: tuple[int, int], min_val=-1, max_val=1) -> 'BrainMatrix':
        return BrainMatrix(brain_random.random_matrix(shape, min_val, max_val))

    def array_activation(self, deriv: bool = False) -> 'BrainMatrix':
        if deriv:
            result = [[1 if val > 0 else 0 for val in row] for row in self._rows]
        else:
            result = [[val if val > 0 else 0 for val in row] for row in self._rows]
        return BrainMatrix(result)

    @property
    def rows(self):
        return self._rows

class BrainLayer:
    def __init__(self, input_size: int, output_size: int):
        self.weights = BrainMatrix.array_random((input_size, output_size), min_val=-0.0001, max_val=0.0001)
        self.biases = BrainMatrix.array_random((1, output_size), min_val=-0.0001, max_val=0.0001)
        self.inputs = None
        self.outputs = None

    def array_push(self, inputs):
        mat = inputs if isinstance(inputs, BrainMatrix) else BrainMatrix(inputs)
        self.inputs = BrainMatrix(mat.to_list())
        self.outputs = self.inputs.array_dot(self.weights)\
                            .array_add(self.biases)\
                            .array_activation()
        return self.outputs

    def array_backpropagate(self, error: BrainMatrix, learning_rate: float = 0.0005) -> BrainMatrix:
        if self.outputs is None or self.inputs is None:
            raise ValueError("Call array_push first.")
        d_act = self.outputs.array_activation(deriv=True)
        delta = error.elementwise_multiply(d_act)
        weight_grad = self.inputs.transpose().array_dot(delta).array_scale(learning_rate)
        self.weights = self.weights.array_subtract(weight_grad)
        bias_grad = delta.array_scale(learning_rate)
        self.biases = self.biases.array_subtract(bias_grad)
        clip_matrix(self.weights, -1.0, 1.0)
        clip_matrix(self.biases, -1.0, 1.0)
        return delta.array_dot(self.weights.transpose())

    def get_params(self):
        return {"weights": self.weights, "biases": self.biases}

    def set_params(self, params):
        self.weights = params["weights"]
        self.biases = params["biases"]

class BrainNetwork:
    def __init__(self, layers: list[BrainLayer]):
        self.layers = layers

    def array_predict(self, inputs):
        mat = inputs if isinstance(inputs, BrainMatrix) else BrainMatrix(inputs)
        output = mat
        for layer in self.layers:
            output = layer.array_push(output)
        return output.to_list()

    def train(self, inputs, expected_output, learning_rate: float = 0.0005):
        prediction = inputs
        for layer in self.layers:
            prediction = layer.array_push(prediction)
        error = BrainMatrix(expected_output).array_subtract(prediction)
        for layer in reversed(self.layers):
            error = layer.array_backpropagate(error, learning_rate)

    def get_weights(self):
        return [layer.get_params() for layer in self.layers]

    def set_weights(self, weights_list):
        for layer, params in zip(self.layers, weights_list):
            layer.set_params(params)
