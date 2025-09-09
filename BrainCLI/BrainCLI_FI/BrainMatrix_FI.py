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


from operator import mul
from array import array
from BrainCLI.BrainCLI_FI.Cuda_Path_FI import cuda_available, gpu_dot, gpu_add, gpu_subtract, gpu_elementwise_multiply, lib
from BrainCLI.BrainCLI_FI.BrainRandom_FI import brain_random
from BrainCLI.BrainCLI_FI.Debug_Log_FI import log_error

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
        # cache transposed columns
        if self._cols is None:
            self._cols = [array('d', col) for col in zip(*self._rows)]
        return self._cols

    def to_list(self):
        return [list(row) for row in self._rows]

    def array_dot(self, other: 'BrainMatrix') -> 'BrainMatrix':
        try:
            if self.shape[1] != other.shape[0]:
                raise ValueError("Matriisien koot eivät ole sama.")

        except Exception as e:
            print(f"Virhe: {e}")
            log_error(f"Virhe: {e}")
            raise

        if (self.use_gpu or getattr(other, "use_gpu", False)) and lib is not None and hasattr(lib, "matmul"):
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
                raise ValueError("Matriisien koot eivät täsmää yhteenlaskuun.")

        except Exception as e:
            print(f"Virhe: {e}")
            log_error(f"Virhe: {e}")
            raise

        if (self.use_gpu or getattr(other, "use_gpu", False)) and lib is not None and hasattr(lib, "addmat"):
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
                raise ValueError("Matriisien koot eivät täsmää vähennyslaskuun.")

        except Exception as e:
            print(f"Virhe: {e}")
            log_error(f"Virhe: {e}")
            raise

        if (self.use_gpu or getattr(other, "use_gpu", False)) and lib is not None and lib is not None and hasattr(lib, "matsub"):
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
                raise ValueError("Matriisien koot eivät ole sama.")
        except Exception as e:
            print(f"Error: {e}")
            log_error(f"Error: {e}")
            raise

        if (self.use_gpu or getattr(other, "use_gpu", False)) and lib is not None and hasattr(lib, "matmply"):
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
