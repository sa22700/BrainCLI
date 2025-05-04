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

    def random_matrix(self, shape, min_val=-1, max_val=1):
        return [[self.uniform(min_val, max_val)
                 for _ in range(shape[1])]
                for _ in range(shape[0])]

brain_random = BrainRandom()

class BrainMatrix:
    def __init__(self, data):
        # data: list of lists of floats
        self._rows = [array('d', row) for row in data]
        self.shape = (len(self._rows), len(self._rows[0]) if self._rows else 0)
        self._cols = None

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
        if self.shape[1] != other.shape[0]:
            raise ValueError("Matriisit eivät-sopivia.")
        rows = self._rows
        cols = other.cols
        result = [[sum(map(mul, r, c)) for c in cols] for r in rows]
        return BrainMatrix(result)

    def array_add(self, other: 'BrainMatrix') -> 'BrainMatrix':
        if self.shape != other.shape:
            raise ValueError("Matriisien koot eivät täsmää yhteenlaskuun.")
        result = [[r[i] + o[i]
                   for i in range(self.shape[1])]
                  for r, o in zip(self._rows, other._rows)]
        return BrainMatrix(result)

    def array_subtract(self, other: 'BrainMatrix') -> 'BrainMatrix':
        if self.shape != other.shape:
            raise ValueError("Matriisien koot eivät täsmää vähennyslaskuun.")
        result = [[r[i] - o[i]
                   for i in range(self.shape[1])]
                  for r, o in zip(self._rows, other._rows)]
        return BrainMatrix(result)

    def array_scale(self, scalar: float) -> 'BrainMatrix':
        result = [[r[i] * scalar for i in range(self.shape[1])]
                  for r in self._rows]
        return BrainMatrix(result)

    def elementwise_multiply(self, other: 'BrainMatrix') -> 'BrainMatrix':
        if self.shape != other.shape:
            raise ValueError("Elementwise-multiplication requires same shapes.")
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
            result = [[1 if val > 0 else 0 for val in row]
                      for row in self._rows]
        else:
            result = [[val if val > 0 else 0 for val in row]
                      for row in self._rows]
        return BrainMatrix(result)


class BrainLayer:
    def __init__(self, input_size: int, output_size: int):
        self.weights = BrainMatrix.array_random((input_size, output_size))
        self.biases = BrainMatrix.array_random((1, output_size))
        self.inputs = None
        self.outputs = None

    def array_push(self, inputs):
        mat = inputs if isinstance(inputs, BrainMatrix) else BrainMatrix(inputs)
        self.inputs = BrainMatrix(mat.to_list())
        self.outputs = self.inputs.array_dot(self.weights)\
                            .array_add(self.biases)\
                            .array_activation()
        return self.outputs

    def array_backpropagate(self, error: BrainMatrix, learning_rate: float = 0.001) -> BrainMatrix:
        if self.outputs is None or self.inputs is None:
            raise ValueError("Call array_push before backpropagation.")
        d_act = self.outputs.array_activation(deriv=True)
        delta = error.elementwise_multiply(d_act)
        weight_grad = self.inputs.transpose().array_dot(delta).array_scale(learning_rate)
        self.weights = self.weights.array_subtract(weight_grad)
        bias_grad = delta.array_scale(learning_rate)
        self.biases = self.biases.array_subtract(bias_grad)
        return delta.array_dot(self.weights.transpose())


class BrainNetwork:
    def __init__(self, layers: list[BrainLayer]):
        self.layers = layers

    def array_predict(self, inputs):
        mat = inputs if isinstance(inputs, BrainMatrix) else BrainMatrix(inputs)
        output = mat
        for layer in self.layers:
            output = layer.array_push(output)
        return output.to_list()

    def train(self, inputs, expected_output, learning_rate: float = 0.001):
        # forward pass
        prediction = inputs
        for layer in self.layers:
            prediction = layer.array_push(prediction)
        # backward pass
        error = BrainMatrix(expected_output).array_subtract(prediction)
        for layer in reversed(self.layers):
            error = layer.array_backpropagate(error, learning_rate)
