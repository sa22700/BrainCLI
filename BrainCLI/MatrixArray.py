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
            for _ in range(shape[1])] for _ in range(shape[0])]

brain_random = BrainRandom()

class BrainMatrix:
    def __init__(self, data):
        self.data = data
        self.shape = (len(data), len(data[0]) if isinstance(data[0], list) else 1)

    def __repr__(self):
        return f"BrainMatrix(shape={self.shape}, data={self.data})"

    def array_dot(self, other):
        if self.shape[1] != other.shape[0]:
            raise ValueError("Matriisien koot eivät täsmää kertolaskuun.")

        result = [[sum(a * b
            for a, b in zip(row, col))
                for col in zip(*other.data)]
                for row in self.data]
        return BrainMatrix(result)

    def array_add(self, other):
        if (not isinstance(other, BrainMatrix)
        or self.shape != other.shape):
            raise ValueError("Matriisien koot eivät täsmää yhteenlaskuun.")

        return BrainMatrix([[self.data[i][j] + other.data[i][j]
            for j in range(self.shape[1])]
            for i in range(self.shape[0])])

    def array_activation(self, deriv=False):
        if deriv:
            return BrainMatrix([[1
            if val > 0
            else 0
            for val in row]
            for row in self.data])

        return BrainMatrix([[max(0, val)
            for val in row]
            for row in self.data])

    def array_subtract(self, other):
        if (not isinstance(other, BrainMatrix)
        or self.shape != other.shape):
            raise ValueError("Matriisien koot eivät täsmää vähennyslaskuun.")

        return BrainMatrix([[self.data[i][j] - other.data[i][j]
            for j in range(self.shape[1])]
            for i in range(self.shape[0])])

    def array_scale(self, scalar):
        return BrainMatrix([[val * scalar
            for val in row]
            for row in self.data])

    @staticmethod
    def array_random(shape, min_val=-1, max_val=1):
        return BrainMatrix(brain_random.random_matrix(shape, min_val, max_val))

    def to_list(self):
        return self.data \
            if isinstance(self.data[0], list) \
            else [self.data]

    def elementwise_multiply(self, other):
        if self.shape != other.shape:
            raise ValueError("Matriisien on oltava samankokoisia elementtikohtaista kertolaskua varten.")

        return BrainMatrix([[self.data[i][j] * other.data[i][j]
            for j in range(self.shape[1])]
            for i in range(self.shape[0])])

    def transpose(self):
        return BrainMatrix([list(col)
            for col in zip(*self.data)])

class BrainLayer:
    def __init__(self, input_size, output_size):
        self.weights = BrainMatrix.array_random((input_size, output_size))
        self.biases = BrainMatrix.array_random((1, output_size))
        self.inputs = None
        self.outputs = None

    def array_push(self, inputs):
        if isinstance(inputs, BrainMatrix):
            inputs = inputs.to_list()

        self.inputs = BrainMatrix([row[:]
            for row in inputs])

        self.outputs = self.inputs.array_dot(self.weights) \
            .array_add(self.biases) \
            .array_activation()
        return self.outputs

    def array_backpropagate(self, error, learning_rate=0.001):
        if (self.outputs is None
            or self.inputs is None):
            raise ValueError("Virhe: array_push() täytyy kutsua ennen backpropagationia!")

        d_activation = self.outputs.array_activation(deriv=True)
        delta = error.elementwise_multiply(d_activation)
        inputs_t = self.inputs.transpose()
        weight_update = inputs_t.array_dot(delta).array_scale(learning_rate)
        self.weights = self.weights.array_subtract(weight_update)
        bias_update = delta.array_scale(learning_rate)
        self.biases = self.biases.array_subtract(bias_update)
        weights_t = self.weights.transpose()
        return delta.array_dot(weights_t)


class BrainNetwork:
    def __init__(self, layers):
        self.layers = layers

    def array_predict(self, inputs):
        for layer in self.layers:
            inputs = layer.array_push(inputs)

        if isinstance(inputs, BrainMatrix):
            return inputs.to_list()

        else:
            return [[float(inputs)]] \
                if not isinstance(inputs, list) \
                else [float(x) for x in inputs]

    def train(self, inputs, expected_output, learning_rate=0.001):
        prediction = inputs

        for layer in self.layers:
            prediction = layer.array_push(prediction)
        error = BrainMatrix(expected_output).array_subtract(prediction)

        for layer in reversed(self.layers):
            error = layer.array_backpropagate(error, learning_rate)
