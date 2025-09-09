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


from BrainCLI.BrainCLI_FI.BrainMatrix_FI import BrainMatrix
from BrainCLI.BrainCLI_FI.Cuda_Path_FI import clip_matrix

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
