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


from BrainCLI.BrainCLI_EN.BrainMatrix_EN import BrainMatrix
from BrainCLI.BrainCLI_EN.BrainLayer_EN import BrainLayer

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
