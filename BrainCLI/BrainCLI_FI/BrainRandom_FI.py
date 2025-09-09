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