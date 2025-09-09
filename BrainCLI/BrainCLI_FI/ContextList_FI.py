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


class ContextMemory:
    def __init__(self):
        self.listed_history = []

    def add_to_context(self, question, answer):
        self.listed_history.append((question, answer))

    def get_last_question(self):
        return self.listed_history[-1][0] if self.listed_history else None

    def get_last_answer(self):
        return self.listed_history[-1][1] if self.listed_history else None
