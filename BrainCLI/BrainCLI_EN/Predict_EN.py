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

import pickle
import os
from BrainCLI.BrainCLI_EN.MatrixArray_EN import BrainNetwork, BrainLayer, BrainMatrix
from BrainCLI.BrainCLI_EN.Vectorizer_EN import BrainVectorizer
from BrainCLI.BrainCLI_EN.Debug_Log_EN import log_error

class BrainPredictor:
    def __init__(self):
        self.vectorizer = BrainVectorizer()
        self.categories = ["Unknown"]
        self.model = BrainNetwork([
            BrainLayer(input_size=300, output_size=1)])
        self.data = self.load_data()

    @staticmethod
    def load_data():
        try:
            with open(os.path.join(os.path.dirname(__file__), "../Models/braindata.en.pkl"), "rb") as f:
                return pickle.load(f)

        except FileNotFoundError:
            log_error("Pickle file not found")
            return []

        except Exception as e:
            print(f"Error loading data: {e}")
            log_error(e)
            return []

    def fuzzy_match(self, user_input):
        user_input_norm = user_input.strip().lower()
        min_distance = float("inf")
        best_match = None
        for question in self.data:
            question_norm = question.strip().lower()
            distance = abs(len(user_input_norm) - len(question_norm)) + sum(a != b
                for a, b in zip(user_input_norm, question_norm))
            if distance < min_distance:
                min_distance = distance
                best_match = question
        return best_match

    def predict(self, question):
        try:
            vector = self.vectorizer.vectorize_text(question)
            prediction = self.model.array_predict([vector])
            if isinstance(prediction, BrainMatrix):
                prediction = prediction.to_list()
            elif isinstance(prediction, (int, float)):
                prediction = [[float(prediction)]]
            elif (isinstance(prediction, list)
                  and len(prediction) == 1
                  and isinstance(prediction[0], (int, float))):
                prediction = [[float(x)] for x in prediction]
            elif not isinstance(prediction, list):
                raise TypeError(f"Error array_predict: Expected list but got {type(prediction)}: {prediction}")
            best_match = self.fuzzy_match(question)
            if best_match:
                response = self.data.get(best_match, "Could not find a response.")
            else:
                response = "I don't know."
            return {"prediction": prediction, "match": best_match, "response": response}

        except Exception as e:
            log_error(e)
            return {"error": f"Error in prediction: {e}"}
