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
import difflib
from BrainCLI.BrainCLI_EN.MatrixArray_EN import BrainNetwork, BrainLayer, BrainMatrix
from BrainCLI.BrainCLI_EN.Vectorizer_EN import BrainVectorizer
from BrainCLI.BrainCLI_EN.Debug_Log_EN import log_error

class BrainPredictor:
    def __init__(self):
        self.vectorizer = BrainVectorizer()
        self.categories = ["Unknown"]
        self.model = BrainNetwork([BrainLayer(input_size=300, output_size=1)])
        self.data = self.load_data()
        # Cache questions and normalized forms
        self.questions = list(self.data.keys())
        self.norm_questions = [q.strip().lower() for q in self.questions]

    @staticmethod
    def load_data():
        try:
            path = os.path.join(os.path.dirname(__file__), "../Models/braindata.en.pkl")
            with open(path, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            log_error("Pickle file not found")
            return {}
        except Exception as e:
            print(f"Error loading data: {e}")
            log_error(e)
            return {}

    def fuzzy_match(self, user_input: str) -> str | None:
        query = user_input.strip().lower()
        if not self.questions:
            return None
        match = difflib.get_close_matches(query, self.norm_questions, n=1, cutoff=0.0)
        if not match:
            return None
        idx = self.norm_questions.index(match[0])
        return self.questions[idx]

    def predict(self, question: str) -> dict:
        try:
            vector = self.vectorizer.vectorize_text(question)
            prediction = self.model.array_predict([vector])
            # Normalize prediction into list-of-lists of floats
            if isinstance(prediction, BrainMatrix):
                prediction = prediction.to_list()
            elif isinstance(prediction, (int, float)):
                prediction = [[float(prediction)]]
            elif isinstance(prediction, list) and all(isinstance(x, (int, float)) for x in prediction):
                prediction = [[float(x)] for x in prediction]
            elif not isinstance(prediction, list):
                raise TypeError(f"Error in array_predict: expected list but got {type(prediction)}: {prediction}")

            best = self.fuzzy_match(question)
            if best:
                response = self.data.get(best, "Could not find a response.")
            else:
                response = "I don't know."

            return {"prediction": prediction, "match": best, "response": response}

        except Exception as e:
            log_error(e)
            return {"error": f"Error in prediction: {e}"}
