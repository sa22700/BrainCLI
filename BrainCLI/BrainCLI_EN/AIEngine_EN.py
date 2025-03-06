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

import os
from BrainCLI.BrainCLI_EN.Utils_EN import normalize_text
from BrainCLI.BrainCLI_EN.DataManager_EN import SaveToFile
from BrainCLI.BrainCLI_EN.FuzzySearcher_EN import FuzzySearch

class AIEngine:
    def __init__(self, data_path=os.path.join(os.path.dirname(__file__), "braindata.en.pkl")):
        try:
            self.data_manager = SaveToFile(data_path)
            self.fuzzy_search = FuzzySearch(self)
            self.data = self.load_data()

        except Exception as e:
            print(f"Error in AIEngine initialization: {e}")
            self.data = {"questions": [], "answers": []}

    def load_data(self):
        try:
            return self.data_manager.load_pickle()

        except Exception as e:
            print(f"Error loading data: {e}")
            return {"questions": [], "answers": []}

    def get_response(self, user_input):
        try:
            user_input_norm = normalize_text(user_input)
            data = self.data_manager.load_pickle()
            normalized_questions = [normalize_text(q)
                for q in data["questions"]]

            if user_input_norm in normalized_questions:
                index = normalized_questions.index(user_input_norm)
                return data["answers"][index] \
                    if index < len(data["answers"]) \
                    else "No response found."

            best_match = self.fuzzy_search.performfuzzysearch(user_input_norm, normalized_questions)

            if best_match:
                index = normalized_questions.index(best_match)
                return data["answers"][index] \
                    if index < len(data["answers"]) \
                    else "No response found."

            print("I did not found a response to your question.")
            user_answer = input("Do you want to add a response? (y/n): ").strip().lower()

            if user_answer == "k":
                new_answer = input("Type your response: ").strip()

                if new_answer:
                    self.data_manager.save_to_pickle(user_input, new_answer)
                    print("Great, thanks! I will learn something new.")
                    self.data = self.data_manager.load_pickle()

                else:
                    print("No response provided.")

        except Exception as e:
            print(f"Error by getting response: {e}")
            return "Something went wrong, i can not handle your question."
