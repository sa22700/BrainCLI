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
from BrainCLI.BrainCLI_FI.Utils_FI import normalize_text
from BrainCLI.BrainCLI_FI.DataManager_FI import SaveToFile
from BrainCLI.BrainCLI_FI.FuzzySearcher_FI import FuzzySearch
from BrainCLI.BrainCLI_FI.Degug_Log_FI import log_error

class AIEngine:
    def __init__(self, data_path = os.path.join(os.path.dirname(__file__), "braindata.fi.pkl")):
        try:
            self.data_manager = SaveToFile(data_path)
            self.fuzzy_search = FuzzySearch(self)
            self.data = self.load_data()

        except Exception as e:
            print(f"Virhe AIEngine-luokan alustuksessa: {e}")
            self.data = {"questions": [], "answers": []}

    def load_data(self):
        try:
            return self.data_manager.load_pickle()

        except Exception as e:
            print(f"Virhe datan lataamisessa: {e}")
            log_error(f"Virhe datan lataamisessa: {e}")
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
                    else "En löytänyt vastausta."

            best_match = self.fuzzy_search.performfuzzysearch(user_input_norm, normalized_questions)

            if best_match:
                index = normalized_questions.index(best_match)
                return data["answers"][index] \
                    if index < len(data["answers"]) \
                    else "En löytänyt vastausta."

            print("En löytänyt vastausta tähän kysymykseen.")
            user_answer = input("Haluatko lisätä vastauksen? (k/e): ").strip().lower()

            if user_answer == "k":
                new_answer = input("Syötä vastaus: ").strip()

                if new_answer:
                    self.data_manager.save_to_pickle(user_input, new_answer)
                    print("Hienoa, kiitos tästä! Opin taas jotain uutta.")
                    self.data = self.data_manager.load_pickle()

                else:
                    print("Tyhjä vastaus ei tallennettu.")

        except Exception as e:
            print(f"Virhe vastauksen käsittelyssä: {e}")
            log_error(f"Virhe vastauksen käsittelyssä: {e}")
            return "Jokin meni pieleen, en pysty käsittelemään kysymystäsi."
