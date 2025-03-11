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
import sys
import time
from BrainCLI.BrainCLI_FI.AIEngine_FI import AIEngine
from BrainCLI.BrainCLI_FI.DataManager_FI import SaveToFile
from BrainCLI.BrainCLI_FI.MatrixArray_FI import BrainMatrix
from BrainCLI.BrainCLI_FI.Degug_Log_FI import log_error
from BrainCLI.BrainCLI_FI.Calculate_FI import calculate_expression, is_math_expression
from BrainCLI.BrainCLI_FI.Randomizer_FI import get_random_fact, load_facts

class Program:
    def __init__(self):
        self.ai_engine = AIEngine()
        self.saver = SaveToFile(os.path.join(os.path.dirname(__file__), "braindata.fi.pkl"))

    @staticmethod
    def slow_type(text, delay=0.05):
        for char in text:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(delay)
        print()

    async def run(self):
        try:
            self.slow_type("Hei! Minä olen BrainCLI.\nKysy mitä tahansa!")
            while True:
                user_input = input("\n> ").strip().lower()
                if not user_input:
                    self.slow_type("Et syöttänyt mitään. Kokeile uudestaan.")
                    continue

                if user_input in ["lopeta", "poistu", "q"]:
                    self.slow_type("Ohjelma suljetaan. Kiitos käytöstä!")
                    break

                else:
                    self.handle_question(user_input)

        except KeyboardInterrupt:
            self.slow_type("\nOhjelma keskeytettiin. Kiitos käytöstä!")

        except Exception as e:
            print(f"Virhe ohjelman suorittamisessa: {e}")
            log_error(f"Virhe ohjelman suorittamisessa: {e}")

    def handle_question(self, question):
        if "fakta" in question or "trivia" in question:
            facts = load_facts(os.path.join(os.path.dirname(__file__), "braindata.fi.pkl"))
            fact = get_random_fact(facts)
            self.slow_type(f"Fakta: {fact}")
            return

        if is_math_expression(question):
            self.slow_type("Näyttää siltä, että syötteessäsi on matemaattinen laskutoimitus.\nHaluatko, että lasken sen puolestasi? (k/e)")
            confirmation = input("> ").strip().lower()
            if confirmation.startswith("k"):
                result = calculate_expression(question)
                self.slow_type(f"Tulos: {result}")
                return

        self.slow_type("Analysoin kysymystäsi...")
        try:
            ai_response = self.ai_engine.get_response(question)

            if isinstance(ai_response, BrainMatrix):
                try:
                    if (isinstance(ai_response.data, list)
                            and len(ai_response.data) > 0
                            and isinstance(ai_response.data[0], list)):
                        ai_response = ai_response.data[0][0]

                    else:
                        ai_response = "En saanut ennustetta"

                except Exception as e:
                    ai_response = f"Virhe BrainMatrixin käsittelyssä: {e}"
                    log_error(f"Virhe BrainMatrixin käsittelyssä: {e}")
            self.slow_type(str(ai_response))

        except Exception as e:
            print(f"Kysymyksen käsittely epäonnistui: {e}")
            log_error(f"Kysymyksen käsittely epäonnistui: {e}")

async def main():
    app = Program()
    await app.run()


