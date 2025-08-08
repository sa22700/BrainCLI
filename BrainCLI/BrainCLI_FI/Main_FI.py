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


import os
import sys
import time
from BrainCLI.BrainCLI_FI.Debug_Log_FI import log_error
from BrainCLI.BrainCLI_FI.Calculate_FI import is_math_expression, command_calculate
from BrainCLI.BrainCLI_FI.Randomizer_FI import command_random_fact
from BrainCLI.BrainCLI_FI.AIEngine_FI import AIEngine
from BrainCLI.BrainCLI_FI.ContextList_FI import ContextMemory


class Program:
    def __init__(self, data_file="../Models/braindata.fi.pkl"):
        data_path = os.path.join(os.path.dirname(__file__), data_file)
        weights_path = os.path.join(os.path.dirname(data_path), "../Models/weights.fi.pkl")
        self.ai_engine = AIEngine(data_path)
        self.context_memory = ContextMemory()
        self.commands = {
            "laske": command_calculate,
            "fakta": command_random_fact,
            "trivia": command_random_fact,
        }
        if os.path.exists(weights_path):
            try:
                print("Ladataan neuroverkon painot tiedostosta...")
                self.ai_engine.data_manager.load_weights(self.ai_engine.nn, weights_path)
                print("Painot ladattu.")

            except Exception as e:
                print("Painotiedoston lataus epäonnistui (syy: {}) – koulutetaan neuroverkko...".format(e))
                os.remove(weights_path)
                self.ai_engine.train_network(epochs=3, learning_rate=0.00001)
                print("Tallennetaan painot tiedostoon...")
                self.ai_engine.data_manager.save_weights(self.ai_engine.nn, weights_path)
                print("Painot tallennettu.")
        else:
            print("Koulutetaan neuroverkko...")
            self.ai_engine.train_network(epochs=3, learning_rate=0.00001)
            print("Tallennetaan painot tiedostoon...")
            self.ai_engine.data_manager.save_weights(self.ai_engine.nn, weights_path)
            print("Painot tallennettu.")

    @staticmethod
    def slow_type(text, delay=0.05):
        for char in text:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(delay)
        print()

    def run(self):
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
        try:
            for command, function in self.commands.items():
                if question.startswith(command):
                    args = question[len(command):].strip()
                    result = function(args) if args else function()
                    self.slow_type(result)
                    self.context_memory.add_to_context(question, result)
                    return
            if is_math_expression(question):
                self.slow_type("Näyttää siltä, että syötteessäsi on matemaattinen laskutoimitus."
                    "\nHaluatko, että lasken sen puolestasi? (k/e)")
                confirmation = input("> ").strip().lower()
                if confirmation.startswith("k"):
                    result = command_calculate(question)
                    self.slow_type(result)
                    self.context_memory.add_to_context(question, result)
                    return
            self.slow_type("Analysoin kysymystäsi...")
            response = self.ai_engine.get_response(question)
            self.slow_type(response)
            self.context_memory.add_to_context(question, response)
            self.collect_feedback(question)

        except Exception as e:
            print(f"Virhe kysymyksen käsittelyssä: {e}")
            log_error(f"Virhe kysymyksen käsittelyssä: {e}")

    def collect_feedback(self, question):
        try:
            satisfaction = input("Oliko vastaus tyydyttävä? (k/e): ").strip().lower()
            if satisfaction == "k":
                self.slow_type("Kiitos palautteestasi!")
            elif satisfaction == "e":
                correct_answer = input("Anna oikea vastaus: ").strip()
                if correct_answer:
                    self.ai_engine.update_knowledge(question, correct_answer)
                    self.slow_type("Kiitos! Tallensin uuden vastauksen.")
                else:
                    self.slow_type("Ei tallennettu, koska vastaus jäi tyhjäksi.")
            else:
                self.slow_type("Palaute ei ollut kelvollinen, jatketaan normaalisti.")

        except Exception as e:
            print(f"Virhe palautteen käsittelyssä: {e}")
            log_error(f"Virhe palautteen käsittelyssä: {e}")

    @staticmethod
    def is_math_expression(expr):
        try:
            eval(expr, {"__builtins__": None}, {})
            return True
        except:
            return False

if __name__ == "__main__":
    app = Program()
    app.run()
