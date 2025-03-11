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
from BrainCLI.BrainCLI_EN.AIEngine_EN import AIEngine
from BrainCLI.BrainCLI_EN.DataManager_EN import SaveToFile
from BrainCLI.BrainCLI_EN.MatrixArray_EN import BrainMatrix
from BrainCLI.BrainCLI_EN.Debug_Log_EN import log_error
from BrainCLI.BrainCLI_EN.Calculate_EN import calculate_expression, is_math_expression
from BrainCLI.BrainCLI_EN.Randomizer_EN import get_random_fact, load_facts

class Program:
    def __init__(self):
        self.ai_engine = AIEngine()
        self.saver = SaveToFile(os.path.join(os.path.dirname(__file__), "braindata.en.pkl"))

    @staticmethod
    def slow_type(text, delay=0.05):
        for char in text:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(delay)
        print()

    async def run(self):
        try:
            self.slow_type("Hi! I am BrainCLI.\nAsk me whatever you want!")
            while True:
                user_input = input("\n> ").strip().lower()
                if not user_input:
                    self.slow_type("You did not write anything. Try again.")
                    continue

                if user_input in ["exit", "quit", "q"]:
                    self.slow_type("The program is shutting down. Thanks for using!")
                    break

                else:
                    self.handle_question(user_input)

        except KeyboardInterrupt:
            self.slow_type("\nThe program was interrupted. Thanks for using!")

        except Exception as e:
            log_error(e)
            print(f"Error while running the program: {e}")

    def handle_question(self, question):
        if "fact" in question or "trivia" in question:
            facts = load_facts(os.path.join(os.path.dirname(__file__), "braindata.en.pkl"))
            fact = get_random_fact(facts)
            self.slow_type(f"Fact: {fact}")
            return

        if is_math_expression(question):
            self.slow_type("It looks like you have a mathematical expression in your question.\nDo you want me to calculate it for you? (y/n)")
            confirmation = input("> ").strip().lower()
            if confirmation.startswith("y"):
                result = calculate_expression(question)
                self.slow_type(f"Result: {result}")
                return

        self.slow_type("Analyzing your question...")
        try:
            ai_response = self.ai_engine.get_response(question)

            if isinstance(ai_response, BrainMatrix):
                try:
                    if (isinstance(ai_response.data, list)
                            and len(ai_response.data) > 0
                            and isinstance(ai_response.data[0], list)):
                        ai_response = ai_response.data[0][0]

                    else:
                        ai_response = "I did not receive a prediction"

                except Exception as e:
                    ai_response = f"Error while processing BrainMatrix: {e}"
            self.slow_type(str(ai_response))

        except Exception as e:
            print(f"Question processing failed: {e}")
            log_error(e)

async def main():
    app = Program()
    await app.run()


