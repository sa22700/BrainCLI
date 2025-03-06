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

import asyncio
import os
import sys
import time
from BrainCLI.BrainCLI_EN.AIEngine_EN import AIEngine
from BrainCLI.BrainCLI_EN.DataManager_EN import SaveToFile
from BrainCLI.BrainCLI_EN.MatrixArray_EN import BrainMatrix

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
            print(f"Error while running the program: {e}")

    def handle_question(self, question):
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

async def main():
    app = Program()
    await app.run()


