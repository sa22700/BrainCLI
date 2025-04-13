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
from BrainCLI.BrainCLI_EN.Debug_Log_EN import log_error
from BrainCLI.BrainCLI_EN.Calculate_EN import is_math_expression, command_calculate
from BrainCLI.BrainCLI_EN.Randomizer_EN import command_random_fact
from BrainCLI.BrainCLI_EN.AIEngine_EN import AIEngine
from BrainCLI.BrainCLI_EN.ContextList_EN import ContextMemory


class Program:
    def __init__(self, data_file="../Models/braindata.en.pkl"):
        data_path = os.path.join(os.path.dirname(__file__), data_file)
        self.ai_engine = AIEngine(data_path)
        self.context_memory = ContextMemory()
        self.commands = {
            "calculate": command_calculate,
            "fact": command_random_fact,
            "trivia": command_random_fact,
        }

    @staticmethod
    def slow_type(text, delay=0.05):
        for char in text:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(delay)
        print()

    def run(self):
        try:
            self.slow_type("Hi! I'm BrainCLI.\nAsk anything!")
            while True:
                user_input = input("\n> ").strip().lower()
                if not user_input:
                    self.slow_type("You didn't say anything. Try again.")
                    continue
                if user_input in ["exit", "quit", "q"]:
                    self.slow_type("Program shutting down. Thank you for using it!")
                    break
                else:
                    self.handle_question(user_input)

        except KeyboardInterrupt:
            self.slow_type("\nProgram interrupted. Thank you for using it!")

        except Exception as e:
            print(f"Error running program: {e}")
            log_error(f"Error running program: {e}")

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
                self.slow_type("It seems like you have a mathematical expression in your question."
                    "\nDo you want me to calculate it for you? (y/n)")
                confirmation = input("> ").strip().lower()
                if confirmation.startswith("y"):
                    result = command_calculate(question)
                    self.slow_type(result)
                    self.context_memory.add_to_context(question, result)
                    return

            self.slow_type("Analyzing your question...")
            last_answer = self.context_memory.get_last_answer()
            response = self.ai_engine.get_response(question, context=last_answer)
            self.slow_type(response)
            self.context_memory.add_to_context(question, response)
            self.collect_feedback(question)

        except Exception as e:
            print(f"Error handling question: {e}")
            log_error(f"Error handling question: {e}")

    def collect_feedback(self, question):
        satisfaction = input("How satisfied were you with my response? (y/n): ").strip().lower()
        if satisfaction == "y":
            self.slow_type("Thank you for your feedback!")
        elif satisfaction == "n":
            correct_answer = input("Please provide the correct answer: ").strip()
            if correct_answer:
                self.ai_engine.update_knowledge(question, correct_answer)
                self.slow_type("Thank you! I have updated my knowledge.")
            else:
                self.slow_type("Answer not saved, as it was empty.")
        else:
            self.slow_type("Feedback not recorded, as it was not 'y' or 'n'.")

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