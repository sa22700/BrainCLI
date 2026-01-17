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


import os
import sys
import time
import ast
from BrainCLI.BrainCLI_EN.Debug_Log_EN import log_error
from BrainCLI.BrainCLI_EN.Calculate_EN import is_math_expression, command_calculate
from BrainCLI.BrainCLI_EN.Randomizer_EN import command_random_fact
from BrainCLI.BrainCLI_EN.AIEngine_EN import AIEngine
from BrainCLI.BrainCLI_EN.ContextList_EN import ContextMemory
from BrainCLI.BrainCLI_EN.Request_EN import fetch_url


class Program:

    def __init__(self, data_file="../Models/braindata.fi.pkl"):
        data_path = os.path.join(os.path.dirname(__file__), data_file)
        weights_path = os.path.join(os.path.dirname(data_path), "../Models/weights.fi.pkl")
        self.ai_engine = AIEngine(data_path)
        self.context_memory = ContextMemory()
        self.commands = {
            "calc": command_calculate,
            "calculate": command_calculate,
            "fact": command_random_fact,
            "trivia": command_random_fact,
        }
        if os.path.exists(weights_path):
            try:
                print("Loading neural network weights from file...")
                self.ai_engine.data_manager.load_weights(self.ai_engine.nn, weights_path)
                print("Weights loaded.")
            except Exception as e:
                print(f"Loading weights file failed (reason: {e}) – training neural network...")
                os.remove(weights_path)
                self.ai_engine.train_network(epochs=1, learning_rate=0.00001)
                print("Saving weights to file...")
                self.ai_engine.data_manager.save_weights(self.ai_engine.nn, weights_path)
                print("Weights saved.")
        else:
            print("Training neural network...")
            self.ai_engine.train_network(epochs=1, learning_rate=0.00001)
            print("Saving weights to file...")
            self.ai_engine.data_manager.save_weights(self.ai_engine.nn, weights_path)
            print("Weights saved.")

    @staticmethod
    def slow_type(text, delay=0.05):
        for char in text:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(delay)
        print()

    def run(self):
        try:
            self.slow_type("Hi! I'm BrainCLI.\nAsk me anything!")
            while True:
                raw_input_ = input("\n> ").strip()
                if not raw_input_:
                    self.slow_type("You didn't type anything. Try again.")
                    continue
                low = raw_input_.lower()
                if low in ["lopeta", "poistu", "exit", "quit", "q"]:
                    self.slow_type("Exiting. Thanks for using BrainCLI!")
                    break
                else:
                    self.handle_question(raw_input_)

        except KeyboardInterrupt:
            self.slow_type("\nProgram interrupted. Thanks for using BrainCLI!")

        except Exception as e:
            print(f"Error running program: {e}")
            log_error(f"Error running program: {e}")

    def handle_question(self, question):
        try:
            low = question.lower()
            if low == "search" or low.startswith("search ") or low == "search" or low.startswith("search "):
                if low.startswith("search"):
                    args = question[len("search"):].strip()
                else:
                    args = question[len("search"):].strip()
                if not args:
                    try:
                        self.slow_type("Search the web: ")
                        args = input().strip()

                    except (EOFError, KeyboardInterrupt):
                        self.slow_type("Cancelled.")
                        return
                if not args or args.lower() in ("cancel", "q", "quit"):
                    self.slow_type("Cancelled.")
                    return
                result = fetch_url(args)
                self.slow_type(result)
                self.context_memory.add_to_context(f"search {args}", result)
                return
            for command, function in self.commands.items():
                if low.startswith(command):
                    args = question[len(command):].strip()
                    result = function(args)
                    self.slow_type(result)
                    self.context_memory.add_to_context(question, result)
                    return
            if is_math_expression(question):
                self.slow_type(
                    "It looks like your input is a math expression.\n"
                    "Do you want me to calculate it? (y/n)"
                )
                confirmation = input("> ").strip().lower()
                if confirmation.startswith(("y", "k")):
                    result = command_calculate(question)
                    self.slow_type(result)
                    self.context_memory.add_to_context(question, result)
                    return
            self.slow_type("Analyzing your question...")
            response = self.ai_engine.get_response(question)
            self.slow_type(response)
            self.context_memory.add_to_context(question, response)
            self.collect_feedback(question)

        except Exception as e:
            print(f"Error handling question: {e}")
            log_error(f"Error handling question: {e}")

    def collect_feedback(self, question):
        try:
            satisfaction = input("Was the answer satisfactory? (y/n): ").strip().lower()
            if satisfaction == "y":
                self.slow_type("Thanks for your feedback!")
            elif satisfaction == "n":
                correct_answer = input("Provide the correct answer: ").strip()
                if correct_answer:
                    self.ai_engine.update_knowledge(question, correct_answer)
                    self.slow_type("Thanks! I saved the new answer.")
                else:
                    self.slow_type("Not saved because the answer was empty.")
            else:
                self.slow_type("Feedback not recognized; continuing.")

        except Exception as e:
            print(f"Error handling feedback: {e}")
            log_error(f"Error handling feedback: {e}")

    @staticmethod
    def is_safe_math_expr(expr: str) -> bool:
        _ALLOWED_NODES = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Constant,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
            ast.UAdd, ast.USub,
        )
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError:
            return False

        for node in ast.walk(tree):
            if not isinstance(node, _ALLOWED_NODES):
                return False
            if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
                return False

        return True

if __name__ == "__main__":
    app = Program()
    app.run()
