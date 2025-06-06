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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from BrainCLI.BrainCLI_EN.Main_EN import Program as Program_EN
from BrainCLI.BrainCLI_FI.Main_FI import Program as Program_FI

class Main:

    @staticmethod
    def slow_type(text, delay=0.05):
        for char in text:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(delay)
        print()

    @staticmethod
    def main():
        while True:
            Main.slow_type("Which one BrainCLI you want to drive (EN/FI): ")
            choice = input("> ").strip().lower()

            if choice == "en":
                Program_EN().run()
                break
            elif choice == "fi":
                Program_FI().run()
                break
            else:
                print("Invalid choice. Please try again.")

if __name__=="__main__":
    Main.main()