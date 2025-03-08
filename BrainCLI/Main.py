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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asyncio
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
        Main.slow_type("Which one BrainCLI you want to drive (EN/FI): ")
        choice = input("> ").strip().lower()

        if choice == "en":
            asyncio.run(Program_EN().run())
        elif choice == "fi":
            asyncio.run(Program_FI().run())
        else:
            print("Invalid choice. Please try again.")

if __name__=="__main__":
    Main.main()