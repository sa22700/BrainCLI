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


import traceback
import os

LOG_FILE = os.path.join(os.path.dirname(__file__), "../Errors_log/debug_FI.log")

def log_error(error_message):
    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(f"Virhe: {error_message}\n")
        log.write(traceback.format_exc())
        log.write("\n" + "="*40 + "\n")
