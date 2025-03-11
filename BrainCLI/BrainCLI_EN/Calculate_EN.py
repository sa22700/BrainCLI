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

import re
from BrainCLI.BrainCLI_EN.Debug_Log_EN import log_error

def is_math_expression(text):
    return re.fullmatch(r"[-\d\s+*/().,^]+", text) is not None

def calculate_expression(expression):
    try:
        expression = expression.replace('^', '**')
        expression = expression.replace(',', '.')
        return eval(expression, {"__builtins__": None}, {})
    except Exception as e:
        log_error(e)
        return f"Error: {e}"

def command_calculate(user_input):
    result = calculate_expression(user_input)
    return f"Result: {result}"
