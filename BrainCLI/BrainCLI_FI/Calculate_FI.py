import re
from BrainCLI.BrainCLI_FI.Degug_Log_FI import log_error

def is_math_expression(text):
    return re.fullmatch(r"[-\d\s+*/().,^]+", text) is not None

def calculate_expression(expression):
    try:
        expression = expression.replace('^', '**')
        expression = expression.replace(',', '.')
        return eval(expression, {"__builtins__": None}, {})
    except Exception as e:
        log_error(e)
        return f"Virhe: {e}"
