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


import ast
import math
import re
from BrainCLI.BrainCLI_FI.Debug_Log_FI import log_error

def deg(x):    return math.radians(x)
def rad(x):    return x
def fact(x):
    if int(x) != x or x < 0:
        raise ValueError("Faktoriaali määritelty vain nollasta ylöspäin kokonaisluvuille.")
    return math.factorial(int(x))

def _preprocess(expr: str) -> str:
    expr = expr.replace(',', '.')
    expr = expr.replace('^', '**')
    expr = re.sub(r'(\d+(?:\.\d+)?)\s*%', r'(\1/100)', expr)
    expr = expr.replace('π', 'pi')
    expr = expr.strip()
    expr = re.sub(r'(\d+(?:\.\d+)?|\([^()]*\))\s*!!', r'fact(fact(\1))', expr)
    expr = re.sub(r'(?<![A-Za-z_])(\d+(?:\.\d+)?|\([^()]*\))\s*!', r'fact(\1)', expr)
    return expr

_ALLOWED_BINOPS = {
    ast.Add:  lambda a, b: a + b,
    ast.Sub:  lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div:  lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod:  lambda a, b: a % b,
    ast.Pow:  lambda a, b: a ** b,
}

_ALLOWED_UNARY = {
    ast.UAdd: lambda a: +a,
    ast.USub: lambda a: -a,
}

_ALLOWED_FUNCS = {
    **{name: getattr(math, name) for name in [
        "sin", "cos", "tan", "asin", "acos", "atan",
        "sinh", "cosh", "tanh",
        "sqrt", "log", "log10", "exp",
        "ceil", "floor", "fabs"
    ]},

    "deg": deg,
    "rad": rad,
    "fact": fact,
}

_ALLOWED_NAMES_BASE = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}

class SafeEval(ast.NodeVisitor):
    def __init__(self, extra_names=None):
        self.names = dict(_ALLOWED_NAMES_BASE)
        if extra_names:
            self.names.update(extra_names)

    def visit(self, node):
        if isinstance(node, ast.Expression):
            return self.visit(node.body)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise TypeError("Vain numerikonstantit ovat sallittuja.")
        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in _ALLOWED_BINOPS:
                raise TypeError(f"Operaattori ei sallittu: {op_type.__name__}")
            left = self.visit(node.left)
            right = self.visit(node.right)
            if op_type is ast.Pow:
                if not isinstance(right, (int, float)) or abs(right) > 10:
                    raise ValueError("Liian suuri eksponentti.")
            return _ALLOWED_BINOPS[op_type](left, right)
        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in _ALLOWED_UNARY:
                raise TypeError(f"Unaarioperaattori ei sallittu: {op_type.__name__}")
            operand = self.visit(node.operand)
            return _ALLOWED_UNARY[op_type](operand)
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise TypeError("Vain nimettyjä funktioita saa kutsua.")
            fname = node.func.id
            if fname not in _ALLOWED_FUNCS:
                raise NameError(f"Tuntematon funktio: {fname}")
            if node.keywords:
                raise TypeError("Avainsanaparametreja ei sallita.")
            args = [self.visit(a) for a in node.args]
            return _ALLOWED_FUNCS[fname](*args)
        elif isinstance(node, ast.Name):
            if node.id in self.names:
                return self.names[node.id]
            raise NameError(f"Tuntematon nimi: {node.id}")
        elif isinstance(node, ast.Tuple):
            raise TypeError("Tuplet eivät ole sallittuja.")
        else:
            raise TypeError(f"Solmutyyppi ei sallittu: {type(node).__name__}")

TOKEN_PATTERN = re.compile(
    r"^\s*(?:[0-9]+(?:[.,][0-9]+)?|ans|pi|e|π|[+\-*/^()%√]|\s)+\s*$",
    re.IGNORECASE,
)

def is_math_expression(text: str) -> bool:
    try:
        expr = _preprocess(text)
        tree = ast.parse(expr, mode="eval")
        SafeEval(extra_names={"ans": 0.0}).visit(tree)
        return True
    except Exception:
        return False

_LAST_RESULT = 0.0

def calculate_expression(expression: str):
    global _LAST_RESULT
    try:
        expr = _preprocess(expression)
        tree = ast.parse(expr, mode="eval")
        evaluator = SafeEval(extra_names={"ans": _LAST_RESULT})
        result = evaluator.visit(tree)
        _LAST_RESULT = float(result) if isinstance(result, (int, float)) else result
        return result
    except Exception as e:
        log_error(e)
        return f"Virhe: {e}"

def command_calculate(user_input: str):
    result = calculate_expression(user_input)
    return f"Tulos: {result}"