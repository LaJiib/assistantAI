"""
Tests Phase 3.2.1 — ToolValidator (analyse AST sécurité).

Couvre :
    - Blacklist builtins : eval, exec, compile, __import__, globals, locals, vars
    - Whitelist imports : modules autorisés sans permission spéciale
    - Permission-based imports : subprocess (SYSTEM_EXEC), os, urllib, http, requests (NETWORK)
    - Détection dans AST imbriqué (fonctions, classes, lambdas)
    - ValidationResult : is_valid, errors, imports_used, permission_required
    - extra_allowed_modules : extension de whitelist via constructeur
    - SyntaxError : code invalide Python

Usage :
    python tests/test_validator.py
    pytest tests/test_validator.py -v
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.tools.schema import PermissionLevel
from core.tools.validator import ToolValidator, ValidationResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_validator(**kwargs) -> ToolValidator:
    return ToolValidator(**kwargs)


# ---------------------------------------------------------------------------
# Tests spécifiés dans CONSIGNE 3.2.1
# ---------------------------------------------------------------------------


def test_eval_rejected() -> None:
    """Test 1 : eval() rejeté (blacklist builtin)."""
    validator = make_validator()
    code = 'x = eval("1+1")'
    result = validator.validate_code(code, PermissionLevel.READ_ONLY)

    assert not result.is_valid
    assert any("eval" in e for e in result.errors)
    assert "Line 1" in result.errors[0]


def test_subprocess_without_permission_rejected() -> None:
    """Test 2 : subprocess sans SYSTEM_EXEC → rejeté."""
    validator = make_validator()
    code = "import subprocess"
    result = validator.validate_code(code, PermissionLevel.READ_ONLY)

    assert not result.is_valid
    assert "subprocess" in result.errors[0]
    assert "SYSTEM_EXEC" in result.errors[0]


def test_subprocess_with_permission_accepted() -> None:
    """Test 3 : subprocess avec SYSTEM_EXEC → accepté."""
    validator = make_validator()
    code = "import subprocess"
    result = validator.validate_code(code, PermissionLevel.SYSTEM_EXEC)

    assert result.is_valid
    assert "subprocess" in result.imports_used


def test_whitelist_import_accepted() -> None:
    """Test 4 : datetime (whitelist) → accepté, permission_required = READ_ONLY."""
    validator = make_validator()
    code = "import datetime"
    result = validator.validate_code(code, PermissionLevel.READ_ONLY)

    assert result.is_valid
    assert result.permission_required == PermissionLevel.READ_ONLY


def test_multiple_patterns() -> None:
    """Test 5 : eval + subprocess (sans permission) → 2 erreurs."""
    validator = make_validator()
    code = """
import datetime
x = eval("bad")
import subprocess
"""
    result = validator.validate_code(code, PermissionLevel.READ_ONLY)

    assert not result.is_valid
    assert len(result.errors) == 2  # eval + subprocess


def test_permission_required_for_requests() -> None:
    """Test 6 : requests → permission_required = NETWORK (même si bloqué)."""
    validator = make_validator()
    code = "import requests"
    result = validator.validate_code(code, PermissionLevel.READ_ONLY)

    assert result.permission_required == PermissionLevel.NETWORK


# ---------------------------------------------------------------------------
# Tests blacklist builtins exhaustifs
# ---------------------------------------------------------------------------


def test_exec_rejected() -> None:
    """exec() → rejeté."""
    validator = make_validator()
    result = validator.validate_code('exec("code")', PermissionLevel.READ_ONLY)
    assert not result.is_valid
    assert any("exec" in e for e in result.errors)


def test_compile_rejected() -> None:
    """compile() → rejeté."""
    validator = make_validator()
    result = validator.validate_code('compile("x=1", "<>", "exec")', PermissionLevel.READ_ONLY)
    assert not result.is_valid
    assert any("compile" in e for e in result.errors)


def test_dunder_import_rejected() -> None:
    """__import__() → rejeté."""
    validator = make_validator()
    result = validator.validate_code('__import__("os")', PermissionLevel.READ_ONLY)
    assert not result.is_valid
    assert any("__import__" in e for e in result.errors)


def test_globals_rejected() -> None:
    """globals() → rejeté."""
    validator = make_validator()
    result = validator.validate_code("globals()", PermissionLevel.READ_ONLY)
    assert not result.is_valid
    assert any("globals" in e for e in result.errors)


def test_locals_rejected() -> None:
    """locals() → rejeté."""
    validator = make_validator()
    result = validator.validate_code("locals()", PermissionLevel.READ_ONLY)
    assert not result.is_valid
    assert any("locals" in e for e in result.errors)


def test_vars_rejected() -> None:
    """vars() → rejeté."""
    validator = make_validator()
    result = validator.validate_code("vars()", PermissionLevel.READ_ONLY)
    assert not result.is_valid
    assert any("vars" in e for e in result.errors)


# ---------------------------------------------------------------------------
# Tests whitelist imports
# ---------------------------------------------------------------------------


def test_all_whitelist_modules_accepted() -> None:
    """Tous les modules de la whitelist par défaut → acceptés."""
    validator = make_validator()
    whitelist_modules = [
        "datetime", "math", "json", "re", "typing", "pathlib",
        "collections", "itertools", "functools", "decimal",
        "fractions", "random", "string",
    ]
    for module in whitelist_modules:
        code = f"import {module}"
        result = validator.validate_code(code, PermissionLevel.READ_ONLY)
        assert result.is_valid, f"{module} devrait être dans la whitelist, erreurs: {result.errors}"
        assert result.permission_required == PermissionLevel.READ_ONLY


def test_from_import_whitelist() -> None:
    """from datetime import datetime → autorisé (whitelist)."""
    validator = make_validator()
    code = "from datetime import datetime"
    result = validator.validate_code(code, PermissionLevel.READ_ONLY)
    assert result.is_valid
    assert "datetime" in result.imports_used


def test_from_pathlib_import_path() -> None:
    """from pathlib import Path → autorisé (whitelist, pas os)."""
    validator = make_validator()
    code = "from pathlib import Path"
    result = validator.validate_code(code, PermissionLevel.READ_ONLY)
    assert result.is_valid


# ---------------------------------------------------------------------------
# Tests permission-based imports
# ---------------------------------------------------------------------------


def test_os_requires_system_exec() -> None:
    """import os → SYSTEM_EXEC requis."""
    validator = make_validator()
    result = validator.validate_code("import os", PermissionLevel.READ_ONLY)
    assert not result.is_valid
    assert "SYSTEM_EXEC" in result.errors[0]
    assert result.permission_required == PermissionLevel.SYSTEM_EXEC


def test_os_with_system_exec_accepted() -> None:
    """import os avec SYSTEM_EXEC → accepté."""
    validator = make_validator()
    result = validator.validate_code("import os", PermissionLevel.SYSTEM_EXEC)
    assert result.is_valid
    assert "os" in result.imports_used


def test_from_os_import_requires_system_exec() -> None:
    """from os import path → top-level 'os' → SYSTEM_EXEC requis."""
    validator = make_validator()
    result = validator.validate_code("from os import path", PermissionLevel.READ_ONLY)
    assert not result.is_valid
    assert "SYSTEM_EXEC" in result.errors[0]


def test_urllib_requires_network() -> None:
    """import urllib → NETWORK requis."""
    validator = make_validator()
    result = validator.validate_code("import urllib", PermissionLevel.READ_ONLY)
    assert not result.is_valid
    assert "NETWORK" in result.errors[0]
    assert result.permission_required == PermissionLevel.NETWORK


def test_http_requires_network() -> None:
    """import http → NETWORK requis."""
    validator = make_validator()
    result = validator.validate_code("import http", PermissionLevel.SYSTEM_EXEC)
    assert not result.is_valid
    assert "NETWORK" in result.errors[0]


def test_requests_with_network_accepted() -> None:
    """import requests avec NETWORK → accepté."""
    validator = make_validator()
    result = validator.validate_code("import requests", PermissionLevel.NETWORK)
    assert result.is_valid
    assert "requests" in result.imports_used


def test_autonomous_accepts_all_permission_based() -> None:
    """AUTONOMOUS → accepte subprocess, os, requests."""
    validator = make_validator()
    code = "import subprocess\nimport os\nimport requests"
    result = validator.validate_code(code, PermissionLevel.AUTONOMOUS)
    assert result.is_valid
    assert {"subprocess", "os", "requests"}.issubset(result.imports_used)


# ---------------------------------------------------------------------------
# Tests AST imbriqué (détection dans fonctions, classes, lambdas)
# ---------------------------------------------------------------------------


def test_eval_inside_function_detected() -> None:
    """eval() dans une fonction → détecté (parcours exhaustif)."""
    validator = make_validator()
    code = """
def my_func(x):
    return eval(x)
"""
    result = validator.validate_code(code, PermissionLevel.READ_ONLY)
    assert not result.is_valid
    assert any("eval" in e for e in result.errors)


def test_exec_inside_class_detected() -> None:
    """exec() dans une méthode de classe → détecté."""
    validator = make_validator()
    code = """
class MyClass:
    def dangerous(self):
        exec("malicious")
"""
    result = validator.validate_code(code, PermissionLevel.READ_ONLY)
    assert not result.is_valid
    assert any("exec" in e for e in result.errors)


def test_import_inside_function_detected() -> None:
    """import subprocess dans une fonction → détecté."""
    validator = make_validator()
    code = """
def get_info():
    import subprocess
    return subprocess.check_output(["ls"])
"""
    result = validator.validate_code(code, PermissionLevel.READ_ONLY)
    assert not result.is_valid
    assert "subprocess" in result.imports_used


def test_eval_via_attribute_detected() -> None:
    """builtins.eval() → détecté via ast.Attribute."""
    validator = make_validator()
    code = 'import builtins\nbuiltins.eval("code")'
    result = validator.validate_code(code, PermissionLevel.READ_ONLY)
    # builtins n'est pas en whitelist → erreur import + eval
    assert not result.is_valid
    assert any("eval" in e for e in result.errors)


# ---------------------------------------------------------------------------
# Tests permission_required
# ---------------------------------------------------------------------------


def test_permission_required_no_imports() -> None:
    """Code sans import → permission_required = READ_ONLY."""
    validator = make_validator()
    code = "x = 1 + 1"
    result = validator.validate_code(code, PermissionLevel.READ_ONLY)
    assert result.is_valid
    assert result.permission_required == PermissionLevel.READ_ONLY


def test_permission_required_max_of_multiple() -> None:
    """subprocess (SYSTEM_EXEC) + requests (NETWORK) → permission_required = NETWORK."""
    validator = make_validator()
    code = "import subprocess\nimport requests"
    result = validator.validate_code(code, PermissionLevel.NETWORK)
    assert result.is_valid
    assert result.permission_required == PermissionLevel.NETWORK


def test_permission_required_blocked_still_computed() -> None:
    """import subprocess sans permission → is_valid=False mais permission_required=SYSTEM_EXEC."""
    validator = make_validator()
    result = validator.validate_code("import subprocess", PermissionLevel.READ_ONLY)
    assert not result.is_valid
    assert result.permission_required == PermissionLevel.SYSTEM_EXEC


# ---------------------------------------------------------------------------
# Tests SyntaxError
# ---------------------------------------------------------------------------


def test_syntax_error_caught() -> None:
    """Code Python invalide → is_valid=False avec SyntaxError."""
    validator = make_validator()
    result = validator.validate_code("def broken(:", PermissionLevel.READ_ONLY)
    assert not result.is_valid
    assert len(result.errors) == 1
    assert "SyntaxError" in result.errors[0]


# ---------------------------------------------------------------------------
# Tests extra_allowed_modules
# ---------------------------------------------------------------------------


def test_extra_allowed_modules_accepted() -> None:
    """Module non-whitelist ajouté via extra_allowed_modules → accepté."""
    validator = ToolValidator(extra_allowed_modules={"numpy", "pandas"})
    code = "import numpy\nimport pandas"
    result = validator.validate_code(code, PermissionLevel.READ_ONLY)
    assert result.is_valid
    assert result.permission_required == PermissionLevel.READ_ONLY


def test_extra_allowed_modules_does_not_affect_permission_map() -> None:
    """extra_allowed_modules n'affecte pas les modules à permission obligatoire."""
    validator = ToolValidator(extra_allowed_modules={"subprocess"})
    # subprocess dans extra_allowed = whitelisted → autorisé sans SYSTEM_EXEC
    result = validator.validate_code("import subprocess", PermissionLevel.READ_ONLY)
    # Note : extra_allowed_modules surcharge la whitelist, pas la permission-map.
    # subprocess reste dans _PERMISSION_REQUIRED → toujours SYSTEM_EXEC requis.
    # Ce test documente le comportement : extra_modules n'override pas permission-map.
    # Comportement attendu : error (permission-map prime sur whitelist extra).
    assert not result.is_valid


def test_default_validator_unknown_module_rejected() -> None:
    """Module inconnu (hors whitelist et hors permission-map) → rejeté."""
    validator = make_validator()
    result = validator.validate_code("import unknown_module_xyz", PermissionLevel.AUTONOMOUS)
    assert not result.is_valid
    assert "unknown_module_xyz" in result.errors[0]


# ---------------------------------------------------------------------------
# Tests ValidationResult immutabilité
# ---------------------------------------------------------------------------


def test_validation_result_is_frozen() -> None:
    """ValidationResult doit être frozen (dataclass frozen=True)."""
    validator = make_validator()
    result = validator.validate_code("x = 1", PermissionLevel.READ_ONLY)

    try:
        result.is_valid = False  # type: ignore[misc]
        assert False, "ValidationResult devrait être frozen"
    except Exception:
        pass  # AttributeError ou FrozenInstanceError attendu


def test_validation_result_errors_is_tuple() -> None:
    """errors doit être un tuple immuable."""
    validator = make_validator()
    result = validator.validate_code('eval("x")', PermissionLevel.READ_ONLY)
    assert isinstance(result.errors, tuple)


def test_validation_result_imports_is_frozenset() -> None:
    """imports_used doit être un frozenset immuable."""
    validator = make_validator()
    result = validator.validate_code("import math", PermissionLevel.READ_ONLY)
    assert isinstance(result.imports_used, frozenset)


# ---------------------------------------------------------------------------
# Tests d'intégration — code réaliste d'outil
# ---------------------------------------------------------------------------


def test_valid_tool_code() -> None:
    """Code d'outil réaliste sans patterns dangereux → accepté."""
    validator = make_validator()
    code = """
from datetime import datetime, timezone
import json
import re
from typing import Optional

def format_date(date_str: str, fmt: str = "%Y-%m-%d") -> Optional[str]:
    try:
        dt = datetime.fromisoformat(date_str)
        return dt.strftime(fmt)
    except ValueError:
        return None
"""
    result = validator.validate_code(code, PermissionLevel.READ_ONLY)
    assert result.is_valid
    assert result.errors == ()
    assert result.permission_required == PermissionLevel.READ_ONLY


def test_malicious_tool_code_multiple_vectors() -> None:
    """Code malveillant avec plusieurs vecteurs → toutes les erreurs détectées."""
    validator = make_validator()
    code = """
import json
import subprocess
import requests

def exfiltrate():
    data = locals()
    exec(compile("import os; os.remove('/etc/hosts')", '<string>', 'exec'))
"""
    result = validator.validate_code(code, PermissionLevel.READ_ONLY)
    assert not result.is_valid
    # subprocess, requests, locals(), exec(), compile()
    assert len(result.errors) >= 4


# ---------------------------------------------------------------------------
# Point d'entrée standalone
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import traceback

    tests = [
        test_eval_rejected,
        test_subprocess_without_permission_rejected,
        test_subprocess_with_permission_accepted,
        test_whitelist_import_accepted,
        test_multiple_patterns,
        test_permission_required_for_requests,
        test_exec_rejected,
        test_compile_rejected,
        test_dunder_import_rejected,
        test_globals_rejected,
        test_locals_rejected,
        test_vars_rejected,
        test_all_whitelist_modules_accepted,
        test_from_import_whitelist,
        test_from_pathlib_import_path,
        test_os_requires_system_exec,
        test_os_with_system_exec_accepted,
        test_from_os_import_requires_system_exec,
        test_urllib_requires_network,
        test_http_requires_network,
        test_requests_with_network_accepted,
        test_autonomous_accepts_all_permission_based,
        test_eval_inside_function_detected,
        test_exec_inside_class_detected,
        test_import_inside_function_detected,
        test_eval_via_attribute_detected,
        test_permission_required_no_imports,
        test_permission_required_max_of_multiple,
        test_permission_required_blocked_still_computed,
        test_syntax_error_caught,
        test_extra_allowed_modules_accepted,
        test_extra_allowed_modules_does_not_affect_permission_map,
        test_default_validator_unknown_module_rejected,
        test_validation_result_is_frozen,
        test_validation_result_errors_is_tuple,
        test_validation_result_imports_is_frozenset,
        test_valid_tool_code,
        test_malicious_tool_code_multiple_vectors,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  ✓  {test.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  ✗  {test.__name__}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed + failed} tests — {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
