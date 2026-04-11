"""
Tests unitaires IrisEngine — sans modèle requis.

Usage : python3 tests/test_llm.py
"""

import asyncio
import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.llm import IrisEngine


def test_instanciation():
    engine = IrisEngine("/tmp/fake")
    assert not engine.is_loaded
    assert engine.model_name == "fake"
    print("[OK] instanciation sans crash")


def test_generate_avant_load():
    engine = IrisEngine("/tmp/fake")
    try:
        engine.generate("test")
        assert False, "Doit lever RuntimeError"
    except RuntimeError as e:
        assert "load()" in str(e)
        print("[OK] generate() sans load() → RuntimeError")


def test_stream_avant_load():
    engine = IrisEngine("/tmp/fake")
    try:
        list(engine.stream("test"))
        assert False, "Doit lever RuntimeError"
    except RuntimeError as e:
        assert "load()" in str(e)
        print("[OK] stream() sans load() → RuntimeError")


def test_interface_synchrone():
    assert not inspect.iscoroutinefunction(IrisEngine.generate)
    assert not inspect.iscoroutinefunction(IrisEngine.stream)
    assert inspect.iscoroutinefunction(IrisEngine.load)
    print("[OK] generate/stream sync, load async")


if __name__ == "__main__":
    test_instanciation()
    test_generate_avant_load()
    test_stream_avant_load()
    test_interface_synchrone()
    print("\n[OK] Tous les tests unitaires passés.")
