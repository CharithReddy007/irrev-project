"""Minimal test runner (works without pytest): python -m tests.run_all"""
import importlib, inspect, pkgutil, sys, traceback
import tests

failed = 0
for m in pkgutil.iter_modules(tests.__path__):
    if not m.name.startswith("test_"):
        continue
    mod = importlib.import_module(f"tests.{m.name}")
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        if name.startswith("test_") and fn.__module__ == mod.__name__:
            try:
                fn(); print(f"PASS {m.name}.{name}")
            except Exception:
                failed += 1; print(f"FAIL {m.name}.{name}"); traceback.print_exc()
sys.exit(1 if failed else 0)
