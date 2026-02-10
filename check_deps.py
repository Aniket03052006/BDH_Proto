
import sys

dependencies = [
    "torch",
    "numpy",
    "requests",
    "fastapi",
    "uvicorn",
    "websockets"
]

missing = []
for dep in dependencies:
    try:
        __import__(dep)
        print(f"✅ {dep} is installed.")
    except ImportError:
        print(f"❌ {dep} is MISSING.")
        missing.append(dep)

if missing:
    print(f"\nMissing dependencies: {', '.join(missing)}")
    sys.exit(1)
else:
    print("\nAll dependencies are installed and verified!")
    sys.exit(0)
