"""
Test if pkg_resources is available and working
"""
import sys

print("=" * 70)
print("PKG_RESOURCES DIAGNOSTIC")
print("=" * 70)

# Test 1: Check Python version
print(f"\n1. Python version: {sys.version}")
print(f"   Executable: {sys.executable}")

# Test 2: Check sys.path
print(f"\n2. Python path (first 3 entries):")
for i, p in enumerate(sys.path[:3]):
    print(f"   {i}: {p}")

# Test 3: Try to import pkg_resources
print("\n3. Testing pkg_resources import...")
try:
    import pkg_resources
    print("   ✅ pkg_resources imported successfully!")
    print(f"   Location: {pkg_resources.__file__}")
    print(f"   Version: {pkg_resources.__version__}")
except ImportError as e:
    print(f"   ❌ Failed to import pkg_resources: {e}")
    print("\n   Checking if setuptools is installed...")
    try:
        import setuptools
        print(f"   ✅ setuptools is installed: {setuptools.__version__}")
        print(f"   Location: {setuptools.__file__}")
        print("\n   ⚠️  setuptools is installed but pkg_resources is missing!")
        print("   This means setuptools installation is broken.")
    except ImportError:
        print("   ❌ setuptools is NOT installed")
        print("   Run: pip install setuptools")
    sys.exit(1)

# Test 4: Try to import tensorboard
print("\n4. Testing TensorBoard import...")
try:
    import tensorboard
    print("   ✅ tensorboard imported successfully!")
    print(f"   Version: {tensorboard.__version__}")
except ImportError as e:
    print(f"   ❌ Failed to import tensorboard: {e}")
    sys.exit(1)

# Test 5: Try to import tensorboard.default (the failing import)
print("\n5. Testing tensorboard.default import...")
try:
    from tensorboard import default
    print("   ✅ tensorboard.default imported successfully!")
except ImportError as e:
    print(f"   ❌ Failed to import tensorboard.default: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED")
print("=" * 70)
print("\nTensorBoard should work now. Try: start_tensorboard.bat")
