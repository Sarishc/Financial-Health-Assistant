# check_api.py
try:
    import fastapi
    print("✅ FastAPI installed")
except ImportError:
    print("❌ FastAPI not installed")

try:
    import uvicorn
    print("✅ Uvicorn installed")
except ImportError:
    print("❌ Uvicorn not installed")

try:
    import passlib
    import bcrypt
    print(f"✅ Passlib installed: {passlib.__version__}")
    print(f"✅ Bcrypt installed: {bcrypt.__version__}")
except ImportError as e:
    print(f"❌ Error with authentication libraries: {e}")

try:
    import jose
    print("✅ Python-jose installed")
except ImportError:
    print("❌ Python-jose not installed")

# Test importing your main app
try:
    from app.api.main import app
    print("✅ API app imports successfully")
except ImportError as e:
    print(f"❌ Error importing API app: {e}")