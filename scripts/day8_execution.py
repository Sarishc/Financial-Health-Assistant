# scripts/day8_execution.py
"""
Day 8 execution script - API Development
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path

# Add parent directory to path for importing app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_day8_tasks():
    """Execute Day 8 implementation tasks: API Development"""
    
    print("=" * 80)
    print("Day 8: API Development")
    print("=" * 80)
    
    # 1. Create necessary directories
    print("\nCreating API directories...")
    os.makedirs('app/api', exist_ok=True)
    os.makedirs('app/api/routes', exist_ok=True)
    os.makedirs('app/api/schemas', exist_ok=True)
    os.makedirs('app/api/auth', exist_ok=True)
    os.makedirs('app/api/docs', exist_ok=True)
    
    # 2. Create __init__.py files
    for directory in [
        'app/api',
        'app/api/routes',
        'app/api/schemas',
        'app/api/auth'
    ]:
        init_file = os.path.join(directory, '__init__.py')
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write("# Package initialization\n")
    
    # 3. Check if we've already copied the API files
    api_files_exist = all([
        os.path.exists('app/api/main.py'),
        os.path.exists('app/api/auth/auth.py'),
        os.path.exists('app/api/routes/transactions.py'),
        os.path.exists('app/api/routes/recommendations.py'),
        os.path.exists('app/api/routes/categories.py'),
        os.path.exists('app/api/routes/forecasts.py'),
    ])
    
    if api_files_exist:
        print("API files already exist. Skipping file creation.")
    else:
        # 4. Copy API files from the script directory
        script_dir = Path(__file__).parent
        
        # List of files to copy or check if they exist in current directory
        api_files = [
            'app/api/main.py',
            'app/api/auth/auth.py',
            'app/api/schemas/auth.py',
            'app/api/schemas/messages.py',
            'app/api/schemas/transaction.py',
            'app/api/schemas/recommendation.py',
            'app/api/schemas/category.py',
            'app/api/schemas/forecast.py',
            'app/api/routes/transactions.py',
            'app/api/routes/recommendations.py',
            'app/api/routes/categories.py',
            'app/api/routes/forecasts.py',
            'app/api/routes/users.py',
            'app/api/docs/api_docs.md',
        ]
        
        for file_path in api_files:
            if not os.path.exists(file_path):
                print(f"Error: {file_path} not found. Make sure you have created all the API files.")
                print("You may need to manually create the API files based on the completed examples.")
    
    # 5. Update config.py to include API settings
    config_path = 'app/utils/config.py'
    if os.path.exists(config_path):
        print("\nUpdating configuration settings...")
        
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        # Check if API settings already exist
        if 'API_V1_STR' not in config_content:
            # Add API settings
            api_settings = """
# API settings
API_V1_STR: str = "/api/v1"
SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-for-development-only")
ALGORITHM: str = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

# CORS settings
BACKEND_CORS_ORIGINS: list = ["http://localhost:3000", "http://localhost:8000"]
"""
            # Append to existing settings
            updated_content = config_content.replace("class Settings(BaseSettings):", f"class Settings(BaseSettings):{api_settings}")
            
            with open(config_path, 'w') as f:
                f.write(updated_content)
    else:
        print(f"Warning: {config_path} not found. Please create the config file manually.")
    
    # 6. Update main app file (app/main.py or create it)
    main_app_path = 'app/main.py'
    if not os.path.exists(main_app_path):
        print("\nCreating main app file...")
        
        main_app_content = """\"\"\"
Main application for Financial Health Assistant
\"\"\"
import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

# Import API
from app.api.main import app as api_app

# Create FastAPI app
app = FastAPI(
    title="Financial Health Assistant",
    description="A tool for analyzing financial transactions and providing recommendations",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the API
app.mount("/api", api_app)

# Root endpoint
@app.get("/")
async def root():
    \"\"\"Redirect to API documentation\"\"\"
    return RedirectResponse(url="/api/docs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""
        with open(main_app_path, 'w') as f:
            f.write(main_app_content)
        
        print(f"Created {main_app_path}")
    
    # 7. Install required dependencies
    print("\nChecking and installing required dependencies...")
    required_packages = [
        'fastapi>=0.68.0',
        'uvicorn[standard]>=0.15.0',
        'pydantic>=1.8.0',
        'python-jose[cryptography]>=3.3.0',
        'passlib[bcrypt]>=1.7.4',
        'python-multipart>=0.0.5',
        'email-validator>=1.1.3'
    ]
    
    # Check if packages are already installed
    import importlib.util
    missing_packages = []
    for package in required_packages:
        package_name = package.split('>=')[0]
        if importlib.util.find_spec(package_name) is None:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Installing missing packages: {', '.join(missing_packages)}")
        subprocess.run([sys.executable, "-m", "pip", "install"] + missing_packages)
    else:
        print("All required packages are already installed.")
    
    # 8. Provide instructions for running the API
    print("\n" + "=" * 80)
    print("API Development Setup Complete!")
    print("=" * 80)
    print("\nTo run the API server, execute:")
    print("  uvicorn app.api.main:app --reload")
    print("\nAPI documentation will be available at:")
    print("  http://localhost:8000/docs")
    print("  http://localhost:8000/redoc")
    
    return True

if __name__ == "__main__":
    success = run_day8_tasks()
    
    if success:
        print("\nDay 8 tasks completed successfully!")
    else:
        print("\nDay 8 tasks completed with errors.")