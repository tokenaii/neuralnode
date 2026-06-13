@echo off
REM PyPI Publishing Script for NeuralNode v1.0.0 (Windows)
REM Author: Assem Sabry

echo 🚀 Publishing NeuralNode v1.0.0 to PyPI...

REM Check if we're in the right directory
if not exist pyproject.toml (
    echo ❌ Error: pyproject.toml not found. Run this from the project root.
    exit /b 1
)

REM Clean previous builds
echo 🧹 Cleaning previous builds...
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build
for /d %%i in (*.egg-info) do rmdir /s /q "%%i"

REM Install build dependencies
echo 📦 Installing build dependencies...
pip install --upgrade build twine

REM Run tests (if they exist)
if exist tests (
    echo 🧪 Running tests...
    python -m pytest tests/ -v
    if errorlevel 1 (
        echo ❌ Tests failed. Please fix before publishing.
        exit /b 1
    )
)

REM Check package structure
echo 🔍 Checking package structure...
python -c "import neuralnode" 2>nul || (
    echo ❌ Package import failed. Check your package structure.
    exit /b 1
)

REM Build the package
echo 🔨 Building package...
python -m build

REM Check the distribution
echo 📋 Checking distribution files...
twine check dist/*

REM Upload to PyPI
echo 📤 Uploading to PyPI...
if "%TWINE_USERNAME%"=="" (
    echo Error: set TWINE_USERNAME and TWINE_PASSWORD before publishing.
    exit /b 1
)
if "%TWINE_PASSWORD%"=="" (
    echo Error: set TWINE_USERNAME and TWINE_PASSWORD before publishing.
    exit /b 1
)
twine upload dist/*

REM Verify upload
echo ✅ Verifying upload...
pip install neuralnode==1.0.0

REM Test installation
echo 🧪 Testing installation...
python -c "import neuralnode; print(f'✅ NeuralNode v{neuralnode.__version__} installed successfully!'); print('🎉 NeuralNode is now live on PyPI!')"

echo 🌟 NeuralNode v1.0.0 published successfully!
echo 📖 View at: https://pypi.org/project/neuralnode/1.0.0/

pause
