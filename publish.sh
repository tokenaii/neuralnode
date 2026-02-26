#!/bin/bash

# PyPI Publishing Script for NeuralNode v1.0.0
# Author: Assem Sabry

echo "🚀 Publishing NeuralNode v1.0.0 to PyPI..."

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: pyproject.toml not found. Run this from the project root."
    exit 1
fi

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info/

# Install build dependencies
echo "📦 Installing build dependencies..."
pip install --upgrade build twine

# Run tests (if they exist)
if [ -d "tests" ]; then
    echo "🧪 Running tests..."
    python -m pytest tests/ -v
    if [ $? -ne 0 ]; then
        echo "❌ Tests failed. Please fix before publishing."
        exit 1
    fi
fi

# Check package structure
echo "🔍 Checking package structure..."
python -c "import neuralnode" 2>/dev/null || {
    echo "❌ Package import failed. Check your package structure."
    exit 1
}

# Build the package
echo "🔨 Building package..."
python -m build

# Check the distribution
echo "📋 Checking distribution files..."
twine check dist/*

# Upload to PyPI
echo "📤 Uploading to PyPI..."
twine upload dist/* \
    --username __token__ \
    --password pypi-AgEIcHlwaS5vcmcCJGQ1ZjQ2NDQyLThjNjctNGNlYi1hOTlhLTYyMGFlZGQ2MzAxOAACKlszLCJiMzI5ODAyMi04MjlhLTRlODItOWQzMC1iNzlmZWM2NjdlZmIiXQAABiDYVt0Dx5n9hI2ZtbQxW1v_fvfgFJd_XwhC3ZT9hp6wfw

# Verify upload
echo "✅ Verifying upload..."
pip install neuralnode==1.0.0

# Test installation
echo "🧪 Testing installation..."
python -c "
import neuralnode
print(f'✅ NeuralNode v{neuralnode.__version__} installed successfully!')
print('🎉 NeuralNode is now live on PyPI!')
"

echo "🌟 NeuralNode v1.0.0 published successfully!"
echo "📖 View at: https://pypi.org/project/neuralnode/1.0.0/"
