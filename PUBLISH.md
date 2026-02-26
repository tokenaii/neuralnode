# PyPI Publishing Commands for NeuralNode v1.0.0

## Prerequisites
1. Make sure all tests pass
2. Update version to 1.0.0 in pyproject.toml
3. Update README.md with latest features
4. Ensure all files are committed to git

## Quick Publish (Linux/Mac)
```bash
# Make script executable
chmod +x publish.sh

# Run the script
./publish.sh
```

## Quick Publish (Windows)
```cmd
# Run the batch file
publish.bat
```

## Manual Publishing Steps

### 1. Clean and Build
```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info/

# Install build tools
pip install --upgrade build twine

# Build the package
python -m build
```

### 2. Check Distribution
```bash
# Verify the package
twine check dist/*
```

### 3. Upload to PyPI
```bash
# Upload with your token
twine upload dist/* \
    --username __token__ \
    --password pypi-AgEIcHlwaS5vcmcCJGQ1ZjQ2NDQyLThjNjctNGNlYi1hOTlhLTYyMGFlZGQ2MzAxOAACKlszLCJiMzI5ODAyMi04MjlhLTRlODItOWQzMC1iNzlmZWM2NjdlZmIiXQAABiDYVt0Dx5n9hI2ZtbQxW1v_fvfgFJd_XwhC3ZT9hp6wfw
```

### 4. Verify Installation
```bash
# Install from PyPI
pip install neuralnode==1.0.0

# Test import
python -c "import neuralnode; print(neuralnode.__version__)"
```

## Test with TestPyPI (Optional)
```bash
# Upload to test server first
twine upload --repository testpypi dist/* \
    --username __token__ \
    --password pypi-AgEIcHlwaS5vcmcCJGQ1ZjQ2NDQyLThjNjctNGNlYi1hOTlhLTYyMGFlZGQ2MzAxOAACKlszLCJiMzI5ODAyMi04MjlhLTRlODItOWQzMC1iNzlmZWM2NjdlZmIiXQAABiDYVt0Dx5n9hI2ZtbQxW1v_fvfgFJd_XwhC3ZT9hp6wfw

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ neuralnode==1.0.0
```

## After Publishing
1. Visit https://pypi.org/project/neuralnode/1.0.0/
2. Check that all files are uploaded correctly
3. Test installation in a fresh environment
4. Update GitHub releases with v1.0.0 tag
5. Announce the release!

## Troubleshooting

### Token Issues
If you get "Invalid token" error:
1. Check your PyPI token is valid
2. Ensure no extra spaces in the token
3. Generate a new token from PyPI account settings

### Build Errors
Common issues:
- Missing dependencies: `pip install build`
- Permission errors: Use `sudo` on Linux
- Python version: Ensure Python 3.8+

### Upload Errors
- File already exists: Increment version number
- Invalid metadata: Check pyproject.toml syntax
- Network issues: Try again or use VPN

## Success!
🎉 NeuralNode v1.0.0 is now live on PyPI!

Link: https://pypi.org/project/neuralnode/1.0.0/
