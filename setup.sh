#!/bin/bash
set -e

echo "=== Signal-to-Sequence Transformer Setup ==="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python 3.8+ required, found $python_version"
    exit 1
fi

echo "✓ Python $python_version detected"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip > /dev/null

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check CUDA availability
python3 -c "import torch; print(f'✓ PyTorch {torch.__version__} installed')"
python3 -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To use:"
echo "  source venv/bin/activate"
echo "  python signal_sequence_classifier.py"
echo ""
echo "To deactivate:"
echo "  deactivate"
