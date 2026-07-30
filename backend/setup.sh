echo "Conda env setup..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n rag python==3.13 -y

echo "Conda activating rag..."
conda activate rag

if [[ "$CONDA_PREFIX" == *"/envs/rag" ]]; then
    echo "Activated!"
else
    echo "Activation failed."
    exit 1
fi

echo "Installing uv for faster installation..."
pip install uv

echo "Installing pytorch..."
uv pip install torch torchvision --no-cache-dir --index-url https://download.pytorch.org/whl/cu126

echo "Installing dependency..."
uv pip install --no-cache-dir -e .
uv pip install --no-cache-dir -e ".[dev]"
uv pip install --no-cache-dir -r requirements.txt
uv pip install --no-cache-dir -r requirements-api.txt


