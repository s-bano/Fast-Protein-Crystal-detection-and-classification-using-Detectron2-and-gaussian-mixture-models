# Dépendances système
echo "Installing system requirements..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt update
    sudo apt install -y tesseract-ocr
elif [[ "$OSTYPE" == "darwin"* ]]; then
    brew install tesseract
fi
echo "System requirements installed !"

# Dépendances Python
echo "Installing python packages required..."
pip install -r requirements.txt
echo "Python packages installed !