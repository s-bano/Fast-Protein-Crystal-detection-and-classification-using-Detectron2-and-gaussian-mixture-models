# Dépendances système
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt update
    sudo apt install -y tesseract-ocr
elif [[ "$OSTYPE" == "darwin"* ]]; then
    brew install tesseract
fi

# Dépendances Python
pip install -r requirements.txt