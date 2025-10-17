# 🚀 Quick Start Guide

## Get Started in 3 Steps

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up the Project
```bash
python setup.py
```

### 3. Run the Application
```bash
python app.py
```

Then visit: **http://127.0.0.1:5000**

## Alternative: Manual Setup

If the automatic setup doesn't work:

### Step 1: Install NLTK Data
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Step 2: Train Models
```bash
python train_models.py
```

### Step 3: Start Web App
```bash
python app.py
```

## Testing the System

Try these sample texts:

**Real News Example:**
```
Scientists at MIT have developed a new solar panel technology that increases efficiency by 40%. The research, published in Nature Energy, shows promising results for renewable energy applications.
```

**Fake News Example:**
```
SHOCKING: Doctors discover this one weird trick that cures all diseases instantly! Big Pharma doesn't want you to know about this amazing secret!
```

## API Usage

```python
import requests

# Make a prediction
response = requests.post('http://127.0.0.1:5000/api/predict', 
                        json={'text': 'Your news text here'})
result = response.json()
print(f"Prediction: {result['prediction']}")
```

## Troubleshooting

- **Port 5000 busy?** Change port in `app.py`
- **NLTK errors?** Run the NLTK download commands
- **Memory issues?** Reduce `max_features` in config

## Need Help?

Check the full `README.md` for detailed documentation!
