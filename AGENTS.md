# AGENTS.md - AI Learning Repository Guidelines

## Project Overview
This is a collaborative AI learning repository where multiple contributors document their learning journey through Kaggle courses and ML projects. Each contributor has their own directory.

## Repository Structure
```
ai-learning/
├── LYGreen/          # Main author's work
├── xipian/           # xipian's projects
├── sandip/           # sandip's projects
├── sara/             # sara's projects
└── README.md         # Main documentation
```

## Environment Setup

### Prerequisites
- Python 3.12+
- uv (recommended) or pip

### Installation
```bash
# Using uv (recommended for xipian projects)
cd xipian/intro-to-machine-learning
uv sync

# Using pip (for sandip projects)
cd sandip
pip install -r requirements.txt
```

### Key Dependencies
- pandas >= 3.0.0
- scikit-learn >= 1.8.0
- numpy >= 2.4.2
- jupyter/jupyterlab
- matplotlib, seaborn
- ipykernel

## Running Code

### Run a Python Script
```bash
# From project root
python xipian/intro-to-machine-learning/main.py

# Or from project directory
cd xipian/intro-to-machine-learning
python main.py
```

### Run a Jupyter Notebook
```bash
jupyter notebook intro-to-machine-learning/exp/pandas.ipynb
# or
jupyter lab
```

### Running a Single Test
This is a learning repository with **no formal test framework**. To test functionality:
```bash
# Manually run the script
python <path-to-file>.py

# Or import in Python REPL
python -c "from xipian.intro-to-machine-learning.main import print_multiplication_table; print_multiplication_table()"
```

## Code Style Guidelines

### Imports
- Standard library imports first
- Third-party imports second (pandas, numpy, sklearn, etc.)
- Local imports last
```python
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor

from src.utils import greet  # local imports
```

### Naming Conventions
- **Files**: lowercase with underscores (`main.py`, `data_processing.py`)
- **Functions**: snake_case (`print_multiplication_table`, `calculate_average`)
- **Classes**: PascalCase (`Factory`, `DecisionTreeRegressor`)
- **Variables**: snake_case (`melbourne_data`, `X`, `y`)
- **Constants**: UPPERCASE (if used)

### Formatting
- 4 spaces for indentation (no tabs)
- Maximum line length: 88-100 characters
- Use f-strings for string formatting
```python
print(f"Average score: {calculate_average(scores)}")
```

### Type Hints
Type hints are **optional** in this learning repository. Add when beneficial:
```python
def greet(name: str) -> str:
    return f"Hello, {name}!"
```

### Error Handling
- Use try-except for file operations and external calls
- Keep error handling simple for learning projects
```python
try:
    data = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"File not found: {file_path}")
```

### Comments & Documentation
- Comments can be in English or Chinese (repository is bilingual)
- Add docstrings for functions when complex
- Keep code self-explanatory when possible
```python
# 创建一个决策树模型
melbourne_model = DecisionTreeRegressor(random_state=1)
```

### Classes & OOP
- Use `__init__` for initialization
- Use `self` for instance attributes
- Keep classes focused on single responsibility
```python
class Factory:
    def __init__(self, material, zips, pockets):
        self.material = material
        self.zips = zips
        self.pockets = pockets
```

### Data Processing Patterns
```python
# Load data
data = pd.read_csv('file.csv')

# Clean data
data = data.dropna(axis=0)

# Select features
X = data[['feature1', 'feature2']]
y = data.target

# Train model
model.fit(X, y)
predictions = model.predict(X)
```

## Git Workflow

### Pull with Merge (Default)
```bash
git pull origin main --tags
```

### Handle Merge Conflicts
1. Edit conflicted files to resolve `<<<<<<<`, `=======`, `>>>>>>>`
2. Mark resolved: `git add <file>`
3. Commit: `git commit -m "Resolve conflict"`

### Commit Messages
- Use clear, descriptive messages
- Prefix with action: "Add", "Fix", "Update", "Remove"
```bash
git commit -m "Add pandas data loading example"
```

## Adding New Content

### New Contributor Directory
```bash
mkdir <username>
cd <username>
mkdir <course-name>
```

### New Project Structure
```
<username>/<course-name>/
├── main.py           # Main entry point
├── src/              # Source modules
├── input/            # Data files (add to .gitignore if large)
├── exp/              # Experiments/notebooks
└── README.md         # Project-specific docs
```

### Large Data Files
Add to `.gitignore` or use git-lfs:
```
# .gitignore
*.csv
*.parquet
input/large-*
```

## No Formal Linting/Testing
This is a learning repository. Focus on:
- Working code over perfect style
- Clear examples over comprehensive tests
- Documentation over CI/CD

## Useful Commands
```bash
# Check Python version
python --version

# List installed packages
pip list  # or: uv pip list

# Run a notebook in background
jupyter notebook --no-browser --port=8888

# View git status
git status

# Push changes
git push origin main
```

## Contributor-Specific Notes
- **LYGreen**: Main author, focuses on ML courses
- **xipian**: Uses uv for dependency management
- **sandip**: Uses traditional pip/requirements.txt
- **sara**: Focuses on Python fundamentals
