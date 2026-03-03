# Sandip's AI Learning Guide — Detailed Course-Based Levels

I am Sandip. This guide provides granular, step-by-step tasks for each learning level.

**System**: Fedora Linux  
**Progress Tracking**: Mark `[ ]` → `[x]` as you complete each sub-task.

> **Note for Fedora Users**: This guide uses `dnf` (Fedora's package manager) instead of `apt`. All Python/pip commands work the same across Linux distributions.

---

## 🐧 Fedora Quick Reference

### Package Management (Fedora)

```bash
# Update system
sudo dnf update -y

# Install packages
sudo dnf install <package-name> -y

# Search for packages
sudo dnf search <keyword>

# Remove packages
sudo dnf remove <package-name>
```

### Common Packages for AI/ML on Fedora

```bash
# Python essentials
sudo dnf install python3 python3-pip python3-devel -y

# Development tools (optional, useful for some packages)
sudo dnf groupinstall "Development Tools" -y

# SQLite
sudo dnf install sqlite -y

# Git (if not installed)
sudo dnf install git -y
```

### Python Virtual Environment

```bash
# Create venv (python3-venv is built-in on Fedora)
python3 -m venv .venv

# Activate
source .venv/bin/activate

# Deactivate
deactivate
```

---

## 🔧 Immediate Setup (do once before starting)

### Step 1: System Update (Fedora)

- [x] Open terminal
- [x] Run `sudo dnf update -y`
- [x] Verify completion (no errors)

### Step 2: Install Python (Fedora)

- [x] Run `sudo dnf install python3 python3-pip -y`
- [x] Verify: Run `python3 --version` (should show 3.x.x)
- [x] Verify: Run `pip3 --version`
- [x] Take note of Python version in a text file
- [x] Note: Fedora includes python3-venv by default

### Step 3: Create Virtual Environment

- [x] Navigate to your project directory: `cd ~/projects/ai-learning`
- [x] Create venv: `python3 -m venv .venv`
- [x] Activate it: `source .venv/bin/activate`
- [x] Verify prompt shows `(.venv)` prefix
- [x] Create a file `setup_notes.md` documenting this step

### Step 4: Install Essential Packages

- [x] Run `pip install --upgrade pip`
- [x] Install JupyterLab: `pip install jupyterlab`
- [x] Install data packages: `pip install pandas numpy`
- [x] Install ML packages: `pip install scikit-learn`
- [x] Install visualization: `pip install matplotlib seaborn`
- [x] Verify installations: `pip list` (check all packages appear)
- [x] Save package list: `pip freeze > requirements.txt`

### Step 5: Setup Project Structure

- [x] Create `mkdir -p projects/figures`
- [x] Create `mkdir -p projects/data`
- [x] Create `mkdir -p projects/scripts`
- [x] Create `mkdir -p projects/notebooks`
- [x] List structure: `tree projects/` or `ls -R projects/`

---

## 📘 Level 1 — Intro to Programming

**Goal**: Get comfortable with terminal, Python basics, and file operations.

### Task 1.1: Verify Python Installation

- [x] Open terminal
- [x] Run `python --version` or `python3 --version`
- [x] Write down the version number
- [x] Run `pip --version`
- [x] Screenshot or note both versions

### Task 1.2: First Python Script

- [x] Create file: `touch hello.py`
- [x] Open in editor: `nano hello.py` or use VS Code
- [x] Write code:

  ```python
  print("Hello, I am Sandip!")
  print("Starting my AI journey")
  ```

- [x] Save and exit
- [x] Run: `python hello.py`
- [x] Verify output appears correctly

### Task 1.3: Variables and Data Types

- [x] Create `basics.py`
- [x] Write code for variables:

  ```python
  name = "Sandip"
  age = 25
  is_learning = True
  print(f"My name is {name}, age {age}")
  ```

- [x] Add a list: `skills = ["Python", "ML", "Data"]`
- [x] Print the list: `print(skills)`
- [x] Run script and verify output

### Task 1.4: Setup Project Directory

- [x] Create directory: `mkdir projects`
- [x] Navigate: `cd projects`
- [x] Create README: `touch README.md`
- [x] Open and write:

  ```markdown
  # My AI Learning Projects
  Author: Sandip
  Started: [today's date]
  ```

- [x] Save file
- [x] Verify: `cat README.md`

### Task 1.5: Basic Terminal Navigation

- [x] Practice `pwd` (print working directory)
- [x] Practice `ls` (list files)
- [x] Practice `cd ..` (go up one directory)
- [x] Practice `cd projects` (enter directory)
- [x] Create a cheat sheet file with these commands

---

## 🐍 Level 2 — Python Core Language

**Goal**: Write functional Python programs with loops, functions, and file I/O.

### Task 2.1: Working with Lists

- [x] Create `list_practice.py`
- [x] Create a list: `numbers = [1, 2, 3, 4, 5]`
- [x] Write a for loop to print each number
- [x] Add: Calculate sum using `sum(numbers)`
- [x] Add: Find max using `max(numbers)`
- [x] Run and verify output

### Task 2.2: Working with Dictionaries

- [x] Create `dict_practice.py`
- [x] Create dictionary:

  ```python
  student = {
      "name": "Sandip",
      "courses": ["ML", "Python", "Data Science"],
      "level": 2
  }
  ```

- [x] Print each key-value pair
- [x] Add a new key: `student["completed"] = []`
- [x] Run script

### Task 2.3: Writing Functions

- [x] Create `utils.py`
- [x] Write a function:

  ```python
  def greet(name):
      return f"Hello, {name}!"
  
  def calculate_average(numbers):
      return sum(numbers) / len(numbers)
  ```

- [x] Test in the same file:

  ```python
  if __name__ == "__main__":
      print(greet("Sandip"))
      print(calculate_average([1, 2, 3, 4, 5]))
  ```

- [x] Run: `python utils.py`

### Task 2.4: Importing Functions

- [x] Create `main.py` in same directory as `utils.py`
- [x] Write:

  ```python
  from utils import greet, calculate_average
  
  print(greet("World"))
  scores = [85, 90, 78, 92]
  print(f"Average score: {calculate_average(scores)}")
  ```

- [x] Run: `python main.py`
- [x] Verify functions imported correctly

### Task 2.5: File Operations

- [x] Create `file_practice.py`
- [x] Write to a file:

  ```python
  with open("test.txt", "w") as f:
      f.write("This is my first file\n")
      f.write("Learning Python file I/O\n")
  ```

- [x] Read from file:

  ```python
  with open("test.txt", "r") as f:
      content = f.read()
      print(content)
  ```

- [x] Run script and verify file creation

### Task 2.6: Introduction to Pandas

- [x] Create `pandas_intro.py`
- [x] Write:

  ```python
  import pandas as pd
  
  data = {
      'name': ['Alice', 'Bob', 'Charlie'],
      'age': [25, 30, 35],
      'city': ['NY', 'LA', 'Chicago']
  }
  df = pd.DataFrame(data)
  print(df)
  df.to_csv('people.csv', index=False)
  print("\nSaved to people.csv")
  ```

- [x] Run script
- [x] Verify CSV file created
- [x] Open CSV in text editor to inspect

### Task 2.7: Reading CSV with Pandas

- [x] Create `read_csv.py`
- [x] Write:

  ```python
  import pandas as pd
  df = pd.read_csv('people.csv')
  print("First few rows:")
  print(df.head())
  print("\nDataFrame info:")
  print(df.info())
  ```

- [x] Run and observe output

---

## 🤖 Level 3 — Intro to Machine Learning

**Goal**: Understand ML concepts and run your first model.

### Task 3.1: Read Course Materials

- [ ] Navigate to `intro-to-machine-learning/`
- [ ] Open `README.md`
- [ ] Read the course structure overview
- [ ] Take notes on key concepts
- [ ] List 3 things you want to learn

### Task 3.2: Study Decision Trees

- [ ] Open `intro-to-machine-learning/LYGreen/machine-learning-2.md`
- [ ] Read about Decision Trees
- [ ] Draw a simple decision tree on paper (e.g., for predicting house prices)
- [ ] Note down: What is a decision tree? (write 2-3 sentences)

### Task 3.3: Setup ML Environment

- [ ] Verify scikit-learn installed: `pip show scikit-learn`
- [ ] If not: `pip install scikit-learn`
- [ ] Test import:

  ```python
  python -c "import sklearn; print(sklearn.__version__)"
  ```

### Task 3.4: Create Sample Dataset

- [ ] Create `create_sample_data.py`
- [ ] Write:

  ```python
  import pandas as pd
  
  data = {
      'rooms': [3, 4, 2, 5, 3],
      'bathrooms': [2, 3, 1, 3, 2],
      'landsize': [500, 600, 300, 800, 450],
      'price': [400000, 550000, 300000, 700000, 420000]
  }
  df = pd.DataFrame(data)
  df.to_csv('projects/data/sample_houses.csv', index=False)
  print("Sample data created!")
  print(df)
  ```

- [ ] Run script
- [ ] Verify file created in `projects/data/`

### Task 3.5: First Decision Tree Model

- [ ] Create `projects/scripts/run_dt.py`
- [ ] Write:

  ```python
  import pandas as pd
  from sklearn.tree import DecisionTreeRegressor
  from sklearn.metrics import mean_absolute_error
  
  # Load data
  df = pd.read_csv('../data/sample_houses.csv')
  print("Data loaded:")
  print(df.head())
  
  # Separate features and target
  X = df[['rooms', 'bathrooms', 'landsize']]
  y = df['price']
  
  # Create and train model
  model = DecisionTreeRegressor(random_state=1)
  model.fit(X, y)
  
  # Make predictions
  predictions = model.predict(X)
  
  # Calculate error
  mae = mean_absolute_error(y, predictions)
  print(f"\nMean Absolute Error: ${mae:,.2f}")
  print("\nPredictions vs Actual:")
  for i, (pred, actual) in enumerate(zip(predictions, y)):
      print(f"  House {i+1}: Predicted ${pred:,.0f}, Actual ${actual:,.0f}")
  ```

- [ ] Run: `cd projects/scripts && python run_dt.py`
- [ ] Verify output shows predictions and MAE

### Task 3.6: Document First Model

- [ ] Create `projects/ml_notes.md`
- [ ] Write:
  - Date and level completed
  - Model type used (Decision Tree)
  - Dataset description
  - MAE value obtained
  - One thing you learned
- [ ] Save file

---

## 📊 Level 4 — Pandas (Data Wrangling & EDA)

**Goal**: Master data loading, exploration, and basic cleaning.

### Task 4.1: Setup Jupyter Environment

- [ ] Navigate to project root
- [ ] Activate venv: `source .venv/bin/activate`
- [ ] Start Jupyter: `jupyter lab`
- [ ] Browser should open automatically
- [ ] Navigate to `xipian/exp/` folder

### Task 4.2: Open Melbourne Dataset Notebook

- [ ] In JupyterLab, open `melb_data.ipynb`
- [ ] If not exists, create new notebook
- [ ] Name it `melb_data_exploration.ipynb`
- [ ] Save in `projects/notebooks/`

### Task 4.3: Load Dataset

- [ ] In first cell, write:

  ```python
  import pandas as pd
  import numpy as np
  
  # Load data
  df = pd.read_csv('../exp/melb_data.csv')
  print("Dataset loaded successfully!")
  print(f"Shape: {df.shape}")
  ```

- [ ] Run cell (Shift+Enter)
- [ ] Verify data loads without errors

### Task 4.4: Initial Exploration

- [ ] Create new cell:

  ```python
  # First 5 rows
  df.head()
  ```

- [ ] Run and observe output
- [ ] New cell:

  ```python
  # Dataset information
  df.info()
  ```

- [ ] Run and note: How many columns? Any missing values?
- [ ] New cell:

  ```python
  # Statistical summary
  df.describe()
  ```

- [ ] Run and observe min, max, mean values

### Task 4.5: Check Column Names

- [ ] New cell:

  ```python
  # List all columns
  print("Columns in dataset:")
  print(df.columns.tolist())
  ```

- [ ] Run cell
- [ ] Copy column names to `ml_notes.md`

### Task 4.6: Identify Missing Values

- [ ] New cell:

  ```python
  # Check missing values
  missing = df.isnull().sum()
  print("Missing values per column:")
  print(missing[missing > 0])
  ```

- [ ] Run cell
- [ ] Note which columns have missing data

### Task 4.7: Handle Missing Values (Simple Method)

- [ ] Choose one column with missing values (e.g., 'Car')
- [ ] New cell:

  ```python
  # Before
  print(f"Missing in 'Car' column: {df['Car'].isnull().sum()}")
  
  # Fill with median (for numeric) or mode (for categorical)
  df['Car'] = df['Car'].fillna(df['Car'].median())
  
  # After
  print(f"Missing after fillna: {df['Car'].isnull().sum()}")
  ```
- [ ] Run and verify missing values reduced

### Task 4.8: Create Subset and Save
- [ ] New cell:
  ```python
  # Select important columns
  columns_to_keep = ['Rooms', 'Price', 'Distance', 'Bedroom2', 'Bathroom', 'Car', 'Landsize']
  df_subset = df[columns_to_keep].copy()
  
  # Drop any remaining rows with missing values
  df_subset = df_subset.dropna()
  
  print(f"Original shape: {df.shape}")
  print(f"Subset shape: {df_subset.shape}")
  
  # Save
  df_subset.to_csv('../../projects/data/melb_cleaned.csv', index=False)
  print("Saved to projects/data/melb_cleaned.csv")
  ```
- [ ] Run cell
- [ ] Verify file saved

### Task 4.9: Document EDA
- [ ] Add markdown cell in notebook
- [ ] Write summary:
  - Number of rows and columns
  - Columns with missing values
  - How you handled missing values
  - Final cleaned dataset size
- [ ] Save notebook

---

## 🔀 Level 5 — Intermediate Machine Learning

**Goal**: Learn proper train/test split and model evaluation.

### Task 5.1: Understand Train-Test Split
- [ ] Create `projects/notebooks/train_test_demo.ipynb`
- [ ] First cell:
  ```python
  import pandas as pd
  from sklearn.model_selection import train_test_split
  
  print("Train-Test Split Demo")
  print("="*50)
  ```
- [ ] Run cell

### Task 5.2: Load Cleaned Data
- [ ] New cell:
  ```python
  df = pd.read_csv('../data/melb_cleaned.csv')
  print(f"Dataset shape: {df.shape}")
  print(f"\nColumns: {df.columns.tolist()}")
  ```
- [ ] Run cell

### Task 5.3: Prepare Features and Target
- [ ] New cell:
  ```python
  # Separate features and target
  X = df.drop('Price', axis=1)
  y = df['Price']
  
  print(f"Features shape: {X.shape}")
  print(f"Target shape: {y.shape}")
  ```
- [ ] Run and verify shapes

### Task 5.4: Split Data
- [ ] New cell:
  ```python
  # Split into train and test sets
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )
  
  print(f"Training set: {X_train.shape}")
  print(f"Test set: {X_test.shape}")
  print(f"\nTrain/Test ratio: {len(X_train)/len(X):.1%} / {len(X_test)/len(X):.1%}")
  ```
- [ ] Run and observe the split

### Task 5.5: Train First Model (Decision Tree)
- [ ] New cell:
  ```python
  from sklearn.tree import DecisionTreeRegressor
  from sklearn.metrics import mean_absolute_error, r2_score
  
  # Create and train model
  dt_model = DecisionTreeRegressor(random_state=42)
  dt_model.fit(X_train, y_train)
  
  # Predictions
  dt_train_pred = dt_model.predict(X_train)
  dt_test_pred = dt_model.predict(X_test)
  
  # Evaluate
  dt_train_mae = mean_absolute_error(y_train, dt_train_pred)
  dt_test_mae = mean_absolute_error(y_test, dt_test_pred)
  
  print("Decision Tree Results:")
  print(f"  Train MAE: ${dt_train_mae:,.2f}")
  print(f"  Test MAE: ${dt_test_mae:,.2f}")
  ```
- [ ] Run and note the MAE values

### Task 5.6: Train Second Model (Random Forest)
- [ ] New cell:
  ```python
  from sklearn.ensemble import RandomForestRegressor
  
  # Create and train model
  rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
  rf_model.fit(X_train, y_train)
  
  # Predictions
  rf_train_pred = rf_model.predict(X_train)
  rf_test_pred = rf_model.predict(X_test)
  
  # Evaluate
  rf_train_mae = mean_absolute_error(y_train, rf_train_pred)
  rf_test_mae = mean_absolute_error(y_test, rf_test_pred)
  
  print("Random Forest Results:")
  print(f"  Train MAE: ${rf_train_mae:,.2f}")
  print(f"  Test MAE: ${rf_test_mae:,.2f}")
  ```
- [ ] Run and note the MAE values

### Task 5.7: Compare Models
- [ ] New cell:
  ```python
  import matplotlib.pyplot as plt
  
  models = ['Decision Tree', 'Random Forest']
  train_scores = [dt_train_mae, rf_train_mae]
  test_scores = [dt_test_mae, rf_test_mae]
  
  x = range(len(models))
  width = 0.35
  
  fig, ax = plt.subplots(figsize=(10, 6))
  ax.bar([i - width/2 for i in x], train_scores, width, label='Train MAE')
  ax.bar([i + width/2 for i in x], test_scores, width, label='Test MAE')
  
  ax.set_xlabel('Model')
  ax.set_ylabel('Mean Absolute Error ($)')
  ax.set_title('Model Comparison: Train vs Test MAE')
  ax.set_xticks(x)
  ax.set_xticklabels(models)
  ax.legend()
  
  plt.tight_layout()
  plt.savefig('../figures/model_comparison.png', dpi=150)
  print("Chart saved to projects/figures/model_comparison.png")
  plt.show()
  ```
- [ ] Run and verify chart appears
- [ ] Check that file saved

### Task 5.8: Cross-Validation
- [ ] New cell:
  ```python
  from sklearn.model_selection import cross_val_score
  
  # 5-fold cross-validation for both models
  dt_cv_scores = cross_val_score(dt_model, X, y, cv=5, 
                                   scoring='neg_mean_absolute_error')
  rf_cv_scores = cross_val_score(rf_model, X, y, cv=5,
                                   scoring='neg_mean_absolute_error')
  
  # Convert to positive MAE
  dt_cv_mae = -dt_cv_scores.mean()
  rf_cv_mae = -rf_cv_scores.mean()
  
  print("Cross-Validation Results (5-fold):")
  print(f"  Decision Tree CV MAE: ${dt_cv_mae:,.2f}")
  print(f"  Random Forest CV MAE: ${rf_cv_mae:,.2f}")
  ```
- [ ] Run and note which model performs better

### Task 5.9: Save Results to Markdown
- [ ] Create `projects/results.md`
- [ ] Write:
  ```markdown
  # Model Evaluation Results
  
  ## Date: [Today's Date]
  ## Level: 5 - Intermediate Machine Learning
  
  ### Dataset
  - Name: Melbourne Housing (Cleaned)
  - Rows: [fill in]
  - Features: Rooms, Distance, Bedroom2, Bathroom, Car, Landsize
  - Target: Price
  
  ### Models Compared
  
  #### 1. Decision Tree Regressor
  - Train MAE: $[fill in]
  - Test MAE: $[fill in]
  - CV MAE: $[fill in]
  
  #### 2. Random Forest Regressor
  - Train MAE: $[fill in]
  - Test MAE: $[fill in]
  - CV MAE: $[fill in]
  
  ### Conclusion
  [Which model performed better? Why do you think so?]
  
  ### Visualization
  ![Model Comparison](figures/model_comparison.png)
  ```
- [ ] Fill in the actual values from your results
- [ ] Save file

### Task 5.10: Commit Progress
- [ ] Save all notebooks
- [ ] Close JupyterLab
- [ ] In terminal: `git status`
- [ ] Add files: `git add projects/`
- [ ] Commit: `git commit -m "Level 5 complete: model comparison"`

---

## 📈 Level 6 — Data Visualization

**Goal**: Create clear, informative plots to understand data and results.

### Task 6.1: Setup Visualization Notebook
- [ ] Create `projects/notebooks/data_viz.ipynb`
- [ ] First cell:
  ```python
  import pandas as pd
  import matplotlib.pyplot as plt
  import seaborn as sns
  import numpy as np
  
  # Set style
  sns.set_style("whitegrid")
  plt.rcParams['figure.figsize'] = (12, 6)
  
  print("Visualization libraries loaded!")
  ```
- [ ] Run cell

### Task 6.2: Load Data
- [ ] New cell:
  ```python
  df = pd.read_csv('../data/melb_cleaned.csv')
  print(f"Data loaded: {df.shape}")
  df.head()
  ```
- [ ] Run cell

### Task 6.3: Create Histogram
- [ ] New cell:
  ```python
  # Histogram of house prices
  plt.figure(figsize=(10, 6))
  plt.hist(df['Price'], bins=50, edgecolor='black', alpha=0.7)
  plt.xlabel('Price ($)', fontsize=12)
  plt.ylabel('Frequency', fontsize=12)
  plt.title('Distribution of House Prices', fontsize=14, fontweight='bold')
  plt.axvline(df['Price'].mean(), color='red', linestyle='--', 
              label=f'Mean: ${df["Price"].mean():,.0f}')
  plt.legend()
  plt.tight_layout()
  plt.savefig('../figures/price_histogram.png', dpi=150)
  print("Saved: price_histogram.png")
  plt.show()
  ```
- [ ] Run and verify chart appears
- [ ] Check file saved

### Task 6.4: Create Box Plot
- [ ] New cell:
  ```python
  # Box plot for rooms vs price
  plt.figure(figsize=(12, 6))
  df.boxplot(column='Price', by='Rooms', figsize=(12, 6))
  plt.xlabel('Number of Rooms', fontsize=12)
  plt.ylabel('Price ($)', fontsize=12)
  plt.title('House Price Distribution by Number of Rooms', fontsize=14)
  plt.suptitle('')  # Remove default title
  plt.tight_layout()
  plt.savefig('../figures/price_by_rooms_boxplot.png', dpi=150)
  print("Saved: price_by_rooms_boxplot.png")
  plt.show()
  ```
- [ ] Run and observe the plot

### Task 6.5: Create Scatter Plot
- [ ] New cell:
  ```python
  # Scatter plot: Distance vs Price
  plt.figure(figsize=(10, 6))
  plt.scatter(df['Distance'], df['Price'], alpha=0.5, s=30)
  plt.xlabel('Distance from CBD (km)', fontsize=12)
  plt.ylabel('Price ($)', fontsize=12)
  plt.title('House Price vs Distance from CBD', fontsize=14, fontweight='bold')
  
  # Add trend line
  z = np.polyfit(df['Distance'], df['Price'], 1)
  p = np.poly1d(z)
  plt.plot(df['Distance'], p(df['Distance']), "r--", alpha=0.8, 
           label='Trend line')
  plt.legend()
  plt.tight_layout()
  plt.savefig('../figures/price_vs_distance_scatter.png', dpi=150)
  print("Saved: price_vs_distance_scatter.png")
  plt.show()
  ```
- [ ] Run and observe relationship

### Task 6.6: Correlation Heatmap
- [ ] New cell:
  ```python
  # Correlation matrix
  plt.figure(figsize=(10, 8))
  correlation = df.corr()
  sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
              center=0, square=True, linewidths=1)
  plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
  plt.tight_layout()
  plt.savefig('../figures/correlation_heatmap.png', dpi=150)
  print("Saved: correlation_heatmap.png")
  plt.show()
  ```
- [ ] Run and identify strongest correlations
- [ ] Note: Which features correlate most with Price?

### Task 6.7: Multiple Plots in One Figure
- [ ] New cell:
  ```python
  # Create 2x2 grid of plots
  fig, axes = plt.subplots(2, 2, figsize=(14, 10))
  
  # Plot 1: Rooms distribution
  axes[0, 0].hist(df['Rooms'], bins=15, edgecolor='black', alpha=0.7)
  axes[0, 0].set_title('Distribution of Rooms')
  axes[0, 0].set_xlabel('Number of Rooms')
  axes[0, 0].set_ylabel('Count')
  
  # Plot 2: Price vs Landsize
  axes[0, 1].scatter(df['Landsize'], df['Price'], alpha=0.5)
  axes[0, 1].set_title('Price vs Landsize')
  axes[0, 1].set_xlabel('Landsize (sqm)')
  axes[0, 1].set_ylabel('Price ($)')
  
  # Plot 3: Bathroom distribution
  axes[1, 0].hist(df['Bathroom'], bins=10, edgecolor='black', alpha=0.7, color='green')
  axes[1, 0].set_title('Distribution of Bathrooms')
  axes[1, 0].set_xlabel('Number of Bathrooms')
  axes[1, 0].set_ylabel('Count')
  
  # Plot 4: Car spaces distribution
  axes[1, 1].bar(df['Car'].value_counts().index, 
                  df['Car'].value_counts().values, 
                  alpha=0.7, color='orange')
  axes[1, 1].set_title('Distribution of Car Spaces')
  axes[1, 1].set_xlabel('Number of Car Spaces')
  axes[1, 1].set_ylabel('Count')
  
  plt.tight_layout()
  plt.savefig('../figures/combined_analysis.png', dpi=150)
  print("Saved: combined_analysis.png")
  plt.show()
  ```
- [ ] Run and verify all 4 plots appear

### Task 6.8: Update Results Document
- [ ] Open `projects/results.md`
- [ ] Add new section:
  ```markdown
  ## Data Visualization (Level 6)
  
  ### Key Visualizations Created:
  
  1. **Price Distribution**
     - ![Histogram](figures/price_histogram.png)
     - Observation: [Write 1-2 sentences about the distribution]
  
  2. **Price by Rooms**
     - ![Boxplot](figures/price_by_rooms_boxplot.png)
     - Observation: [How does price change with rooms?]
  
  3. **Price vs Distance**
     - ![Scatter](figures/price_vs_distance_scatter.png)
     - Observation: [What relationship do you see?]
  
  4. **Feature Correlations**
     - ![Heatmap](figures/correlation_heatmap.png)
     - Strongest correlations with Price: [List top 3]
  ```
- [ ] Fill in your observations
- [ ] Save file

### Task 6.9: Create Visualization Summary
- [ ] In notebook, add markdown cell
- [ ] Write summary:
  - 3 insights learned from visualizations
  - Which visualization was most useful?
  - Any surprising findings?
- [ ] Save notebook

---

## 🔧 Level 7 — Feature Engineering

**Goal**: Transform and create features to improve model performance.

### Task 7.1: Setup Feature Engineering Notebook
- [ ] Create `projects/notebooks/feature_engineering.ipynb`
- [ ] First cell:
  ```python
  import pandas as pd
  import numpy as np
  from sklearn.model_selection import train_test_split
  from sklearn.ensemble import RandomForestRegressor
  from sklearn.metrics import mean_absolute_error
  
  print("Feature Engineering Notebook")
  ```
- [ ] Run cell

### Task 7.2: Load Original Data
- [ ] New cell:
  ```python
  # Load data
  df = pd.read_csv('../data/melb_cleaned.csv')
  print(f"Original shape: {df.shape}")
  print(f"Columns: {df.columns.tolist()}")
  df.head()
  ```
- [ ] Run cell

### Task 7.3: Baseline Model (Before Feature Engineering)
- [ ] New cell:
  ```python
  # Baseline model with existing features
  X_baseline = df.drop('Price', axis=1)
  y = df['Price']
  
  X_train, X_test, y_train, y_test = train_test_split(
      X_baseline, y, test_size=0.2, random_state=42
  )
  
  baseline_model = RandomForestRegressor(n_estimators=100, random_state=42)
  baseline_model.fit(X_train, y_train)
  
  baseline_pred = baseline_model.predict(X_test)
  baseline_mae = mean_absolute_error(y_test, baseline_pred)
  
  print(f"Baseline MAE: ${baseline_mae:,.2f}")
  ```
- [ ] Run and note the baseline MAE

### Task 7.4: Create Interaction Feature
- [ ] New cell:
  ```python
  # Create copy for feature engineering
  df_fe = df.copy()
  
  # Feature 1: Rooms × Landsize (total livable potential)
  df_fe['Rooms_x_Landsize'] = df_fe['Rooms'] * df_fe['Landsize']
  
  print("New feature created: Rooms_x_Landsize")
  print(f"Sample values:")
  print(df_fe[['Rooms', 'Landsize', 'Rooms_x_Landsize']].head())
  ```
- [ ] Run and verify new feature

### Task 7.5: Handle Categorical Variables
- [ ] Check if dataset has categorical columns
- [ ] If yes, new cell:
  ```python
  # Check data types
  print("Data types:")
  print(df_fe.dtypes)
  
  # If there are object/categorical columns, create dummies
  # Example: if there's a 'Type' column
  if 'Type' in df_fe.columns:
      df_fe = pd.get_dummies(df_fe, columns=['Type'], drop_first=True)
      print("\nCreated dummy variables for 'Type'")
      print(f"New shape: {df_fe.shape}")
  ```
- [ ] Run if applicable

### Task 7.6: Create Ratio Features
- [ ] New cell:
  ```python
  # Feature 2: Bathroom to Bedroom ratio
  df_fe['Bath_to_Bedroom_ratio'] = df_fe['Bathroom'] / (df_fe['Bedroom2'] + 1)
  
  # Feature 3: Price per room estimate potential
  # We'll use this after we see model performance
  
  print("Created Bath_to_Bedroom_ratio")
  print(df_fe[['Bathroom', 'Bedroom2', 'Bath_to_Bedroom_ratio']].head())
  ```
- [ ] Run and verify

### Task 7.7: Polynomial Features (Simple)
- [ ] New cell:
  ```python
  # Feature 4: Square of distance (captures non-linear effect)
  df_fe['Distance_squared'] = df_fe['Distance'] ** 2
  
  print("Created Distance_squared feature")
  print(df_fe[['Distance', 'Distance_squared']].head())
  ```
- [ ] Run cell

### Task 7.8: Train Model with New Features
- [ ] New cell:
  ```python
  # Train model with engineered features
  X_fe = df_fe.drop('Price', axis=1)
  y_fe = df_fe['Price']
  
  X_train_fe, X_test_fe, y_train_fe, y_test_fe = train_test_split(
      X_fe, y_fe, test_size=0.2, random_state=42
  )
  
  fe_model = RandomForestRegressor(n_estimators=100, random_state=42)
  fe_model.fit(X_train_fe, y_train_fe)
  
  fe_pred = fe_model.predict(X_test_fe)
  fe_mae = mean_absolute_error(y_test_fe, fe_pred)
  
  print(f"Feature Engineered MAE: ${fe_mae:,.2f}")
  print(f"Baseline MAE: ${baseline_mae:,.2f}")
  print(f"Improvement: ${baseline_mae - fe_mae:,.2f} ({(baseline_mae-fe_mae)/baseline_mae*100:.2f}%)")
  ```
- [ ] Run and compare results

### Task 7.9: Feature Importance
- [ ] New cell:
  ```python
  # Get feature importances
  feature_importance = pd.DataFrame({
      'feature': X_fe.columns,
      'importance': fe_model.feature_importances_
  }).sort_values('importance', ascending=False)
  
  print("Top 10 Most Important Features:")
  print(feature_importance.head(10))
  
  # Visualize
  import matplotlib.pyplot as plt
  
  plt.figure(figsize=(10, 6))
  top_features = feature_importance.head(10)
  plt.barh(top_features['feature'], top_features['importance'])
  plt.xlabel('Importance')
  plt.title('Top 10 Feature Importances')
  plt.gca().invert_yaxis()
  plt.tight_layout()
  plt.savefig('../figures/feature_importance.png', dpi=150)
  print("\nSaved: feature_importance.png")
  plt.show()
  ```
- [ ] Run and identify most important features

### Task 7.10: Document Feature Engineering
- [ ] Update `projects/results.md`
- [ ] Add section:
  ```markdown
  ## Feature Engineering (Level 7)
  
  ### New Features Created:
  1. Rooms_x_Landsize - Interaction feature
  2. Bath_to_Bedroom_ratio - Ratio feature
  3. Distance_squared - Polynomial feature
  
  ### Performance Comparison:
  - Baseline MAE: $[fill in]
  - After Feature Engineering MAE: $[fill in]
  - Improvement: [fill in]%
  
  ### Top 5 Most Important Features:
  1. [fill in]
  2. [fill in]
  3. [fill in]
  4. [fill in]
  5. [fill in]
  
  ### Conclusion:
  [Did feature engineering help? Which features were most valuable?]
  ```
- [ ] Fill in values and save

---

## 🗄️ Level 8 — Intro to SQL

**Goal**: Learn basic SQL queries for data manipulation.

### Task 8.1: Install SQLite (Fedora)

- [ ] Check if installed: `sqlite3 --version`
- [ ] If not: `sudo dnf install sqlite -y`
- [ ] Verify: `sqlite3 --version`

### Task 8.2: Create Sample Database

- [ ] Create `projects/scripts/create_db.py`
- [ ] Write:

  ```python
  import sqlite3
  import pandas as pd
  
  # Load our data
  df = pd.read_csv('../data/melb_cleaned.csv')
  
  # Create database
  conn = sqlite3.connect('../data/housing.db')
  
  # Write dataframe to SQL table
  df.to_sql('houses', conn, if_exists='replace', index=False)
  
  print("Database created: housing.db")
  print(f"Table 'houses' created with {len(df)} rows")
  
  conn.close()
  ```

- [ ] Run: `cd projects/scripts && python create_db.py`
- [ ] Verify: `ls ../data/` should show `housing.db`

### Task 8.3: Basic SELECT Query

- [ ] In terminal, run: `sqlite3 projects/data/housing.db`
- [ ] Try query:

  ```sql
  SELECT * FROM houses LIMIT 5;
  ```

- [ ] Observe output
- [ ] Exit: `.quit`

### Task 8.4: Practice SQL Queries - SELECT Specific Columns

- [ ] Create `projects/scripts/sql_queries.py`
- [ ] Write:

  ```python
  import sqlite3
  import pandas as pd
  
  # Connect to database
  conn = sqlite3.connect('../data/housing.db')
  
  # Query 1: Select specific columns
  query1 = """
  SELECT Rooms, Price, Distance
  FROM houses
  LIMIT 10;
  """
  df1 = pd.read_sql_query(query1, conn)
  print("Query 1: First 10 houses with Rooms, Price, Distance")
  print(df1)
  print("\n" + "="*50 + "\n")
  ```

- [ ] Run and verify output

### Task 8.5: WHERE Clause

- [ ] Add to `sql_queries.py`:

  ```python
  # Query 2: Filter with WHERE
  query2 = """
  SELECT Rooms, Price, Distance
  FROM houses
  WHERE Rooms >= 4 AND Price < 800000
  LIMIT 10;
  """
  df2 = pd.read_sql_query(query2, conn)
  print("Query 2: Houses with 4+ rooms and price < $800k")
  print(df2)
  print("\n" + "="*50 + "\n")
  ```

- [ ] Run and verify filtered results

### Task 8.6: Aggregation (COUNT, AVG, MAX, MIN)

- [ ] Add to `sql_queries.py`:

  ```python
  # Query 3: Aggregations
  query3 = """
  SELECT 
      COUNT(*) as total_houses,
      AVG(Price) as avg_price,
      MAX(Price) as max_price,
      MIN(Price) as min_price
  FROM houses;
  """
  df3 = pd.read_sql_query(query3, conn)
  print("Query 3: Summary statistics")
  print(df3)
  print("\n" + "="*50 + "\n")
  ```

- [ ] Run and observe summary stats

### Task 8.7: GROUP BY

- [ ] Add to `sql_queries.py`:

  ```python
  # Query 4: GROUP BY
  query4 = """
  SELECT 
      Rooms,
      COUNT(*) as count,
      AVG(Price) as avg_price
  FROM houses
  GROUP BY Rooms
  ORDER BY Rooms;
  """
  df4 = pd.read_sql_query(query4, conn)
  print("Query 4: Average price by number of rooms")
  print(df4)
  print("\n" + "="*50 + "\n")
  
  conn.close()
  ```

- [ ] Run complete script

### Task 8.8: Export Query Results

- [ ] Create `export_sql.py`:

  ```python
  import sqlite3
  import pandas as pd
  
  conn = sqlite3.connect('../data/housing.db')
  
  # Custom query
  query = """
  SELECT *
  FROM houses
  WHERE Distance < 10 AND Rooms >= 3
  ORDER BY Price DESC;
  """
  
  df = pd.read_sql_query(query, conn)
  
  # Export to CSV
  df.to_csv('../data/filtered_houses.csv', index=False)
  print(f"Exported {len(df)} rows to filtered_houses.csv")
  
  conn.close()
  ```

- [ ] Run script
- [ ] Verify CSV created

### Task 8.9: SQL Cheat Sheet

- [ ] Create `projects/sql_notes.md`
- [ ] Write:

  ```markdown
  # SQL Basics - Quick Reference
  
  ## SELECT
  ```sql
  SELECT column1, column2 FROM table;
  SELECT * FROM table;  -- All columns
  ```
  
  ## WHERE (Filtering)

  ```sql
  SELECT * FROM table WHERE column > 100;
  SELECT * FROM table WHERE column = 'value' AND other_column < 50;
  ```
  
  ## ORDER BY

  ```sql
  SELECT * FROM table ORDER BY column DESC;
  ```
  
  ## LIMIT

  ```sql
  SELECT * FROM table LIMIT 10;
  ```
  
  ## Aggregations

  ```sql
  SELECT COUNT(*), AVG(column), MAX(column) FROM table;
  ```
  
  ## GROUP BY

  ```sql
  SELECT category, COUNT(*), AVG(price)
  FROM table
  GROUP BY category;
  ```
  
  ## My Practice Queries

  [Add your queries here as you practice]
  ```

- [ ] Save file

---

## 🧠 Level 9 — Intro to Deep Learning

**Goal**: Understand neural networks and train a simple model.

### Task 9.1: Deep Learning Concepts

- [ ] Read about neural networks (find 1 article online)
- [ ] Create `projects/dl_notes.md`
- [ ] Write brief summary:
  - What is a neural network?
  - What are layers?
  - What is an activation function?
  - What is backpropagation?
- [ ] Save file

### Task 9.2: Install Deep Learning Libraries

- [ ] Run: `pip install tensorflow` (or `pip install torch` for PyTorch)
- [ ] Test: `python -c "import tensorflow as tf; print(tf.__version__)"`
- [ ] Or: `python -c "import torch; print(torch.__version__)"`

### Task 9.3: Create Simple Classification Dataset

- [ ] Create `projects/scripts/simple_nn.py`
- [ ] Write:

  ```python
  import numpy as np
  from sklearn.datasets import make_classification
  from sklearn.model_selection import train_test_split
  from sklearn.preprocessing import StandardScaler
  from sklearn.neural_network import MLPClassifier
  from sklearn.metrics import accuracy_score, classification_report
  
  # Create dataset
  X, y = make_classification(n_samples=1000, n_features=20, 
                              n_informative=15, n_redundant=5,
                              random_state=42)
  
  print(f"Dataset shape: {X.shape}")
  print(f"Classes: {np.unique(y)}")
  ```

- [ ] Run and verify dataset created

### Task 9.4: Prepare Data

- [ ] Add to script:

  ```python
  # Split data
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )
  
  # Scale features (important for neural networks!)
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)
  
  print(f"Training set: {X_train_scaled.shape}")
  print(f"Test set: {X_test_scaled.shape}")
  ```

- [ ] Run and verify

### Task 9.5: Train Simple Neural Network

- [ ] Add to script:

  ```python
  # Create neural network (Multi-Layer Perceptron)
  # Architecture: 20 inputs -> 50 hidden -> 25 hidden -> 2 outputs
  mlp = MLPClassifier(
      hidden_layer_sizes=(50, 25),
      activation='relu',
      max_iter=100,
      random_state=42,
      verbose=True
  )
  
  # Train
  print("\nTraining neural network...")
  mlp.fit(X_train_scaled, y_train)
  print("Training complete!")
  ```
- [ ] Run and watch training progress

### Task 9.6: Evaluate Model
- [ ] Add to script:
  ```python
  # Predictions
  train_pred = mlp.predict(X_train_scaled)
  test_pred = mlp.predict(X_test_scaled)
  
  # Accuracy
  train_acc = accuracy_score(y_train, train_pred)
  test_acc = accuracy_score(y_test, test_pred)
  
  print(f"\nTrain Accuracy: {train_acc:.4f}")
  print(f"Test Accuracy: {test_acc:.4f}")
  
  print("\nClassification Report:")
  print(classification_report(y_test, test_pred))
  ```
- [ ] Run complete script
- [ ] Note accuracy values

### Task 9.7: Visualize Training Progress
- [ ] Create notebook: `projects/notebooks/neural_network.ipynb`
- [ ] Cell 1:
  ```python
  import numpy as np
  import matplotlib.pyplot as plt
  from sklearn.neural_network import MLPClassifier
  from sklearn.datasets import make_classification
  from sklearn.model_selection import train_test_split
  from sklearn.preprocessing import StandardScaler
  
  # Create and prepare data
  X, y = make_classification(n_samples=1000, n_features=20, 
                              n_informative=15, random_state=42)
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42)
  
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)
  ```

### Task 9.8: Track Loss Over Iterations
- [ ] Cell 2:
  ```python
  # Train with verbose output
  mlp = MLPClassifier(hidden_layer_sizes=(50, 25), 
                      max_iter=200, 
                      random_state=42)
  mlp.fit(X_train_scaled, y_train)
  
  # Plot loss curve
  plt.figure(figsize=(10, 6))
  plt.plot(mlp.loss_curve_)
  plt.xlabel('Iteration')
  plt.ylabel('Loss')
  plt.title('Neural Network Training Loss')
  plt.grid(True)
  plt.tight_layout()
  plt.savefig('../figures/nn_loss_curve.png', dpi=150)
  print("Saved: nn_loss_curve.png")
  plt.show()
  ```
- [ ] Run and observe loss decreasing

### Task 9.9: Compare Network Architectures
- [ ] Cell 3:
  ```python
  # Try different architectures
  architectures = [
      (50,),           # Single hidden layer
      (50, 25),        # Two hidden layers
      (100, 50, 25),   # Three hidden layers
  ]
  
  results = []
  
  for arch in architectures:
      mlp = MLPClassifier(hidden_layer_sizes=arch, 
                          max_iter=100, 
                          random_state=42)
      mlp.fit(X_train_scaled, y_train)
      test_score = mlp.score(X_test_scaled, y_test)
      results.append((str(arch), test_score))
      print(f"Architecture {arch}: Test Accuracy = {test_score:.4f}")
  
  # Visualize
  archs, scores = zip(*results)
  plt.figure(figsize=(10, 6))
  plt.bar(archs, scores, alpha=0.7)
  plt.xlabel('Architecture')
  plt.ylabel('Test Accuracy')
  plt.title('Neural Network Architecture Comparison')
  plt.ylim(0, 1)
  plt.tight_layout()
  plt.savefig('../figures/nn_architecture_comparison.png', dpi=150)
  plt.show()
  ```
- [ ] Run and identify best architecture

### Task 9.10: Document Deep Learning Experience
- [ ] Update `projects/dl_notes.md`
- [ ] Add:
  ```markdown
  ## My First Neural Network
  
  ### Architecture Used:
  - Input layer: 20 features
  - Hidden layer 1: 50 neurons, ReLU activation
  - Hidden layer 2: 25 neurons, ReLU activation
  - Output layer: 2 classes
  
  ### Results:
  - Train Accuracy: [fill in]
  - Test Accuracy: [fill in]
  - Training iterations: 100
  
  ### Observations:
  - [How did loss change during training?]
  - [Did deeper networks perform better?]
  - [What did you learn about neural networks?]
  
  ### Next Steps:
  - Try on real dataset
  - Experiment with more layers
  - Learn about regularization
  ```
- [ ] Save file

---

## 📸 Level 10 — Computer Vision

**Goal**: Introduction to image processing and CNNs.

### Task 10.1: Read About CNNs
- [ ] Find article on Convolutional Neural Networks
- [ ] Read about: convolution, pooling, filters
- [ ] Create `projects/cv_notes.md`
- [ ] Write 1-paragraph summary of how CNNs work

### Task 10.2: Install Image Processing Libraries
- [ ] Run: `pip install opencv-python pillow`
- [ ] Test: `python -c "import cv2; print(cv2.__version__)"`
- [ ] Test: `python -c "from PIL import Image; print('PIL works')"`

### Task 10.3: Load and Display Image
- [ ] Find or download a sample image (save to `projects/data/sample_image.jpg`)
- [ ] Create `projects/scripts/image_basics.py`
- [ ] Write:
  ```python
  import cv2
  import matplotlib.pyplot as plt
  
  # Load image
  img = cv2.imread('../data/sample_image.jpg')
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  
  # Display
  plt.figure(figsize=(8, 6))
  plt.imshow(img_rgb)
  plt.title('Original Image')
  plt.axis('off')
  plt.tight_layout()
  plt.savefig('../figures/original_image.png', dpi=150)
  print("Image loaded and saved!")
  plt.show()
  
  print(f"Image shape: {img.shape}")
  print(f"Image dtype: {img.dtype}")
  ```
- [ ] Run and verify image displays

### Task 10.4: Basic Image Transformations
- [ ] Add to script:
  ```python
  # Grayscale
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
  # Resize
  resized = cv2.resize(img_rgb, (200, 200))
  
  # Rotate
  rows, cols = img_rgb.shape[:2]
  M = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)
  rotated = cv2.warpAffine(img_rgb, M, (cols, rows))
  
  # Display all
  fig, axes = plt.subplots(2, 2, figsize=(12, 10))
  
  axes[0, 0].imshow(img_rgb)
  axes[0, 0].set_title('Original')
  axes[0, 0].axis('off')
  
  axes[0, 1].imshow(gray, cmap='gray')
  axes[0, 1].set_title('Grayscale')
  axes[0, 1].axis('off')
  
  axes[1, 0].imshow(resized)
  axes[1, 0].set_title('Resized (200x200)')
  axes[1, 0].axis('off')
  
  axes[1, 1].imshow(rotated)
  axes[1, 1].set_title('Rotated 45°')
  axes[1, 1].axis('off')
  
  plt.tight_layout()
  plt.savefig('../figures/image_transformations.png', dpi=150)
  print("Transformations saved!")
  plt.show()
  ```
- [ ] Run and verify

### Task 10.5: Try Pretrained Model (Optional - if system supports)
- [ ] This task requires more setup, mark as optional
- [ ] Read about pretrained models (ResNet, VGG, etc.)
- [ ] Note: Will explore this more in advanced courses

### Task 10.6: Document CV Learning
- [ ] Update `projects/cv_notes.md`
- [ ] Add sections on:
  - What you learned about image representations
  - How grayscale conversion works
  - Difference between CV and traditional ML
- [ ] Save file

---

## 📊 Level 11 — Time Series

**Goal**: Understand temporal data and basic forecasting.

### Task 11.1: Create Time Series Data
- [ ] Create `projects/notebooks/time_series.ipynb`
- [ ] Cell 1:
  ```python
  import pandas as pd
  import numpy as np
  import matplotlib.pyplot as plt
  from datetime import datetime, timedelta
  
  # Create sample time series data
  dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
  
  # Simulate data with trend and seasonality
  trend = np.linspace(100, 200, len(dates))
  seasonal = 20 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365)
  noise = np.random.normal(0, 5, len(dates))
  values = trend + seasonal + noise
  
  df_ts = pd.DataFrame({
      'date': dates,
      'value': values
  })
  
  print(df_ts.head())
  print(f"\nDataset shape: {df_ts.shape}")
  ```
- [ ] Run and verify data created

### Task 11.2: Visualize Time Series
- [ ] Cell 2:
  ```python
  plt.figure(figsize=(14, 6))
  plt.plot(df_ts['date'], df_ts['value'], linewidth=1)
  plt.xlabel('Date')
  plt.ylabel('Value')
  plt.title('Time Series Data - Daily Values')
  plt.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.savefig('../figures/time_series_plot.png', dpi=150)
  print("Saved: time_series_plot.png")
  plt.show()
  ```
- [ ] Run and observe patterns

### Task 11.3: Decompose Time Series
- [ ] Cell 3:
  ```python
  # Set date as index
  df_ts_indexed = df_ts.set_index('date')
  
  # Rolling statistics
  rolling_mean = df_ts_indexed['value'].rolling(window=30).mean()
  rolling_std = df_ts_indexed['value'].rolling(window=30).std()
  
  # Plot
  fig, axes = plt.subplots(2, 1, figsize=(14, 10))
  
  # Original with rolling mean
  axes[0].plot(df_ts_indexed.index, df_ts_indexed['value'], 
               label='Original', alpha=0.5)
  axes[0].plot(df_ts_indexed.index, rolling_mean, 
               label='30-day Rolling Mean', color='red', linewidth=2)
  axes[0].set_title('Time Series with Rolling Mean')
  axes[0].set_ylabel('Value')
  axes[0].legend()
  axes[0].grid(True, alpha=0.3)
  
  # Rolling standard deviation
  axes[1].plot(df_ts_indexed.index, rolling_std, 
               label='30-day Rolling Std', color='green')
  axes[1].set_title('Rolling Standard Deviation')
  axes[1].set_xlabel('Date')
  axes[1].set_ylabel('Std Dev')
  axes[1].legend()
  axes[1].grid(True, alpha=0.3)
  
  plt.tight_layout()
  plt.savefig('../figures/time_series_decomposition.png', dpi=150)
  plt.show()
  ```
- [ ] Run and analyze trend/variability

### Task 11.4: Naive Forecast
- [ ] Cell 4:
  ```python
  # Split data
  train_size = int(len(df_ts) * 0.8)
  train = df_ts[:train_size].copy()
  test = df_ts[train_size:].copy()
  
  # Naive forecast: use last value
  last_value = train['value'].iloc[-1]
  test['naive_forecast'] = last_value
  
  # Calculate error
  from sklearn.metrics import mean_absolute_error, mean_squared_error
  
  mae = mean_absolute_error(test['value'], test['naive_forecast'])
  rmse = np.sqrt(mean_squared_error(test['value'], test['naive_forecast']))
  
  print(f"Naive Forecast Results:")
  print(f"  MAE: {mae:.2f}")
  print(f"  RMSE: {rmse:.2f}")
  
  # Plot
  plt.figure(figsize=(14, 6))
  plt.plot(train['date'], train['value'], label='Training Data', alpha=0.7)
  plt.plot(test['date'], test['value'], label='Test Data (Actual)', alpha=0.7)
  plt.plot(test['date'], test['naive_forecast'], 
           label='Naive Forecast', linestyle='--', color='red')
  plt.axvline(train['date'].iloc[-1], color='black', 
              linestyle=':', label='Train/Test Split')
  plt.xlabel('Date')
  plt.ylabel('Value')
  plt.title('Naive Forecast vs Actual')
  plt.legend()
  plt.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.savefig('../figures/naive_forecast.png', dpi=150)
  plt.show()
  ```
- [ ] Run and evaluate forecast quality

### Task 11.5: Moving Average Forecast
- [ ] Cell 5:
  ```python
  # Moving average forecast
  window = 7
  test['ma_forecast'] = train['value'].rolling(window=window).mean().iloc[-1]
  
  mae_ma = mean_absolute_error(test['value'], test['ma_forecast'])
  rmse_ma = np.sqrt(mean_squared_error(test['value'], test['ma_forecast']))
  
  print(f"\n{window}-Day Moving Average Forecast:")
  print(f"  MAE: {mae_ma:.2f}")
  print(f"  RMSE: {rmse_ma:.2f}")
  
  # Compare methods
  print(f"\nComparison:")
  print(f"  Naive - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
  print(f"  MA    - MAE: {mae_ma:.2f}, RMSE: {rmse_ma:.2f}")
  ```
- [ ] Run and compare methods

### Task 11.6: Document Time Series Learning
- [ ] Create `projects/ts_notes.md`
- [ ] Write:
  ```markdown
  # Time Series Analysis Notes
  
  ## Key Concepts Learned:
  - Time series components: trend, seasonality, noise
  - Rolling statistics for smoothing
  - Train/test split for temporal data
  
  ## Forecasting Methods Tried:
  1. Naive forecast (last value)
     - MAE: [fill in]
  2. Moving average
     - MAE: [fill in]
  
  ## Observations:
  - [Which method performed better?]
  - [What patterns did you observe?]
  
  ## Next Steps:
  - Learn ARIMA models
  - Explore seasonal decomposition
  - Try exponential smoothing
  ```
- [ ] Save file

---

## 🧹 Level 12 — Data Cleaning

**Goal**: Master techniques for handling messy data.

### Task 12.1: Create Messy Dataset
- [ ] Create `projects/scripts/create_messy_data.py`
- [ ] Write:
  ```python
  import pandas as pd
  import numpy as np
  
  # Create messy dataset
  data = {
      'ID': [1, 2, 3, 3, 4, 5, 6, 7, 8, 9],  # Duplicate
      'Name': ['John', 'jane', 'Bob  ', 'Bob', 'ALICE', 
               'Charlie', 'Eve', 'frank', 'Grace', 'Henry'],
      'Age': [25, 30, np.nan, 35, 28, 150, 22, 27, 29, 31],  # Invalid value
      'Email': ['john@email.com', 'jane@email', 'bob@email.com',
                'bob@email.com', 'alice@email.com', 'charlie@email.com',
                None, 'frank@email.com', 'grace@', 'henry@email.com'],
      'Salary': [50000, 60000, 55000, 55000, '70000', 80000, 
                 45000, None, 62000, 58000],  # Mixed types
      'Date': ['2024-01-15', '2024/02/20', '15-03-2024', 
               '2024-04-10', '2024-05-22', '2024-06-30',
               '2024-07-18', None, '2024-09-05', '2024-10-12']  # Inconsistent format
  }
  
  df_messy = pd.DataFrame(data)
  df_messy.to_csv('../data/messy_data.csv', index=False)
  
  print("Messy dataset created!")
  print(df_messy)
  print(f"\nShape: {df_messy.shape}")
  print(f"\nData types:\n{df_messy.dtypes}")
  print(f"\nMissing values:\n{df_messy.isnull().sum()}")
  ```
- [ ] Run and inspect messy data

### Task 12.2: Identify Data Quality Issues
- [ ] Create `projects/notebooks/data_cleaning.ipynb`
- [ ] Cell 1:
  ```python
  import pandas as pd
  import numpy as np
  
  # Load messy data
  df = pd.read_csv('../data/messy_data.csv')
  
  print("=== DATA QUALITY ISSUES ===\n")
  
  # 1. Duplicates
  print(f"1. Duplicate rows: {df.duplicated().sum()}")
  print(f"   Duplicated IDs: {df['ID'].duplicated().sum()}\n")
  
  # 2. Missing values
  print(f"2. Missing values:\n{df.isnull().sum()}\n")
  
  # 3. Data types
  print(f"3. Data types:\n{df.dtypes}\n")
  
  # 4. Inconsistent formatting
  print("4. Name formatting issues:")
  print(f"   {df['Name'].tolist()}\n")
  
  # 5. Invalid values
  print("5. Age statistics:")
  print(f"   {df['Age'].describe()}\n")
  
  print("6. Email issues:")
  print(df[df['Email'].notna()]['Email'].tolist())
  ```
- [ ] Run and list all issues found

### Task 12.3: Remove Duplicates
- [ ] Cell 2:
  ```python
  # Create copy for cleaning
  df_clean = df.copy()
  
  print("Before removing duplicates:")
  print(f"  Shape: {df_clean.shape}")
  
  # Remove exact duplicate rows
  df_clean = df_clean.drop_duplicates()
  
  print(f"\nAfter removing duplicates:")
  print(f"  Shape: {df_clean.shape}")
  print(f"  Removed: {len(df) - len(df_clean)} rows")
  ```
- [ ] Run and verify duplicates removed

### Task 12.4: Standardize Text
- [ ] Cell 3:
  ```python
  # Standardize Name column
  print("Before standardization:")
  print(df_clean['Name'].tolist())
  
  # Clean: lowercase, strip whitespace, capitalize
  df_clean['Name'] = df_clean['Name'].str.strip().str.lower().str.capitalize()
  
  print("\nAfter standardization:")
  print(df_clean['Name'].tolist())
  ```
- [ ] Run and verify names cleaned

### Task 12.5: Fix Data Types
- [ ] Cell 4:
  ```python
  # Fix Salary column (mixed types)
  print("Salary column issues:")
  print(f"  Type: {df_clean['Salary'].dtype}")
  print(f"  Sample: {df_clean['Salary'].head()}")
  
  # Convert to numeric
  df_clean['Salary'] = pd.to_numeric(df_clean['Salary'], errors='coerce')
  
  print(f"\nAfter conversion:")
  print(f"  Type: {df_clean['Salary'].dtype}")
  print(f"  Sample: {df_clean['Salary'].head()}")
  ```
- [ ] Run and verify type conversion

### Task 12.6: Handle Invalid Values
- [ ] Cell 5:
  ```python
  # Fix invalid age (150 is unrealistic)
  print("Age distribution:")
  print(df_clean['Age'].describe())
  
  # Replace invalid ages (> 100) with NaN
  df_clean.loc[df_clean['Age'] > 100, 'Age'] = np.nan
  
  print("\nAfter fixing:")
  print(df_clean['Age'].describe())
  ```
- [ ] Run and verify

### Task 12.7: Standardize Dates
- [ ] Cell 6:
  ```python
  # Standardize date format
  print("Date column before:")
  print(df_clean['Date'].tolist())
  
  # Convert to datetime (handles multiple formats)
  df_clean['Date'] = pd.to_datetime(df_clean['Date'], 
                                     format='mixed', 
                                     errors='coerce')
  
  print("\nDate column after:")
  print(df_clean['Date'].tolist())
  print(f"\nType: {df_clean['Date'].dtype}")
  ```
- [ ] Run and verify dates standardized

### Task 12.8: Handle Missing Values
- [ ] Cell 7:
  ```python
  print("Missing values summary:")
  print(df_clean.isnull().sum())
  
  # Strategy 1: Fill Age with median
  median_age = df_clean['Age'].median()
  df_clean['Age'].fillna(median_age, inplace=True)
  
  # Strategy 2: Fill Salary with mean
  mean_salary = df_clean['Salary'].mean()
  df_clean['Salary'].fillna(mean_salary, inplace=True)
  
  # Strategy 3: Drop rows with missing Email or Date (critical fields)
  df_clean = df_clean.dropna(subset=['Email', 'Date'])
  
  print("\nAfter handling missing values:")
  print(df_clean.isnull().sum())
  print(f"\nFinal shape: {df_clean.shape}")
  ```
- [ ] Run and verify missing values handled

### Task 12.9: Validate Email Format
- [ ] Cell 8:
  ```python
  import re
  
  # Simple email validation
  def is_valid_email(email):
      pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
      return bool(re.match(pattern, str(email)))
  
  # Check emails
  df_clean['Email_Valid'] = df_clean['Email'].apply(is_valid_email)
  
  print("Email validation results:")
  print(df_clean[['Email', 'Email_Valid']])
  
  # Keep only valid emails
  df_clean = df_clean[df_clean['Email_Valid']]
  df_clean = df_clean.drop('Email_Valid', axis=1)
  
  print(f"\nRows after email validation: {len(df_clean)}")
  ```
- [ ] Run and verify

### Task 12.10: Save Cleaned Data
- [ ] Cell 9:
  ```python
  # Final cleaned dataset
  print("=== CLEANING SUMMARY ===")
  print(f"Original rows: {len(df)}")
  print(f"Cleaned rows: {len(df_clean)}")
  print(f"Rows removed: {len(df) - len(df_clean)}")
  print(f"\nCleaned columns: {df_clean.columns.tolist()}")
  print(f"\nData types:\n{df_clean.dtypes}")
  print(f"\nNo missing values: {df_clean.isnull().sum().sum() == 0}")
  
  # Save
  df_clean.to_csv('../data/cleaned_data.csv', index=False)
  print("\nSaved to: projects/data/cleaned_data.csv")
  
  # Display final data
  print("\nFinal cleaned data:")
  print(df_clean)
  ```
- [ ] Run and verify saved
- [ ] Open CSV to inspect

### Task 12.11: Document Cleaning Process
- [ ] Create `projects/data_cleaning_log.md`
- [ ] Write:
  ```markdown
  # Data Cleaning Log
  
  ## Original Dataset Issues:
  1. Duplicate rows: 1
  2. Missing values: Age (1), Email (1), Salary (1), Date (1)
  3. Inconsistent formatting: Names had mixed case and spaces
  4. Invalid values: Age = 150
  5. Mixed data types: Salary column
  6. Inconsistent date formats
  7. Invalid emails: Missing @ or domain
  
  ## Cleaning Steps Performed:
  1. ✓ Removed duplicate rows
  2. ✓ Standardized name formatting (strip, lowercase, capitalize)
  3. ✓ Converted Salary to numeric type
  4. ✓ Replaced invalid age values with NaN
  5. ✓ Standardized date format to datetime
  6. ✓ Filled missing Age with median
  7. ✓ Filled missing Salary with mean
  8. ✓ Dropped rows with missing critical fields (Email, Date)
  9. ✓ Validated email format
  10. ✓ Removed rows with invalid emails
  
  ## Results:
  - Original: 10 rows
  - Final: [fill in] rows
  - Data quality: 100% complete, standardized formats
  
  ## Saved Files:
  - Original: `messy_data.csv`
  - Cleaned: `cleaned_data.csv`
  ```
- [ ] Save file

---

## ⚖️ Level 13 — Intro to AI Ethics

**Goal**: Understand responsible AI practices and considerations.

### Task 13.1: Read About AI Ethics
- [ ] Find and read article on one of these topics:
  - Bias in machine learning
  - Fairness in AI systems
  - Privacy and data protection
  - Algorithmic transparency
  - AI safety
- [ ] Take notes while reading

### Task 13.2: Create Ethics Notes
- [ ] Create `projects/ai_ethics_notes.md`
- [ ] Write summary (1-2 paragraphs) covering:
  - Main ethical concern discussed
  - Real-world example or case study
  - Why this matters for AI practitioners
  - How to address this concern
- [ ] Save file

### Task 13.3: Bias Detection Exercise
- [ ] Create `projects/notebooks/bias_detection.ipynb`
- [ ] Cell 1:
  ```python
  import pandas as pd
  import numpy as np
  from sklearn.model_selection import train_test_split
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.metrics import accuracy_score, classification_report
  
  # Create synthetic dataset with potential bias
  np.random.seed(42)
  n_samples = 1000
  
  # Protected attribute: gender (0=Female, 1=Male)
  gender = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
  
  # Features
  experience = np.random.randint(0, 20, n_samples)
  education = np.random.randint(1, 5, n_samples)  # 1=HS, 4=PhD
  
  # Target: promotion (biased toward males in data generation)
  # This simulates historical bias in promotion decisions
  promotion = []
  for i in range(n_samples):
      score = experience[i] * 2 + education[i] * 3
      # Add bias: males more likely to be promoted
      if gender[i] == 1:
          score += 10  # Unfair advantage
      threshold = 35 + np.random.normal(0, 5)
      promotion.append(1 if score > threshold else 0)
  
  df = pd.DataFrame({
      'gender': gender,
      'experience': experience,
      'education': education,
      'promotion': promotion
  })
  
  print("Dataset created (with simulated bias)")
  print(df.head(10))
  print(f"\nPromotion rate by gender:")
  print(df.groupby('gender')['promotion'].mean())
  ```
- [ ] Run and observe bias in data

### Task 13.4: Train Model and Check Fairness
- [ ] Cell 2:
  ```python
  # Train model
  X = df[['gender', 'experience', 'education']]
  y = df['promotion']
  
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )
  
  model = RandomForestClassifier(random_state=42)
  model.fit(X_train, y_train)
  
  # Predictions
  y_pred = model.predict(X_test)
  
  print("Overall Accuracy:", accuracy_score(y_test, y_pred))
  
  # Check fairness: accuracy by gender
  test_with_pred = X_test.copy()
  test_with_pred['actual'] = y_test.values
  test_with_pred['predicted'] = y_pred
  
  print("\n=== Fairness Analysis ===")
  for gender_val in [0, 1]:
      gender_name = "Female" if gender_val == 0 else "Male"
      subset = test_with_pred[test_with_pred['gender'] == gender_val]
      accuracy = (subset['actual'] == subset['predicted']).mean()
      print(f"{gender_name} accuracy: {accuracy:.3f}")
  ```
- [ ] Run and note differences

### Task 13.5: Mitigation Strategy
- [ ] Cell 3:
  ```python
  # Try training without gender feature (fairness-aware)
  X_fair = df[['experience', 'education']]  # Remove gender
  X_train_fair, X_test_fair, y_train_fair, y_test_fair = train_test_split(
      X_fair, y, test_size=0.2, random_state=42
  )
  
  model_fair = RandomForestClassifier(random_state=42)
  model_fair.fit(X_train_fair, y_train_fair)
  
  y_pred_fair = model_fair.predict(X_test_fair)
  
  print("Fair Model (without gender feature):")
  print("Overall Accuracy:", accuracy_score(y_test_fair, y_pred_fair))
  
  # Check if removing gender improved fairness
  # Note: Gender bias may still exist through correlated features
  test_fair = X_test.copy()
  test_fair['predicted'] = y_pred_fair
  test_fair['actual'] = y_test.values
  
  print("\n=== Fairness Analysis (Fair Model) ===")
  for gender_val in [0, 1]:
      gender_name = "Female" if gender_val == 0 else "Male"
      subset = test_fair[test_fair['gender'] == gender_val]
      accuracy = (subset['actual'] == subset['predicted']).mean()
      print(f"{gender_name} accuracy: {accuracy:.3f}")
  ```
- [ ] Run and compare

### Task 13.6: Document Ethics Learnings
- [ ] Update `projects/ai_ethics_notes.md`
- [ ] Add section:
  ```markdown
  ## Bias Detection Exercise
  
  ### Scenario:
  Simulated promotion decisions with gender bias in historical data.
  
  ### Findings:
  - Model trained with gender feature:
    - Overall accuracy: [fill in]
    - Female accuracy: [fill in]
    - Male accuracy: [fill in]
  
  - Model without gender feature:
    - Overall accuracy: [fill in]
    - Improved fairness: [Yes/No]
  
  ### Key Lessons:
  1. Historical bias in data propagates to models
  2. Removing protected attributes doesn't always solve bias
  3. Need systematic fairness evaluation
  4. Accuracy alone isn't enough - must check subgroup performance
  
  ### Best Practices:
  - Always analyze model performance across different groups
  - Question data sources and collection methods
  - Consider fairness metrics beyond accuracy
  - Document known limitations and biases
  - Involve diverse stakeholders in AI development
  ```
- [ ] Save file

---

## 🗺️ Level 14 — Geospatial Analysis (Optional)

**Goal**: Basic understanding of geographic data.

### Task 14.1: Check for Coordinate Data
- [ ] Review Melbourne dataset for latitude/longitude
- [ ] If coordinates exist, proceed; otherwise, skip this level
- [ ] Create `projects/geo_notes.md`

### Task 14.2: Install Geospatial Libraries
- [ ] Run: `pip install folium geopandas`
- [ ] Test: `python -c "import folium; print('Folium installed')"`

### Task 14.3: Simple Map Visualization
- [ ] Create `projects/notebooks/geospatial.ipynb`
- [ ] Cell 1:
  ```python
  import pandas as pd
  import folium
  
  # Load data (if it has coordinates)
  df = pd.read_csv('../data/melb_cleaned.csv')
  
  # Check for coordinate columns
  print("Columns:", df.columns.tolist())
  
  # If coordinates exist:
  if 'Latitude' in df.columns and 'Longitude' in df.columns:
      print(f"\nCoordinates available for {df['Latitude'].notna().sum()} properties")
      
      # Create map centered on Melbourne
      melbourne_map = folium.Map(
          location=[-37.8136, 144.9631],
          zoom_start=11
      )
      
      # Add markers for first 100 properties
      sample = df[df['Latitude'].notna()].head(100)
      for idx, row in sample.iterrows():
          folium.CircleMarker(
              location=[row['Latitude'], row['Longitude']],
              radius=3,
              popup=f"Price: ${row['Price']:,.0f}",
              color='blue',
              fill=True
          ).add_to(melbourne_map)
      
      # Save map
      melbourne_map.save('../figures/melbourne_properties_map.html')
      print("Map saved to: melbourne_properties_map.html")
  else:
      print("No coordinate columns found")
  ```
- [ ] Run if data has coordinates
- [ ] Open HTML file in browser

### Task 14.4: Document Geospatial Work
- [ ] Update `projects/geo_notes.md`
- [ ] Note whether coordinates were available
- [ ] If completed: describe what you learned about spatial patterns

---

## 🔍 Level 15 — Explainability

**Goal**: Understand model decisions and interpretability.

### Task 15.1: Feature Importance (Simple Method)
- [ ] Create `projects/notebooks/explainability.ipynb`
- [ ] Cell 1:
  ```python
  import pandas as pd
  import numpy as np
  from sklearn.ensemble import RandomForestRegressor
  from sklearn.model_selection import train_test_split
  import matplotlib.pyplot as plt
  
  # Load data
  df = pd.read_csv('../data/melb_cleaned.csv')
  
  # Prepare data
  X = df.drop('Price', axis=1)
  y = df['Price']
  
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )
  
  # Train model
  model = RandomForestRegressor(n_estimators=100, random_state=42)
  model.fit(X_train, y_train)
  
  print("Model trained!")
  ```
- [ ] Run cell

### Task 15.2: Extract and Visualize Feature Importance
- [ ] Cell 2:
  ```python
  # Get feature importances
  importance_df = pd.DataFrame({
      'feature': X.columns,
      'importance': model.feature_importances_
  }).sort_values('importance', ascending=False)
  
  print("Feature Importances:")
  print(importance_df)
  
  # Visualize
  plt.figure(figsize=(10, 6))
  plt.barh(importance_df['feature'], importance_df['importance'])
  plt.xlabel('Importance Score')
  plt.title('Feature Importance for Price Prediction')
  plt.gca().invert_yaxis()
  plt.tight_layout()
  plt.savefig('../figures/feature_importance_explainability.png', dpi=150)
  print("\nSaved: feature_importance_explainability.png")
  plt.show()
  ```
- [ ] Run and identify most important features

### Task 15.3: Single Prediction Explanation
- [ ] Cell 3:
  ```python
  # Pick one sample to explain
  sample_idx = 0
  sample = X_test.iloc[sample_idx:sample_idx+1]
  actual_price = y_test.iloc[sample_idx]
  predicted_price = model.predict(sample)[0]
  
  print("=== Explaining a Single Prediction ===")
  print(f"\nSample features:")
  for col in sample.columns:
      print(f"  {col}: {sample[col].values[0]}")
  
  print(f"\nActual Price: ${actual_price:,.2f}")
  print(f"Predicted Price: ${predicted_price:,.2f}")
  print(f"Difference: ${abs(actual_price - predicted_price):,.2f}")
  
  # Show feature contributions (simplified)
  contributions = sample.values[0] * importance_df.set_index('feature')['importance']
  contributions_df = pd.DataFrame({
      'feature': sample.columns,
      'value': sample.values[0],
      'importance': importance_df.set_index('feature').loc[sample.columns, 'importance'].values,
      'contribution': contributions.values
  }).sort_values('contribution', ascending=False)
  
  print("\nFeature Contributions (simplified):")
  print(contributions_df)
  ```
- [ ] Run and understand how features influenced prediction

### Task 15.4: Partial Dependence Plot (Optional)
- [ ] Install: `pip install scikit-learn --upgrade`
- [ ] Cell 4:
  ```python
  from sklearn.inspection import PartialDependenceDisplay
  
  # Create partial dependence plot for top feature
  top_feature = importance_df.iloc[0]['feature']
  
  fig, ax = plt.subplots(figsize=(10, 6))
  PartialDependenceDisplay.from_estimator(
      model, X_train, [top_feature], ax=ax
  )
  plt.suptitle(f'Partial Dependence Plot: {top_feature}')
  plt.tight_layout()
  plt.savefig('../figures/partial_dependence.png', dpi=150)
  print(f"PDP saved for feature: {top_feature}")
  plt.show()
  ```
- [ ] Run and interpret relationship

### Task 15.5: Document Explainability Findings
- [ ] Create `projects/explainability_notes.md`
- [ ] Write:
  ```markdown
  # Model Explainability Analysis
  
  ## Model: Random Forest Regressor
  ## Task: House Price Prediction
  
  ### Top 5 Most Important Features:
  1. [Feature name] - Importance: [value]
  2. [Feature name] - Importance: [value]
  3. [Feature name] - Importance: [value]
  4. [Feature name] - Importance: [value]
  5. [Feature name] - Importance: [value]
  
  ### Insights:
  - [Which feature matters most and why?]
  - [Any surprising findings?]
  - [How do features interact?]
  
  ### Single Prediction Analysis:
  - Sample house features: [describe]
  - Predicted price: $[value]
  - Key contributing factors: [list]
  
  ### Why Explainability Matters:
  - Builds trust in model predictions
  - Helps identify biases or errors
  - Enables domain experts to validate logic
  - Required for regulatory compliance in some domains
  
  ### Methods Used:
  1. Feature importance (Random Forest built-in)
  2. Single prediction breakdown
  3. Partial dependence plots (optional)
  ```
- [ ] Save file

---

## 🎮 Level 16 — Intro to Reinforcement Learning / Game AI

**Goal**: Understand RL concepts and basic applications.

### Task 16.1: Read About Reinforcement Learning
- [ ] Find article or video on RL basics
- [ ] Learn about: agent, environment, reward, policy
- [ ] Create `projects/rl_notes.md`
- [ ] Write 1-paragraph summary of RL

### Task 16.2: Simple RL Concept
- [ ] In `rl_notes.md`, describe:
  - What is an agent?
  - What is an environment?
  - What is a reward?
  - What is a policy?
- [ ] Give one real-world use case

### Task 16.3: Install RL Library (Optional)
- [ ] Run: `pip install gymnasium`
- [ ] Test: `python -c "import gymnasium as gym; print('Gym installed')"`

### Task 16.4: Simple Bandit Problem (Conceptual)
- [ ] Create `projects/notebooks/rl_intro.ipynb`
- [ ] Cell 1:
  ```python
  import numpy as np
  import matplotlib.pyplot as plt
  
  # Multi-armed bandit problem (simplified RL)
  # Imagine 3 slot machines with different win rates
  
  # True probabilities (unknown to agent)
  true_probs = [0.3, 0.5, 0.7]  # Machine 3 is best
  n_machines = len(true_probs)
  
  # Agent's estimates (start at 0.5)
  estimates = [0.5] * n_machines
  counts = [0] * n_machines
  
  # Exploration vs exploitation
  epsilon = 0.1  # 10% exploration
  n_rounds = 1000
  
  rewards = []
  
  for round in range(n_rounds):
      # Choose action (epsilon-greedy)
      if np.random.random() < epsilon:
          # Explore: random choice
          choice = np.random.randint(n_machines)
      else:
          # Exploit: choose best estimate
          choice = np.argmax(estimates)
      
      # Get reward (1 or 0)
      reward = 1 if np.random.random() < true_probs[choice] else 0
      rewards.append(reward)
      
      # Update estimate
      counts[choice] += 1
      estimates[choice] += (reward - estimates[choice]) / counts[choice]
  
  print("=== Multi-Armed Bandit Results ===")
  print(f"True probabilities: {true_probs}")
  print(f"Learned estimates: {[f'{e:.3f}' for e in estimates]}")
  print(f"Times chosen: {counts}")
  print(f"Average reward: {np.mean(rewards):.3f}")
  
  # Plot
  plt.figure(figsize=(10, 6))
  plt.plot(np.cumsum(rewards) / (np.arange(len(rewards)) + 1))
  plt.xlabel('Round')
  plt.ylabel('Average Reward')
  plt.title('Learning Curve: Multi-Armed Bandit')
  plt.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.savefig('../figures/rl_learning_curve.png', dpi=150)
  print("\nSaved: rl_learning_curve.png")
  plt.show()
  ```
- [ ] Run and observe learning

### Task 16.5: Document RL Understanding
- [ ] Update `projects/rl_notes.md`
- [ ] Add section:
  ```markdown
  ## Multi-Armed Bandit Exercise
  
  ### Problem:
  3 slot machines with unknown win rates. Goal: maximize rewards.
  
  ### Strategy:
  - Epsilon-greedy: 10% explore, 90% exploit
  
  ### Results:
  - True probabilities: [fill in]
  - Learned estimates: [fill in]
  - Agent successfully learned which machine was best: [Yes/No]
  
  ### Key RL Concepts:
  1. **Exploration**: Trying new actions to learn
  2. **Exploitation**: Using current knowledge to maximize reward
  3. **Trade-off**: Balance between exploring and exploiting
  
  ### Real-World RL Applications:
  - Game playing (Chess, Go, video games)
  - Robotics control
  - Resource allocation
  - Recommendation systems
  - Autonomous vehicles
  
  ### Next Steps to Learn:
  - Q-learning
  - Deep Q-Networks (DQN)
  - Policy gradients
  - OpenAI Gym environments
  ```
- [ ] Save file

---

## ✅ Final Review and Next Steps

### Task: Complete Learning Summary

- [ ] Create `projects/LEARNING_SUMMARY.md`
- [ ] Write:

  ```markdown
  # Sandip's AI Learning Journey - Summary
  
  ## Completion Date: [Date]
  
  ## Levels Completed:
  - [x] Level 1: Intro to Programming
  - [x] Level 2: Python Core
  - [x] Level 3: Intro to Machine Learning
  - [x] Level 4: Pandas & EDA
  - [x] Level 5: Intermediate ML
  - [x] Level 6: Data Visualization
  - [x] Level 7: Feature Engineering
  - [x] Level 8: SQL
  - [x] Level 9: Deep Learning
  - [x] Level 10: Computer Vision
  - [x] Level 11: Time Series
  - [x] Level 12: Data Cleaning
  - [x] Level 13: AI Ethics
  - [x] Level 14: Geospatial (Optional)
  - [x] Level 15: Explainability
  - [x] Level 16: RL Intro
  
  ## Key Projects Completed:
  1. Melbourne housing price prediction
  2. Decision tree and Random Forest models
  3. Feature engineering experiments
  4. Data cleaning pipeline
  5. Bias detection in ML
  6. Multi-armed bandit RL
  
  ## Skills Acquired:
  - Python programming
  - Data manipulation (Pandas)
  - Machine learning (scikit-learn)
  - Data visualization (Matplotlib, Seaborn)
  - Feature engineering
  - SQL queries
  - Basic neural networks
  - Model explainability
  - AI ethics awareness
  
  ## Best Model Performance:
  - Task: House price prediction
  - Model: [Your best model]
  - Metric: MAE = $[value]
  
  ## Favorite Learning Moment:
  [Write 2-3 sentences]
  
  ## Next Goals:
  1. [Advanced deep learning]
  2. [Kaggle competitions]
  3. [Real-world project]
  4. [Specialized area of interest]
  
  ## Files Created:
  - Notebooks: [count]
  - Scripts: [count]
  - Datasets: [count]
  - Visualizations: [count]
  
  ## Advice to Future Self:
  [What did you learn about learning AI/ML?]
  ```

- [ ] Save file

### Task: Commit All Work

- [ ] Run: `git add projects/`
- [ ] Run: `git commit -m "Completed all 16 levels of AI learning guide"`
- [ ] Run: `git push origin sandip`

### Task: Celebrate! 🎉

- [ ] Review all notebooks and scripts you created
- [ ] Choose your best visualization to share
- [ ] Plan next learning goals

---

## 📚 Additional Resources

### Online Courses:

- Kaggle Learn (kaggle.com/learn)
- Fast.ai Practical Deep Learning
- Andrew Ng's Machine Learning course
- Google Machine Learning Crash Course

### Practice Platforms:

- Kaggle competitions
- LeetCode (for coding)
- HackerRank (Python & ML)

### Books (Free):

- "Python for Data Analysis" by Wes McKinney
- "Hands-On Machine Learning" by Aurélien Géron
- "Deep Learning" by Goodfellow et al. (deeplearningbook.org)

---

**Remember**: Learning AI/ML is a journey. Take breaks, celebrate small wins, and don't rush. Good luck, Sandip! 🚀