# How to Run This Project - Step by Step Guide

This guide will walk you through running the Traffic Accidents Analysis project from scratch, even if you've never done this before.

---

## Table of Contents
1. [Before You Start](#before-you-start)
2. [Quick Start (If Everything is Already Set Up)](#quick-start)
3. [Detailed Setup (First Time)](#detailed-setup-first-time)
4. [Running the Notebook](#running-the-notebook)
5. [Running the Web App](#running-the-web-app)
6. [Troubleshooting](#troubleshooting)
7. [What Each File Does](#what-each-file-does)

---

## Before You Start

### What You Need
- **Computer**: Windows, Mac, or Linux
- **Python**: Version 3.8 or higher
- **Internet**: For downloading packages
- **Time**: About 10-15 minutes for first-time setup

### Check if Python is Installed
Open Command Prompt (Windows) or Terminal (Mac/Linux) and type:
```bash
python --version
```

If you see something like `Python 3.8.0` or higher, you're good!

If not, download Python from: https://www.python.org/downloads/

**IMPORTANT FOR WINDOWS**: When installing, check the box that says "Add Python to PATH"

---

## Quick Start (If Everything is Already Set Up)

If you've already run this project before and just want to start it again:

### Option 1: Run Everything with One Command (Windows)
1. Open Command Prompt
2. Navigate to project folder:
   ```cmd
   cd "c:\Users\User\Desktop\New folder"
   ```
3. Double-click `START_PROJECT.bat`

### Option 2: Run Manually
```bash
# Train the models (if not already trained)
jupyter notebook traffic_analysis.ipynb

# OR run the web app directly
streamlit run app.py
```

---

## Detailed Setup (First Time)

### Step 1: Open Command Prompt / Terminal

**Windows**:
1. Press `Windows Key + R`
2. Type `cmd` and press Enter

**Mac**:
1. Press `Command + Space`
2. Type `terminal` and press Enter

**Linux**:
- Press `Ctrl + Alt + T`

### Step 2: Navigate to Project Folder

```bash
cd "c:\Users\User\Desktop\New folder"
```

**Tip**: You can drag the folder into the terminal window to auto-fill the path!

### Step 3: Verify Files Are Present

Type:
```bash
dir        # Windows
ls         # Mac/Linux
```

You should see:
- `traffic_accidents.csv` ← The data file
- `traffic_analysis.ipynb` ← Jupyter notebook
- `app.py` ← Web application
- `requirements.txt` ← List of packages needed
- `README.md` ← Project documentation

If any file is missing, download the complete project again!

### Step 4: Install Required Packages

This installs all the Python libraries the project needs:

```bash
pip install -r requirements.txt
```

**What gets installed**:
- pandas (data handling)
- numpy (math operations)
- scikit-learn (machine learning)
- matplotlib (creating charts)
- seaborn (pretty visualizations)
- plotly (interactive graphs)
- streamlit (web app framework)
- jupyter (notebook interface)
- Pillow (image handling)

**This will take 2-5 minutes** depending on your internet speed.

### Step 5: Verify Installation

Check if everything installed correctly:

```bash
python -c "import pandas, numpy, sklearn, matplotlib, streamlit; print('All packages installed successfully!')"
```

If you see "All packages installed successfully!", you're ready!

If you see errors, see the [Troubleshooting](#troubleshooting) section.

---

## Running the Notebook

The notebook trains the machine learning models and creates visualizations.

### Method 1: Using Jupyter Notebook (Recommended)

1. **Start Jupyter**:
   ```bash
   jupyter notebook
   ```

2. **What happens**:
   - Your browser will open automatically
   - You'll see a file browser
   - Click on `traffic_analysis.ipynb`

3. **Run the notebook**:
   - Click on the first cell
   - Press `Shift + Enter` to run it
   - Keep pressing `Shift + Enter` for each cell
   - OR click `Kernel` → `Restart & Run All` to run everything

4. **What you'll see**:
   - Data loading messages
   - Lots of colorful charts
   - Model training progress
   - Accuracy scores
   - "All models trained successfully!"

5. **How long it takes**:
   - About 30-60 seconds total
   - Most time is spent on creating visualizations

6. **Files created**:
   After running, you'll see new files:
   - `all_models.pkl`
   - `best_model.pkl`
   - `label_encoders.pkl`
   - `feature_columns.pkl`
   - `model_metrics.csv`
   - `feature_importance.csv`
   - Several `.png` image files

### Method 2: Using Jupyter Lab (Alternative)

```bash
jupyter lab
```
Similar interface, more features. Same steps as above.

### Important Notes

**Cell Execution Order Matters!**
- Run cells from top to bottom
- Don't skip cells
- If you get an error, restart and run all cells again

**To Restart**:
- Click `Kernel` → `Restart & Clear Output`
- Then run all cells again from the top

---

## Running the Web App

The web app lets you interact with the trained models through a browser.

### Prerequisites
**IMPORTANT**: You must run the notebook first! The web app needs the `.pkl` files created by the notebook.

If you see "Model not found" error, go back and run the notebook.

### Step 1: Start the App

In your terminal/command prompt:

```bash
streamlit run app.py
```

### Step 2: What Happens

You'll see messages like:
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.1.5:8501
```

### Step 3: Open in Browser

- Your browser should open automatically
- If not, manually go to: `http://localhost:8501`
- You'll see the Traffic Accidents Analysis homepage

### Step 4: Explore the App

The app has 5 pages (use the sidebar to navigate):

#### 1. 🏠 Home Page
- Overview of the dataset
- Key statistics
- Project information

#### 2. 📈 Data Exploration
- View the data table
- Interactive charts:
  - Crash severity distribution
  - Accidents by hour
  - Accidents by day of week
  - Weather conditions
  - Crash types
  - Monthly patterns
  - Lighting conditions

**Try it**: Select different visualizations from the dropdown!

#### 3. 🔍 Insights
- Peak accident times
- Environmental factors
- Contributing causes
- Injury statistics
- Feature importance chart

**What you learn**: When and why accidents happen most

#### 4. 🤖 Prediction (The Cool Part!)
- **Select a model**: Choose Decision Tree or Logistic Regression
- **Fill in the form**:
  - Select weather condition
  - Choose lighting
  - Pick crash type
  - Set time of day
  - Etc.
- **Click "Predict Crash Severity"**
- **See results**:
  - Predicted severity level
  - Confidence percentage
  - Probability for all severity levels
  - Risk interpretation

**Try different scenarios**:
- Clear weather + daytime + straight road = Usually minor
- Rain + night + curve + high speed = Often severe
- Snow + dark + multiple vehicles = Very dangerous

#### 5. 📊 Model Performance
- Compare both models
- See which features matter most
- View accuracy scores
- Look at visualization gallery

### Step 5: Stop the App

When you're done:
- Press `Ctrl + C` in the terminal
- Type `y` if asked to confirm

---

## Troubleshooting

### Problem 1: "Python not found" or "command not found"

**Solution**:
```bash
# Try with python3 instead
python3 --version
pip3 install -r requirements.txt
python3 -m jupyter notebook
streamlit run app.py
```

### Problem 2: "pip is not recognized"

**Solution**:
```bash
# Windows
python -m pip install -r requirements.txt

# Mac/Linux
python3 -m pip install -r requirements.txt
```

### Problem 3: "Permission denied" or "Access denied"

**Solution**:
```bash
# Windows: Run as Administrator
# Mac/Linux: Use sudo
sudo pip install -r requirements.txt
```

### Problem 4: "Module not found" when running notebook

**Solution**:
```bash
# Reinstall packages
pip install --upgrade -r requirements.txt

# Or install specific missing package
pip install pandas numpy scikit-learn matplotlib streamlit
```

### Problem 5: "Model not found" in web app

**Solution**:
1. Make sure you ran the notebook first!
2. Check if these files exist:
   - `all_models.pkl`
   - `best_model.pkl`
   - `label_encoders.pkl`
   - `feature_columns.pkl`
3. If not, run the notebook again completely

### Problem 6: Notebook won't start / Browser doesn't open

**Solution**:
```bash
# Try specifying port
jupyter notebook --port 8888

# Or try different browser
jupyter notebook --browser=chrome
jupyter notebook --browser=firefox
```

### Problem 7: Streamlit app shows errors

**Solution**:
```bash
# Clear Streamlit cache
streamlit cache clear

# Run with different port
streamlit run app.py --server.port 8502
```

### Problem 8: "CSV file not found"

**Solution**:
- Make sure `traffic_accidents.csv` is in the same folder
- Check the spelling is exactly: `traffic_accidents.csv`
- Make sure you're in the right directory:
  ```bash
  cd "c:\Users\User\Desktop\New folder"
  ```

### Problem 9: Notebook cells won't run

**Solution**:
1. Click `Kernel` → `Restart`
2. Run cells again from top to bottom
3. If still failing, click `Kernel` → `Restart & Clear Output`
4. Then `Kernel` → `Restart & Run All`

### Problem 10: Charts not showing in notebook

**Solution**:
Add this to the first cell:
```python
%matplotlib inline
```

### Still Having Problems?

1. **Check Python version**: Must be 3.8+
2. **Update pip**: `pip install --upgrade pip`
3. **Create virtual environment**:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   # Then install packages
   pip install -r requirements.txt
   ```

---

## What Each File Does

### Input Files (You Need These)

**`traffic_accidents.csv`**
- The raw data with accident records
- Must be present before running anything

**`requirements.txt`**
- List of Python packages needed
- Used by: `pip install -r requirements.txt`

### Code Files (The Programs)

**`traffic_analysis.ipynb`**
- Jupyter notebook for training models
- Run this FIRST
- Creates all the `.pkl` and visualization files

**`app.py`**
- Streamlit web application
- Run AFTER training models
- Uses the `.pkl` files created by notebook

**`README.md`**
- Project documentation
- Explains what the project does

**`PROJECT_EXPLANATION.md`**
- Detailed explanation of everything
- Read this to understand the concepts

**`RUN_PROJECT.md`** (This file!)
- Step-by-step instructions
- You're reading it now!

### Output Files (Created by Running Notebook)

**Model Files** (Binary files that store trained models):
- `all_models.pkl` - Both trained models (Decision Tree + Logistic Regression)
- `best_model.pkl` - The best performing model
- `label_encoders.pkl` - How to convert text to numbers and back
- `feature_columns.pkl` - Which columns to use for predictions

**Data Files** (CSV files with results):
- `model_metrics.csv` - Performance comparison of both models
- `feature_importance.csv` - Which features matter most

**Visualization Files** (PNG images):
- `comprehensive_analysis.png` - Dashboard of 9 charts
- `model_comparison.png` - Comparing model performance
- `confusion_matrices.png` - Where models make mistakes
- `feature_importance.png` - Feature importance bar chart

---

## Complete Workflow Diagram

```
1. Install Python (3.8+)
        ↓
2. Open Terminal/Command Prompt
        ↓
3. Navigate to project folder
   cd "c:\Users\User\Desktop\New folder"
        ↓
4. Install packages
   pip install -r requirements.txt
        ↓
5. Train models (Choose one):
   
   Option A: Jupyter Notebook (Interactive)
   jupyter notebook traffic_analysis.ipynb
   → Run all cells (Shift+Enter)
   → Wait ~1 minute
   → See .pkl files created
   
        ↓
        
6. Run web app:
   streamlit run app.py
   → Browser opens automatically
   → Explore 5 pages
   → Make predictions!
```

---

## Common Scenarios

### Scenario 1: First Time Setup
```bash
# Step 1: Check Python
python --version

# Step 2: Go to folder
cd "c:\Users\User\Desktop\New folder"

# Step 3: Install packages
pip install -r requirements.txt

# Step 4: Train models
jupyter notebook traffic_analysis.ipynb
# Run all cells in browser

# Step 5: Launch app
streamlit run app.py
```

### Scenario 2: Already Set Up, Just Want to Run
```bash
cd "c:\Users\User\Desktop\New folder"
streamlit run app.py
```

### Scenario 3: Want to Retrain Models
```bash
cd "c:\Users\User\Desktop\New folder"
jupyter notebook traffic_analysis.ipynb
# Kernel → Restart & Run All
```

### Scenario 4: Presentation Day
```bash
# 1. Make sure models are trained (check for .pkl files)
# 2. Start the app
streamlit run app.py
# 3. Go to Prediction page
# 4. Show different scenarios
# 5. Explain the results
```

---

## Tips for Success

### Before Running
- [ ] Check Python version (3.8+)
- [ ] Verify all files are present
- [ ] Close other programs to free memory
- [ ] Have stable internet for first-time setup

### During Setup
- [ ] Install packages without errors
- [ ] Run notebook cells in order
- [ ] Wait for each cell to complete
- [ ] Check that .pkl files are created

### When Using Web App
- [ ] Try different weather conditions
- [ ] Change time of day
- [ ] Test extreme scenarios
- [ ] Look at probability distributions
- [ ] Compare both models

### For Presentations
- [ ] Test everything beforehand
- [ ] Prepare interesting scenarios
- [ ] Know your accuracy scores
- [ ] Understand feature importance
- [ ] Have backup screenshots

---

## Time Estimates

| Task | First Time | Subsequent Times |
|------|-----------|------------------|
| Install Python | 10-15 min | Already done |
| Install packages | 3-5 min | Already done |
| Run notebook | 1-2 min | 1-2 min |
| Start web app | 10 seconds | 10 seconds |
| **Total First Time** | **15-25 min** | - |
| **Total After Setup** | - | **1-2 min** |

---

## Quick Reference Commands

### Windows
```cmd
cd "c:\Users\User\Desktop\New folder"
pip install -r requirements.txt
jupyter notebook traffic_analysis.ipynb
streamlit run app.py
```

### Mac/Linux
```bash
cd "/Users/username/Desktop/New folder"
pip3 install -r requirements.txt
jupyter notebook traffic_analysis.ipynb
streamlit run app.py
```

---

## Success Indicators

You'll know everything is working when:

✅ **Notebook**:
- All cells run without errors
- You see colorful charts
- "All models trained successfully!" appears
- New .pkl files appear in folder

✅ **Web App**:
- Browser opens to localhost:8501
- No red error messages
- Can navigate between pages
- Predictions show different results
- Charts are interactive

---

## Need Help?

If you're stuck:

1. **Check error message**: Read it carefully, often tells you what's wrong
2. **Google the error**: Copy exact error message
3. **Check file locations**: Make sure everything is in same folder
4. **Restart everything**: Close terminal, close browser, start fresh
5. **Check PROJECT_EXPLANATION.md**: Might answer your question

---

## You're Ready!

Follow the steps above and you'll have the project running in no time.

**Remember**:
1. Run notebook FIRST (trains models)
2. Then run web app (uses trained models)
3. Have fun exploring the predictions!

Good luck! 🚀
