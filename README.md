# Traffic Accidents Analysis & Prediction

This is my data science project where I built a machine learning model to predict traffic accident severity. I used a Decision Tree classifier and created an interactive web app to explore the data.

## What This Project Does

I analyzed traffic accident data to find patterns and predict how severe crashes might be based on things like weather, lighting, road conditions, and time of day. The project has:

- Data visualizations to explore accident patterns
- A Decision Tree model that predicts crash severity
- A web app built with Streamlit for predictions
- Insights about what causes accidents

## Dataset

I'm using a CSV file called `traffic_accidents.csv` with accident records. It has information about:

- When accidents happened (date, hour, day of week, month)
- Environmental conditions (weather, lighting, road surface)
- Accident details (crash type, damage, what caused it)
- Location info (traffic control devices, road type)
- Injury data (total injuries, severity levels)

## How to Run This

### What You Need

- Python 3.8 or newer
- pip installed

### Setup Steps

1. Download all the project files

2. Open command prompt and go to the project folder:
   ```cmd
   cd "c:\Users\User\Desktop\New folder"
   ```

3. Install the required libraries:
   ```cmd
   pip install -r requirements.txt
   ```

   This installs:
   - pandas and numpy for data handling
   - scikit-learn for machine learning
   - matplotlib and seaborn for charts
   - plotly for interactive graphs
   - streamlit for the web app
   - Pillow for images

## How to Use

### Step 1: Train the Model

I recommend using the Jupyter notebook because you can see everything step by step:

```cmd
jupyter notebook traffic_analysis.ipynb
```

Just run each cell and you'll see the results right away. It's easier to understand what's happening.

If you want to run everything at once without stopping, you can use the Python script instead:
```cmd
python train_model.py
```

The training process:
- Loads the accident data
- Cleans and prepares the features
- Trains the Decision Tree model
- Tests how well it works
- Saves everything
- Creates visualization charts

**What gets created:**
- Model files (.pkl) - these are the trained models
- CSV files with metrics
- PNG images with charts

**Takes about**: 10-30 seconds

### Step 2: Run the Web App

After training, start the web app:

```cmd
streamlit run app.py
```

It should open in your browser automatically at `http://localhost:8501`

If it doesn't open, just type that URL in your browser.

## What's in the Web App

### Home Page
- Shows basic info about the dataset
- Some key statistics

### Data Exploration
- Charts showing:
  - When accidents happen most
  - Weather conditions during crashes
  - Types of crashes
  - Monthly patterns

### Insights
- Patterns I found in the data
- What environmental factors matter
- Main causes of accidents
- Injury statistics

### Prediction Tool
- Form where you can enter accident conditions
- The model predicts how severe it might be
- Shows confidence scores

### Model Performance
- How well the model works
- Which features are most important
- Charts and metrics

## The Machine Learning Model

I used a Decision Tree Classifier because:
- It's easy to understand how it makes decisions
- Good for beginners, not too complicated
- Works with different types of data
- Trains pretty fast
- Doesn't need data scaling

### Settings I Used
```python
DecisionTreeClassifier(
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
```

### Features Used for Prediction
The model uses 12 key features:
1. Traffic control device
2. Weather condition
3. Lighting condition
4. First crash type
5. Roadway surface condition
6. Crash type classification
7. Damage estimate
8. Primary contributing cause
9. Number of units involved
10. Crash hour
11. Day of week
12. Month

### Target Variable
**most_severe_injury** - Predicts the severity level:
- NO INDICATION OF INJURY
- REPORTED, NOT EVIDENT
- NONINCAPACITATING INJURY
- INCAPACITATING INJURY
- FATAL

## Files in This Project

```
New folder/
│
├── traffic_accidents.csv          # Dataset
├── train_model.py                 # Model training script
├── traffic_analysis.ipynb         # Jupyter Notebook (comprehensive analysis)
├── app.py                         # Streamlit web application
├── requirements.txt               # Python dependencies
├── README.md                      # This file
│
├── traffic_model.pkl              # Trained model (generated)
├── all_models.pkl                 # All trained models (generated)
├── best_model.pkl                 # Best model (generated)
├── label_encoders.pkl             # Feature encoders (generated)
├── feature_columns.pkl            # Feature list (generated)
├── feature_importance.csv         # Importance scores (generated)
├── model_metrics.csv              # Model comparison (generated)
│
└── *.png                          # Visualization images (generated)
    ├── comprehensive_analysis.png
    ├── model_comparison.png
    ├── confusion_matrices.png
    ├── feature_importance_models.png
    ├── learning_curves.png
    ├── roc_curves.png
    └── final_summary.png
```

## Common Problems

**If you get "module not found" errors:**
Run this again:
```cmd
pip install -r requirements.txt
```

**If the app says "model not found":**
You need to train the model first:
```cmd
python train_model.py
```

**If Streamlit won't start:**
Try a different port:
```cmd
streamlit run app.py --server.port 8502
```

**If it can't find the CSV file:**
Make sure traffic_accidents.csv is in the same folder

## Results I Got

When I ran everything:

### Model Performance
- Accuracy was around 70-80%
- Training took about 20 seconds
- Predictions are instant

### Interesting Findings
- Most accidents happen during the day
- Even in clear weather, lots of accidents happen
- Rear-end crashes are most common
- Following too close is a major cause
- Rush hour = more accidents

## What I Learned

1. How to analyze real data
2. Training machine learning models
3. Building web apps with Streamlit
4. Creating visualizations
5. Making predictions with trained models

## Notes

- This was my first ML project, Decision Trees are a good start
- The accuracy isn't perfect but that's okay
- I tried changing different parameters to see what works better
- The web app makes it easy to explore the data
- Understanding the process matters more than getting perfect scores

## Future Improvements

Things I want to try:
- Testing other algorithms like Random Forest
- Adding more features
- Making the web app better
- Maybe deploy it online

## Notes

This is a class project for Introduction to Data Science. I used:
- Streamlit for the web app
- Scikit-learn for machine learning
- Matplotlib and Plotly for charts

If you have questions, check the troubleshooting section or make sure all files are in the right folder.
