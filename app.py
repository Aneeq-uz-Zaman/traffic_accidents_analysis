"""
Traffic Accidents Prediction - Streamlit Web Application
An interactive dashboard for exploring traffic accident data and predicting crash severity
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os

# Page configuration
st.set_page_config(
    page_title="Traffic Accidents Analysis",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 1rem;
    }
    h2 {
        color: #ff7f0e;
        padding-top: 1rem;
    }
    h3 {
        color: #2ca02c;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the traffic accidents dataset"""
    df = pd.read_csv('traffic_accidents.csv')
    df['crash_date'] = pd.to_datetime(df['crash_date'])
    return df

@st.cache_resource
def load_model_artifacts():
    """Load trained models and encoders"""
    try:
        # Try loading all models first
        try:
            with open('all_models.pkl', 'rb') as f:
                all_models = pickle.load(f)
        except:
            # Fall back to best model only
            with open('best_model.pkl', 'rb') as f:
                best_model = pickle.load(f)
            all_models = {'Decision Tree': best_model}
        
        with open('label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        with open('feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
        return all_models, label_encoders, feature_columns
    except FileNotFoundError:
        return None, None, None

def main():
    # Header
    st.title("🚗 Traffic Accidents Analysis & Prediction System")
    st.markdown("---")
    
    # Load data
    try:
        df = load_data()
    except FileNotFoundError:
        st.error("❌ Dataset not found! Please make sure 'traffic_accidents.csv' is in the same directory.")
        return
    
    # Sidebar navigation
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        ["🏠 Home", "📈 Data Exploration", "🔍 Insights", "🤖 Prediction", "📊 Model Performance"]
    )
    
    # HOME PAGE
    if page == "🏠 Home":
        st.header("Welcome to Traffic Accidents Analysis System")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Accidents", f"{len(df):,}")
        with col2:
            st.metric("Date Range", f"{df['crash_date'].dt.year.min()} - {df['crash_date'].dt.year.max()}")
        with col3:
            injury_rate = (df['injuries_total'] > 0).sum() / len(df) * 100
            st.metric("Injury Rate", f"{injury_rate:.1f}%")
        with col4:
            fatal_count = df['injuries_fatal'].sum()
            st.metric("Fatal Injuries", f"{int(fatal_count)}")
        
        st.markdown("---")
        
        st.subheader("📋 Project Overview")
        st.write("""
        This interactive application provides comprehensive analysis of traffic accident data 
        and uses machine learning to predict crash severity based on various factors.
        
        **Features:**
        - 📈 **Data Exploration**: Visualize accident patterns and trends
        - 🔍 **Insights**: Discover key factors contributing to accidents
        - 🤖 **Prediction**: Predict crash severity using our trained model
        - 📊 **Model Performance**: Evaluate the prediction model's accuracy
        """)
        
        st.subheader("📊 Dataset Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Dataset Statistics:**")
            st.write(f"- Total Records: {len(df):,}")
            st.write(f"- Total Features: {len(df.columns)}")
            st.write(f"- Date Range: {df['crash_date'].min().date()} to {df['crash_date'].max().date()}")
            st.write(f"- Total Injuries: {int(df['injuries_total'].sum())}")
            st.write(f"- Fatal Injuries: {int(df['injuries_fatal'].sum())}")
        
        with col2:
            st.write("**Key Features:**")
            st.write("- Weather conditions")
            st.write("- Lighting conditions")
            st.write("- Road surface conditions")
            st.write("- Traffic control devices")
            st.write("- Crash types and causes")
    
    # DATA EXPLORATION PAGE
    elif page == "📈 Data Exploration":
        st.header("📈 Data Exploration")
        
        # Dataset preview
        st.subheader("📋 Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("---")
        
        # Visualization selection
        st.subheader("📊 Visualizations")
        
        viz_option = st.selectbox(
            "Select a visualization:",
            ["Crash Severity Distribution", "Accidents by Hour", "Accidents by Day of Week",
             "Weather Conditions", "Crash Types", "Accidents by Month", "Lighting Conditions"]
        )
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if viz_option == "Crash Severity Distribution":
                fig = px.bar(
                    df['most_severe_injury'].value_counts().reset_index(),
                    x='most_severe_injury', y='count',
                    title="Distribution of Crash Severity",
                    labels={'most_severe_injury': 'Injury Type', 'count': 'Number of Accidents'},
                    color='count',
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_option == "Accidents by Hour":
                hourly_data = df['crash_hour'].value_counts().sort_index()
                fig = px.line(
                    x=hourly_data.index, y=hourly_data.values,
                    title="Traffic Accidents by Hour of Day",
                    labels={'x': 'Hour', 'y': 'Number of Accidents'},
                    markers=True
                )
                fig.update_traces(line_color='#ff7f0e', line_width=3)
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_option == "Accidents by Day of Week":
                days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                daily_data = df['crash_day_of_week'].value_counts().sort_index()
                fig = px.bar(
                    x=[days[i-1] for i in daily_data.index], y=daily_data.values,
                    title="Traffic Accidents by Day of Week",
                    labels={'x': 'Day', 'y': 'Number of Accidents'},
                    color=daily_data.values,
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_option == "Weather Conditions":
                weather_data = df['weather_condition'].value_counts().head(10)
                fig = px.bar(
                    x=weather_data.values, y=weather_data.index,
                    title="Top 10 Weather Conditions During Accidents",
                    labels={'x': 'Number of Accidents', 'y': 'Weather Condition'},
                    orientation='h',
                    color=weather_data.values,
                    color_continuous_scale='Greens'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_option == "Crash Types":
                crash_data = df['first_crash_type'].value_counts().head(10)
                fig = px.pie(
                    values=crash_data.values, names=crash_data.index,
                    title="Top 10 Crash Types Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_option == "Accidents by Month":
                months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                monthly_data = df['crash_month'].value_counts().sort_index()
                fig = px.bar(
                    x=[months[i-1] for i in monthly_data.index], y=monthly_data.values,
                    title="Traffic Accidents by Month",
                    labels={'x': 'Month', 'y': 'Number of Accidents'},
                    color=monthly_data.values,
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_option == "Lighting Conditions":
                lighting_data = df['lighting_condition'].value_counts()
                fig = px.pie(
                    values=lighting_data.values, names=lighting_data.index,
                    title="Lighting Conditions During Accidents"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Statistics")
            if viz_option == "Crash Severity Distribution":
                st.write(df['most_severe_injury'].value_counts())
            elif viz_option == "Accidents by Hour":
                st.write(f"Peak hour: {df['crash_hour'].mode()[0]}:00")
                st.write(f"Accidents at peak: {df['crash_hour'].value_counts().max()}")
            elif viz_option == "Weather Conditions":
                st.write(f"Most common: {df['weather_condition'].mode()[0]}")
                st.write(df['weather_condition'].value_counts().head())
            elif viz_option == "Crash Types":
                st.write(f"Most common: {df['first_crash_type'].mode()[0]}")
                st.write(df['first_crash_type'].value_counts().head())
    
    # INSIGHTS PAGE
    elif page == "🔍 Insights":
        st.header("🔍 Key Insights")
        
        # Calculate insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⏰ Time-Based Insights")
            
            # Peak hour
            peak_hour = df['crash_hour'].mode()[0]
            st.info(f"🕐 **Peak Accident Hour**: {peak_hour}:00 with {df[df['crash_hour']==peak_hour].shape[0]} accidents")
            
            # Peak day
            days = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 
                   5: 'Friday', 6: 'Saturday', 7: 'Sunday'}
            peak_day = df['crash_day_of_week'].mode()[0]
            st.info(f"📅 **Peak Accident Day**: {days.get(peak_day, 'Unknown')} with {df[df['crash_day_of_week']==peak_day].shape[0]} accidents")
            
            # Peak month
            months = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                     7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
            peak_month = df['crash_month'].mode()[0]
            st.info(f"📆 **Peak Accident Month**: {months.get(peak_month, 'Unknown')} with {df[df['crash_month']==peak_month].shape[0]} accidents")
        
        with col2:
            st.subheader("🌦️ Environmental Insights")
            
            # Weather
            top_weather = df['weather_condition'].mode()[0]
            weather_count = df[df['weather_condition']==top_weather].shape[0]
            st.info(f"☀️ **Most Common Weather**: {top_weather} ({weather_count} accidents)")
            
            # Lighting
            top_lighting = df['lighting_condition'].mode()[0]
            lighting_count = df[df['lighting_condition']==top_lighting].shape[0]
            st.info(f"💡 **Most Common Lighting**: {top_lighting} ({lighting_count} accidents)")
            
            # Road surface
            top_surface = df['roadway_surface_cond'].mode()[0]
            surface_count = df[df['roadway_surface_cond']==top_surface].shape[0]
            st.info(f"🛣️ **Most Common Road Surface**: {top_surface} ({surface_count} accidents)")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🚦 Contributing Factors")
            
            # Primary cause
            top_cause = df['prim_contributory_cause'].value_counts().head(5)
            st.write("**Top 5 Contributing Causes:**")
            for i, (cause, count) in enumerate(top_cause.items(), 1):
                st.write(f"{i}. {cause}: {count} accidents")
        
        with col2:
            st.subheader("💥 Crash Characteristics")
            
            # Crash types
            top_crashes = df['first_crash_type'].value_counts().head(5)
            st.write("**Top 5 Crash Types:**")
            for i, (crash_type, count) in enumerate(top_crashes.items(), 1):
                st.write(f"{i}. {crash_type}: {count} accidents")
        
        st.markdown("---")
        
        st.subheader("🏥 Injury Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            no_injury = (df['most_severe_injury'] == 'NO INDICATION OF INJURY').sum()
            st.metric("No Injury", f"{no_injury:,}", f"{no_injury/len(df)*100:.1f}%")
        
        with col2:
            minor_injury = (df['most_severe_injury'].isin(['REPORTED, NOT EVIDENT', 'NONINCAPACITATING INJURY'])).sum()
            st.metric("Minor Injury", f"{minor_injury:,}", f"{minor_injury/len(df)*100:.1f}%")
        
        with col3:
            severe_injury = (df['most_severe_injury'].isin(['INCAPACITATING INJURY', 'FATAL'])).sum()
            st.metric("Severe/Fatal", f"{severe_injury:,}", f"{severe_injury/len(df)*100:.1f}%")
        
        # Feature importance visualization
        st.markdown("---")
        st.subheader("🎯 Feature Importance")
        
        if os.path.exists('feature_importance.png'):
            st.image('feature_importance.png', caption='Most Important Features for Prediction', use_container_width=True)
        else:
            st.warning("⚠️ Feature importance chart not found. Please run the notebook first.")
    
    # PREDICTION PAGE
    elif page == "🤖 Prediction":
        st.header("🤖 Crash Severity Prediction")
        
        # Load models
        all_models, label_encoders, feature_columns = load_model_artifacts()
        
        if all_models is None:
            st.error("❌ Model not found! Please run the notebook first to train the model.")
            st.code("jupyter notebook traffic_analysis.ipynb", language="bash")
            return
        
        st.success(f"✅ {len(all_models)} model(s) loaded successfully!")
        
        # Model selection
        selected_model_name = st.selectbox(
            "Select Model:",
            list(all_models.keys())
        )
        model = all_models[selected_model_name]
        
        st.write("Fill in the accident details below to predict the crash severity:")
        
        # Create input form
        col1, col2, col3 = st.columns(3)
        
        with col1:
            traffic_control = st.selectbox(
                "Traffic Control Device",
                df['traffic_control_device'].unique()
            )
            
            weather = st.selectbox(
                "Weather Condition",
                df['weather_condition'].unique()
            )
            
            lighting = st.selectbox(
                "Lighting Condition",
                df['lighting_condition'].unique()
            )
            
            crash_type = st.selectbox(
                "First Crash Type",
                df['first_crash_type'].unique()
            )
        
        with col2:
            road_surface = st.selectbox(
                "Roadway Surface Condition",
                df['roadway_surface_cond'].unique()
            )
            
            crash_classification = st.selectbox(
                "Crash Type",
                df['crash_type'].unique()
            )
            
            damage = st.selectbox(
                "Damage Estimate",
                df['damage'].unique()
            )
            
            cause = st.selectbox(
                "Primary Contributing Cause",
                df['prim_contributory_cause'].unique()
            )
        
        with col3:
            num_units = st.number_input(
                "Number of Units Involved",
                min_value=1, max_value=10, value=2
            )
            
            crash_hour = st.slider(
                "Crash Hour",
                min_value=0, max_value=23, value=12
            )
            
            crash_day = st.slider(
                "Day of Week (1=Mon, 7=Sun)",
                min_value=1, max_value=7, value=3
            )
            
            crash_month = st.slider(
                "Month",
                min_value=1, max_value=12, value=6
            )
        
        # Predict button
        if st.button("🔮 Predict Crash Severity", type="primary", use_container_width=True):
            # Prepare input data
            input_data = {
                'traffic_control_device': traffic_control,
                'weather_condition': weather,
                'lighting_condition': lighting,
                'first_crash_type': crash_type,
                'roadway_surface_cond': road_surface,
                'crash_type': crash_classification,
                'damage': damage,
                'prim_contributory_cause': cause,
                'num_units': num_units,
                'crash_hour': crash_hour,
                'crash_day_of_week': crash_day,
                'crash_month': crash_month
            }
            
            # Encode input
            input_encoded = []
            for col in feature_columns:
                if col in label_encoders:
                    try:
                        encoded_value = label_encoders[col].transform([str(input_data[col])])[0]
                        input_encoded.append(encoded_value)
                    except:
                        # If value not seen during training, use mode
                        input_encoded.append(0)
                else:
                    input_encoded.append(input_data[col])
            
            # Make prediction
            prediction = model.predict([input_encoded])[0]
            prediction_proba = model.predict_proba([input_encoded])[0]
            
            # Decode prediction
            predicted_severity = label_encoders['most_severe_injury'].inverse_transform([prediction])[0]
            confidence = prediction_proba[prediction] * 100
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Predicted Severity", predicted_severity)
                st.metric("Confidence", f"{confidence:.1f}%")
            
            with col2:
                # Show probability for all classes
                st.write("**Probability Distribution:**")
                all_classes = label_encoders['most_severe_injury'].classes_
                prob_df = pd.DataFrame({
                    'Severity': all_classes,
                    'Probability': prediction_proba * 100
                }).sort_values('Probability', ascending=False)
                
                st.dataframe(prob_df.style.format({'Probability': '{:.2f}%'}), use_container_width=True)
            
            # Severity interpretation
            st.markdown("---")
            st.subheader("💡 Interpretation")
            
            if predicted_severity == "NO INDICATION OF INJURY":
                st.success("✅ Low risk: No injuries expected. Minor property damage likely.")
            elif predicted_severity in ["REPORTED, NOT EVIDENT"]:
                st.info("ℹ️ Low-Medium risk: Minor injuries possible but not evident.")
            elif predicted_severity == "NONINCAPACITATING INJURY":
                st.warning("⚠️ Medium risk: Non-severe injuries expected. Medical attention recommended.")
            elif predicted_severity == "INCAPACITATING INJURY":
                st.error("🚨 High risk: Severe injuries expected. Immediate medical attention required.")
            else:
                st.error("☠️ Critical risk: Fatal injuries possible. Emergency response needed.")
    
    # MODEL PERFORMANCE PAGE
    elif page == "📊 Model Performance":
        st.header("📊 Model Performance Analysis")
        
        all_models, label_encoders, feature_columns = load_model_artifacts()
        
        if all_models is None:
            st.error("❌ Model not found! Please run the notebook first to train the model.")
            st.code("jupyter notebook traffic_analysis.ipynb", language="bash")
            return
        
        st.success(f"✅ {len(all_models)} model(s) loaded successfully!")
        
        # Model info
        st.subheader("🤖 Model Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Models Trained", len(all_models))
            st.write("**Available Models:**")
            for model_name in all_models.keys():
                st.write(f"- {model_name}")
        with col2:
            st.metric("Features Used", len(feature_columns))
        with col3:
            st.metric("Target Classes", len(label_encoders['most_severe_injury'].classes_))
        
        st.markdown("---")
        
        # Show model metrics if available
        if os.path.exists('model_metrics.csv'):
            st.subheader("📊 Model Accuracy Comparison")
            metrics_df = pd.read_csv('model_metrics.csv')
            
            col1, col2 = st.columns([2, 1])
            with col1:
                fig = px.bar(
                    metrics_df,
                    x='Model', y='Accuracy',
                    title="Model Accuracy Comparison",
                    color='Accuracy',
                    color_continuous_scale='Blues',
                    text='Accuracy'
                )
                fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("**Model Scores:**")
                for _, row in metrics_df.iterrows():
                    st.write(f"**{row['Model']}**")
                    st.write(f"Accuracy: {row['Accuracy']:.2%}")
                    st.write(f"F1-Score: {row['F1-Score']:.4f}")
                    st.write("---")
        
        st.markdown("---")
        
        # Display feature importance
        st.subheader("🎯 Feature Importance")
        
        if os.path.exists('feature_importance.csv'):
            feature_imp = pd.read_csv('feature_importance.csv')
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = px.bar(
                    feature_imp.head(10),
                    x='importance', y='feature',
                    orientation='h',
                    title="Top 10 Most Important Features",
                    color='importance',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.write("**Top 5 Features:**")
                for i, row in feature_imp.head(5).iterrows():
                    st.write(f"{i+1}. {row['feature']}")
                    st.progress(row['importance'])
        else:
            st.warning("⚠️ Feature importance data not found.")
        
        st.markdown("---")
        
        # Visualizations
        st.subheader("📈 Model Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if os.path.exists('severity_distribution.png'):
                st.image('severity_distribution.png', caption='Target Distribution', use_container_width=True)
        
        with col2:
            if os.path.exists('crashes_by_hour.png'):
                st.image('crashes_by_hour.png', caption='Temporal Patterns', use_container_width=True)
        
        # Model description
        st.markdown("---")
        st.subheader("📝 Model Description")
        
        if 'Decision Tree' in all_models:
            st.write("""
            **Decision Tree Classifier**
            - ✅ Easy to interpret and understand
            - ✅ Handles both categorical and numerical features
            - ✅ Requires minimal data preprocessing
            - ✅ Can capture non-linear relationships
            
            **Parameters:**
            - Maximum depth: 5
            - Minimum samples per split: 20
            - Minimum samples per leaf: 10
            """)
        
        if 'Logistic Regression' in all_models:
            st.write("""
            **Logistic Regression**
            - ✅ Fast and efficient for linear relationships
            - ✅ Provides probability estimates
            - ✅ Works well with multiple classes
            - ✅ Less prone to overfitting
            
            **Parameters:**
            - Max iterations: 500
            - Solver: lbfgs
            """)

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: gray; padding: 1rem;'>
            <p>🚗 Traffic Accidents Analysis System | Built with Streamlit & Scikit-learn</p>
            <p>Data Science Project - Introduction to Data Science</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
