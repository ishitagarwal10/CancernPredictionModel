import streamlit as st # type: ignore
import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import LabelEncoder # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
import joblib # type: ignore

def load_data():
    """Load and preprocess the data"""
    data = pd.read_csv('cancer.csv')
    data_cleaned = data.drop(["id", "Unnamed: 32"], axis=1)
    
    # Encode diagnosis
    le = LabelEncoder()
    data_cleaned['diagnosis'] = le.fit_transform(data_cleaned['diagnosis'])
    
    return data_cleaned

def train_model(data_cleaned):
    """Train and save the model"""
    X = data_cleaned.drop('diagnosis', axis=1)
    y = data_cleaned['diagnosis']
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    
    # Save model and feature names
    joblib.dump(model, 'model.pkl')
    joblib.dump(X.columns.tolist(), 'features.pkl')
    
    return model, X.columns.tolist()

def main():
    # Custom CSS styling
    st.markdown("""
        <style>
            body {
                background-color: #f0f2f6;
                font-family: 'Arial', sans-serif;
            }
            .stButton>button {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 24px;
                border-radius: 12px;
                cursor: pointer;
                font-size: 16px;
            }
            .stButton>button:hover {
                background-color: #45a049;
            }
            .stTitle {
                color: #2c3e50;
                font-size: 36px;
                text-align: center;
                margin-bottom: 20px;
            }
            .stHeader {
                color: #34495e;
                font-size: 28px;
                margin-bottom: 10px;
            }
            .stInfo, .stError, .stWarning {
                border-radius: 10px;
                padding: 15px;
                margin-bottom: 20px;
            }
            .stInfo {
                background-color: #dff9fb;
                color: #0984e3;
            }
            .stError {
                background-color: #ffcccc;
                color: #c0392b;
            }
            .stWarning {
                background-color: #fff3cd;
                color: #856404;
            }
            .stTabs {
                background-color: #ecf0f1;
                padding: 15px;
                border-radius: 10px;
            }
            .stNumberInput label {
                font-weight: bold;
                color: #2c3e50;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("Cancer Detection Assistant")
    
    # Load data and train model
    try:
        data_cleaned = load_data()
        try:
            # Try to load existing model and features
            model = joblib.load('model.pkl')
            features = joblib.load('features.pkl')
        except:
            print()
        
        # Educational section
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.info("### Benign Tumor\n"
                    "- NOT cancerous\n"
                    "- Doesn't spread to other parts\n"
                    "- Usually removable\n"
                    "- Rarely life threatening")
        
        with col2:
            st.error("### Malignant Tumor\n"
                     "- IS cancerous\n"
                     "- Can spread to other parts\n"
                     "- Requires immediate attention\n"
                     "- Needs quick treatment")
        
        # Input section
        st.markdown("---")
        st.header("Enter Measurements")
        
        # Create organized input sections
        user_inputs = {}
        
        # Create tabs for different measurement categories
        tabs = st.tabs(["Size Measurements", "Texture Features", "Other Features"])
        
        # Get mean values from cleaned data for defaults
        X = data_cleaned.drop('diagnosis', axis=1)
        
        with tabs[0]:
            col1, col2 = st.columns(2)
            with col1:
                for feature in ['radius_mean', 'perimeter_mean', 'area_mean']:
                    user_inputs[feature] = st.number_input(
                        f"{feature.replace('_', ' ').title()}", 
                        value=float(X[feature].mean())
                    )
            with col2:
                for feature in ['radius_worst', 'perimeter_worst', 'area_worst']:
                    user_inputs[feature] = st.number_input(
                        f"{feature.replace('_', ' ').title()}", 
                        value=float(X[feature].mean())
                    )
        
        with tabs[1]:
            col1, col2 = st.columns(2)
            with col1:
                for feature in ['texture_mean', 'smoothness_mean', 'compactness_mean']:
                    user_inputs[feature] = st.number_input(
                        f"{feature.replace('_', ' ').title()}", 
                        value=float(X[feature].mean())
                    )
            with col2:
                for feature in ['texture_worst', 'smoothness_worst', 'compactness_worst']:
                    user_inputs[feature] = st.number_input(
                        f"{feature.replace('_', ' ').title()}", 
                        value=float(X[feature].mean())
                    )
        
        with tabs[2]:
            remaining_features = [f for f in features if f not in user_inputs]
            col1, col2 = st.columns(2)
            for i, feature in enumerate(remaining_features):
                with col1 if i % 2 == 0 else col2:
                    user_inputs[feature] = st.number_input(
                        f"{feature.replace('_', ' ').title()}", 
                        value=float(X[feature].mean())
                    )
        
        # Make prediction
        if st.button("Get Results", type="primary"):
            input_vector = [user_inputs[feature] for feature in features]
            prediction = model.predict([input_vector])[0]
            
            st.markdown("---")
            if prediction == 0:
                st.success("### 🟢 Result: Likely Benign\n"
                          "The tumor characteristics suggest it is likely benign (non-cancerous).")
            else:
                st.error("### 🔴 Result: Likely Malignant\n"
                        "The tumor characteristics suggest it is likely malignant (cancerous).")

            st.warning("\u26a0\ufe0f **Important:** This is only a screening tool. Please consult with a "
                      "healthcare professional for proper diagnosis and treatment.")
            
    except FileNotFoundError:
        st.error("Error: Please make sure 'cancer.csv' is in the same directory as the application.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
