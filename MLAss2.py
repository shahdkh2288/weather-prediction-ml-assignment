import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt

def load_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    print("Data Loaded Successfully")
    return df

def check_missing_data(df):
    missing_data = df.isnull().sum()
    print("Missing Data:\n", missing_data)
    return missing_data

def handle_missing_data(df):
    # Option 1: Drop rows with missing values
    df_dropped = df.dropna()
    print("\nData after dropping rows with missing values:\n", df_dropped)

    # Option 2: Fill missing values with the mean
    df_filled = df.fillna(df.mean())
    print("\nData after filling missing values with mean:\n", df_filled)

    return df_dropped, df_filled

def preprocess_data(df, target_column):
    # Encode target variable
    df[target_column] = df[target_column].map({'no rain': 0, 'rain': 1})
    
    # Feature scaling
    features = df.drop(target_column, axis=1)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    return features_scaled, df[target_column], features.columns

def train_decision_tree(X_train, y_train, max_depth=None, random_state=40):
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    print("Decision Tree Model Trained Successfully")
    return model

# Main flow
if __name__ == "__main__":
    file_path = "weather_forecast_data.csv"  # Update your file path
    target_column = 'Rain'

    # 1. Load data
    df = load_data(file_path)

    # 2. Check missing data
    check_missing_data(df)

    # 3. Handle missing data
    df_dropped, df_filled = handle_missing_data(df)

    # 4. Use the filled data for further processing
    X, y, feature_names = preprocess_data(df_filled, target_column)

    # 5. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

    # 6. Train Decision Tree
    decision_tree_model = train_decision_tree(X_train, y_train)

    
