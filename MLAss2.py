import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

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

    # Option 2: Fill missing values with the mean for numeric columns only
    numeric_cols = df.select_dtypes(include=['number']).columns
    df_filled = df.copy()
    df_filled[numeric_cols] = df_filled[numeric_cols].fillna(df_filled[numeric_cols].mean())

    print("\nData after filling missing values with mean (for numeric columns):\n", df_filled)

    return df_dropped, df_filled


def preprocess_data(df, target_column):
    # Encode target variable
    df[target_column] = df[target_column].map({'no rain': 0, 'rain': 1})

    # Separate features and target
    features = df.drop(target_column, axis=1)
    target = df[target_column]

    return features, target, features.columns


def train_decision_tree(X_train, y_train, max_depth=None, random_state=40):
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    print("Decision Tree Model Trained Successfully")
    return model

def evaluate_decision_tree(model, X_test,y_test):
    """Evaluate the model using accuracy, precision, and recall."""
    y_test_pred = model.predict(X_test)

    # 3 Metrics for Testing Data
    print("\nTesting Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.2f}")
    print(f"Precision: {precision_score(y_test, y_test_pred):.2f}")
    print(f"Recall: {recall_score(y_test, y_test_pred):.2f}")

def plot_decision_tree(model, feature_names):

    #visualize decision tree with custom font sizes and color
    plt.figure(figsize=(25, 15))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=["No Rain", "Rain"],
        filled=True,
        rounded=True,
        proportion=True,
        fontsize=14
    )
    plt.title("Decision Tree Visualization", fontsize=20, pad=20)
    plt.show()

def train_naive_bayes(X_train, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)
    print("Naive Bayes Model Trained Successfully")
    return model

def evaluate_naive_bayes(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    return accuracy, precision, recall


if __name__ == "__main__":
    file_path = "weather_forecast_data.csv"  # Update your file path
    target_column = 'Rain'

    # 1. Load data
    df = load_data(file_path)

    # 2. Check missing data
    check_missing_data(df)

    # 3. Handle missing data
    df_dropped, df_filled = handle_missing_data(df)

    #Case 1: Using data after dropping rows with missing values
    print("\nEvaluating models > knn,Naive bayes, decision tree with dropping rows with missing value option: \n")
    X_dropped, y_dropped, feature_names = preprocess_data(df_dropped, target_column)
    X_train_dropped, X_test_dropped, y_train_dropped, y_test_dropped = train_test_split(X_dropped, y_dropped,
                                                                                        test_size=0.2, random_state=40)
    # Scale features
    scaler = StandardScaler()
    X_train_scaled_dropped = scaler.fit_transform(X_train_dropped)
    X_test_scaled_dropped = scaler.transform(X_test_dropped)

    #evaluate and plot decision tree model
    decision_tree_model_dropped = train_decision_tree(X_train_scaled_dropped, y_train_dropped)
    evaluate_decision_tree(decision_tree_model_dropped, X_test_scaled_dropped,y_test_dropped)
    plot_decision_tree(decision_tree_model_dropped, feature_names)
    print("\n")
    #evaluate naive bayes model
    naive_bayes_model_dropped = train_naive_bayes(X_train_scaled_dropped, y_train_dropped)
    evaluate_naive_bayes(naive_bayes_model_dropped, X_test_scaled_dropped, y_test_dropped)


    #Case 2: Using data after filling missing values with the mean
    print("\nEvaluating models > knn,Naive bayes, decision tree with filled missing values with mean option :\n")
    X_filled, y_filled, feature_names = preprocess_data(df_filled, target_column)
    X_train_filled, X_test_filled, y_train_filled, y_test_filled = train_test_split(X_filled, y_filled, test_size=0.2,
                                                                                    random_state=40)

    # Scale features
    X_train_scaled_filled = scaler.fit_transform(X_train_filled)
    X_test_scaled_filled = scaler.transform(X_test_filled)

    # evaluate and plot decision tree model
    decision_tree_model_filled = train_decision_tree(X_train_scaled_filled, y_train_filled)
    evaluate_decision_tree(decision_tree_model_filled, X_test_scaled_filled,y_test_filled)
    plot_decision_tree(decision_tree_model_filled, feature_names)

    print("\n")
    # evaluate naive bayes model
    naive_bayes_model_filled = train_naive_bayes(X_train_scaled_filled, y_train_filled)
    evaluate_naive_bayes(naive_bayes_model_filled, X_test_scaled_filled,y_test_filled)