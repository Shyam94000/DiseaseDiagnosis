import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb


CHOSEN_MODEL = "NAIVE_BAYES"

def download_and_load_data():
    local_file_path = "/medquad.csv"
    print(f"Attempting to load '{local_file_path}'...")
    if os.path.exists(local_file_path):
        try:
            print(f"Found '{local_file_path}'. Loading into DataFrame...")
            df = pd.read_csv(local_file_path)
            print("File loaded successfully.")
            return df
        except Exception as e:
            print(f"An error occurred while loading '{local_file_path}': {e}")
            return None
    else:
        print(f"Error: '{local_file_path}' not found.")
        print("\nPlease manually download 'medquad.csv' from a source like Kaggle:")
        print("Example Kaggle link: https://www.kaggle.com/datasets/jpmiller/layoutlm (Note: This link is for a dataset that *contains* medquad.csv, you might need to find a direct source or use this one)")
        print("1. Download 'medquad.csv'.")
        print("2. Upload 'medquad.csv' to the same directory as this script, or if using Google Colab, upload it to your Colab session files.")
        print("   (In Colab: click the folder icon on the left, then the 'Upload to session storage' button).")
        print("3. Re-run this script after uploading the file.")
        return None

def preprocess_data(df, top_n_classes=100):
    if df is None:
        return None, None, None, None

    print("\n--- Initial Data Info ---")
    df.info()
    print("\n--- Missing values before preprocessing ---")
    print(df[['question', 'focus_area']].isnull().sum())

    df_processed = df[['question', 'focus_area']].copy()
    df_processed.dropna(subset=['question', 'focus_area'], inplace=True)

    print("\n--- Missing values after dropping NaNs from 'question' and 'focus_area' ---")
    print(df_processed.isnull().sum())

    if df_processed.empty:
        print("No data after dropping NaNs.")
        return None, None, None, None

    print(f"\n--- Filtering to keep top {top_n_classes} most frequent focus areas ---")
    focus_area_counts = df_processed['focus_area'].value_counts()
    top_focus_areas = focus_area_counts.nlargest(top_n_classes).index
    original_class_count_before_top_n = len(df_processed['focus_area'].unique())
    df_processed = df_processed[df_processed['focus_area'].isin(top_focus_areas)]
    print(f"Original number of unique focus areas before top N filtering: {original_class_count_before_top_n}")
    print(f"Number of unique focus areas after keeping top {top_n_classes}: {len(df_processed['focus_area'].unique())}")
    print(f"Number of samples after top N filtering: {len(df_processed)}")


    print("\n--- Filtering classes with insufficient samples for stratification ---")
    class_counts = df_processed['focus_area'].value_counts()
    MIN_SAMPLES_PER_CLASS = 2
    classes_to_keep = class_counts[class_counts >= MIN_SAMPLES_PER_CLASS].index
    original_class_count_before_min_samples = len(df_processed['focus_area'].unique())
    df_processed = df_processed[df_processed['focus_area'].isin(classes_to_keep)]
    print(f"Number of unique focus areas before min sample filtering: {original_class_count_before_min_samples}")
    print(f"Number of focus areas after filtering (>= {MIN_SAMPLES_PER_CLASS} samples): {len(classes_to_keep)}")
    print(f"Number of samples after filtering small classes: {len(df_processed)}")


    if df_processed.empty or len(df_processed['focus_area'].unique()) < 2:
        print("Not enough data or classes to proceed after filtering.")
        return None, None, None, None

    X = df_processed['question']
    y_raw = df_processed['focus_area']

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    print(f"\nTarget variable 'focus_area' encoded. Number of unique classes after all filtering and encoding: {len(label_encoder.classes_)}")
    return X, y, label_encoder, df_processed

def train_and_evaluate_model(X, y, label_encoder):
    if X is None or y is None or len(np.unique(y)) < 2:
        print("Skipping model training due to insufficient data or classes.")
        return

    class_counts = np.bincount(y)
    if np.any(class_counts < 2):
        print("Stratification constraint violated: a class has less than 2 samples even after preprocessing filter.")
        print("Skipping model training.")
        return

    if len(X) <= len(label_encoder.classes_):
        print(f"Dataset too small ({len(X)} samples for {len(label_encoder.classes_)} classes) for a meaningful train/test split. Skipping training.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1,2))
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    print(f"\nTF-IDF vectorized data shape (train): {X_train_tfidf.shape}")
    print(f"TF-IDF vectorized data shape (test): {X_test_tfidf.shape}")

    if X_test_tfidf.shape[0] == 0:
        print("Test set is empty after splitting. Cannot evaluate model. Consider a larger dataset or different test_size.")
        return

    print(f"\nUsing model: {CHOSEN_MODEL}")
    if CHOSEN_MODEL == "NAIVE_BAYES":
        model = MultinomialNB()
        print("\nTraining Multinomial Naive Bayes model...")
    elif CHOSEN_MODEL == "LOGISTIC_REGRESSION":
        model = LogisticRegression(solver='saga', multi_class='multinomial', random_state=42, max_iter=100, C=1.0)
        print("\nTraining Logistic Regression model...")
    elif CHOSEN_MODEL == "XGBOOST":
        model = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=len(label_encoder.classes_),
            eval_metric='mlogloss',
            use_label_encoder=False,
            random_state=42,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=7,
            tree_method='hist'
        )
        print("\nTraining XGBoost model...")
    else:
        print(f"Error: Unknown model '{CHOSEN_MODEL}' chosen.")
        return

    model.fit(X_train_tfidf, y_train)

    print("\nMaking predictions...")
    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n--- Model Evaluation ({CHOSEN_MODEL}) ---")
    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    try:
        present_labels_encoded = np.unique(np.concatenate((y_test, y_pred)))
        valid_indices_for_report = [l for l in present_labels_encoded if l < len(label_encoder.classes_)]
        target_names_report = [label_encoder.classes_[i] for i in valid_indices_for_report]
        report = classification_report(y_test, y_pred, labels=valid_indices_for_report, target_names=target_names_report, zero_division=0)
        print(report)
    except Exception as e:
        print(f"Could not generate full classification report with names: {e}")
        print("Generating report with numerical labels instead:")
        print(classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    num_classes_to_display_cm = min(len(label_encoder.classes_), 20)
    unique_labels_in_y_test_and_pred = np.unique(np.concatenate((y_test, y_pred)))


    valid_labels_for_cm_plot = sorted([l for l in unique_labels_in_y_test_and_pred if l < len(label_encoder.classes_)])


    if not valid_labels_for_cm_plot:
        print("No valid classes in y_test or y_pred to display in confusion matrix after ensuring consistency with label_encoder.")
        plt.close()
        return

    labels_to_plot_indices = valid_labels_for_cm_plot[:num_classes_to_display_cm]

    if not labels_to_plot_indices:
        print("No classes selected for confusion matrix display.")
        plt.close()
        return

    labels_to_plot_indices = [l for l in labels_to_plot_indices if l < len(label_encoder.classes_)]
    if not labels_to_plot_indices:
        print("No valid labels to plot in confusion matrix after final check.")
        plt.close()
        return

    display_labels_names_cm = label_encoder.inverse_transform(labels_to_plot_indices)


    cm_filtered = confusion_matrix(y_test, y_pred, labels=labels_to_plot_indices)


    sns.heatmap(cm_filtered, annot=True, fmt='d', cmap='Blues',
                  xticklabels=display_labels_names_cm, yticklabels=display_labels_names_cm,
                  annot_kws={"size": 8})
    plt.title(f'Confusion Matrix ({CHOSEN_MODEL} - Subset of up to {num_classes_to_display_cm} Focus Areas)')
    plt.ylabel('Actual Focus Area')
    plt.xlabel('Predicted Focus Area')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    CHOSEN_MODEL = "NAIVE_BAYES"
    # CHOSEN_MODEL = "LOGISTIC_REGRESSION"
    # CHOSEN_MODEL = "XGBOOST"

    df_medquad = download_and_load_data()
    if df_medquad is not None:
        X_data, y_data, le, df_processed = preprocess_data(df_medquad, top_n_classes=100)
        if X_data is not None and y_data is not None and le is not None and df_processed is not None:
            print(f"\nNumber of samples after preprocessing: {len(X_data)}")
            print(f"Number of unique focus areas to predict: {len(le.classes_)}")

            value_counts = df_processed['focus_area'].value_counts()
            print("\nFocus Area Distribution (Top 10 after all filtering):")
            print(value_counts.head(10))

            # --- Start of new plotting code (no comments) ---
            if not df_processed.empty and 'focus_area' in df_processed.columns:
                plt.figure(figsize=(12, 8))
                order = df_processed['focus_area'].value_counts().index
                sns.countplot(y=df_processed['focus_area'], order=order)
                plt.title(f'Distribution of Focus Areas (Total: {len(df_processed["focus_area"].unique())} classes)')
                plt.xlabel('Number of Questions')
                plt.ylabel('Focus Area')
                plt.tight_layout()
                plt.show()
            else:
                print("Skipping focus area countplot as df_processed is empty or 'focus_area' column is missing.")

            if not df_processed.empty and 'question' in df_processed.columns:
                df_processed_copy = df_processed.copy()
                df_processed_copy['question_length_words'] = df_processed_copy['question'].astype(str).apply(lambda x: len(x.split()))
                plt.figure(figsize=(10, 6))
                sns.histplot(df_processed_copy['question_length_words'], kde=True, bins=50)
                plt.title('Distribution of Question Lengths (Number of Words)')
                plt.xlabel('Number of Words')
                plt.ylabel('Frequency')
                plt.show()
            else:
                print("Skipping question length histogram as df_processed is empty or 'question' column is missing.")
            # --- End of new plotting code ---

            if len(X_data) > 10 and len(le.classes_) >= 2 :
                train_and_evaluate_model(X_data, y_data, le)
            else:
                print("\nNot enough data or classes (need at least 2 classes with sufficient samples) to proceed with model training and evaluation.")
        else:
            print("Data preprocessing failed or yielded insufficient data/classes.")
    else:
        print("Failed to load data. Exiting.")
