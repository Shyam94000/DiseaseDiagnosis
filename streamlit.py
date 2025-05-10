import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import plot_partial_dependence # Changed from partial_dependence
import io # For capturing df.info()

# --- Backend Logic (Adapted from your script) ---

def download_and_load_data():
    # For a Streamlit app, you might want to use st.file_uploader
    # For now, we'll keep the local file path logic
    local_file_path = "medquad.csv" # Assuming it's in the same directory or provide full path
    
    # Check if file exists for Streamlit deployment (or local run)
    if not os.path.exists(local_file_path):
        st.error(f"Error: '{local_file_path}' not found.")
        st.info("Please ensure 'medquad.csv' is in the same directory as the Streamlit app, or modify the path.")
        st.info("You can download it from a source like Kaggle (e.g., https://www.kaggle.com/datasets/jpmiller/layoutlm - note: this link is for a dataset that *contains* medquad.csv, you might need to find a direct source or use this one).")
        return None

    try:
        df = pd.read_csv(local_file_path)
        return df
    except Exception as e:
        st.error(f"An error occurred while loading '{local_file_path}': {e}")
        return None

def preprocess_data_streamlit(df):
    if df is None:
        return None, None, None, None, None

    # Capture df.info()
    buffer = io.StringIO()
    df.info(buf=buffer)
    df_info_str = buffer.getvalue()

    missing_before = df[['question', 'focus_area']].isnull().sum().to_frame('missing_before')

    df_processed = df[['question', 'focus_area']].copy()
    df_processed.dropna(subset=['question', 'focus_area'], inplace=True)

    missing_after = df_processed.isnull().sum().to_frame('missing_after') # Should be all zeros

    if df_processed.empty:
        st.warning("No data after dropping NaNs.")
        return None, None, None, None, (df_info_str, missing_before, missing_after, "No data after dropping NaNs.")


    class_counts_val = df_processed['focus_area'].value_counts()
    MIN_SAMPLES_PER_CLASS = 2
    classes_to_keep = class_counts_val[class_counts_val >= MIN_SAMPLES_PER_CLASS].index
    original_class_count = len(df_processed['focus_area'].unique())
    df_processed = df_processed[df_processed['focus_area'].isin(classes_to_keep)]
    
    filtering_log = (
        f"Original number of unique focus areas: {original_class_count}\n"
        f"Number of focus areas after filtering (>= {MIN_SAMPLES_PER_CLASS} samples): {len(classes_to_keep)}\n"
        f"Number of samples after filtering small classes: {len(df_processed)}"
    )

    if df_processed.empty or len(df_processed['focus_area'].unique()) < 2:
        st.warning("Not enough data or classes to proceed after filtering small classes.")
        return None, None, None, None, (df_info_str, missing_before, missing_after, filtering_log + "\nNot enough data/classes post-filtering.")

    X = df_processed['question']
    y_raw = df_processed['focus_area']
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    
    encoding_log = f"Target variable 'focus_area' encoded. Number of unique classes: {len(label_encoder.classes_)}"
    
    logs = (df_info_str, missing_before, missing_after, filtering_log + "\n" + encoding_log)
    return X, y, label_encoder, df_processed, logs

def generate_word_freq_heatmap_streamlit(df_processed):
    if df_processed.empty or 'focus_area' not in df_processed.columns or 'question' not in df_processed.columns:
        return None
    
    fig = None
    try:
        N_TOP_FOCUS_AREAS = 10
        N_TOP_WORDS = 15
        actual_top_n_focus_areas = min(N_TOP_FOCUS_AREAS, len(df_processed['focus_area'].unique()))
        if actual_top_n_focus_areas == 0: return None

        top_focus_areas_names = df_processed['focus_area'].value_counts().nlargest(actual_top_n_focus_areas).index
        df_top_focus = df_processed[df_processed['focus_area'].isin(top_focus_areas_names)]
        
        corpus_for_heatmap = []
        focus_area_order_for_heatmap = []
        
        for area in top_focus_areas_names:
            text = " ".join(df_top_focus[df_top_focus['focus_area'] == area]['question'].astype(str).tolist())
            if text.strip():
                corpus_for_heatmap.append(text)
                focus_area_order_for_heatmap.append(area)
        
        if corpus_for_heatmap and len(focus_area_order_for_heatmap) > 0:
            vectorizer_heatmap = CountVectorizer(stop_words='english', max_features=N_TOP_WORDS)
            word_counts_matrix = vectorizer_heatmap.fit_transform(corpus_for_heatmap)
            word_names = vectorizer_heatmap.get_feature_names_out()

            if word_counts_matrix.shape[1] > 0:
                heatmap_df = pd.DataFrame(word_counts_matrix.toarray(), index=focus_area_order_for_heatmap, columns=word_names)
                
                fig, ax = plt.subplots(figsize=(12, max(8, len(focus_area_order_for_heatmap) * 0.6)))
                sns.heatmap(heatmap_df, annot=True, cmap="YlGnBu", fmt="d", ax=ax)
                ax.set_title(f'Top {word_names.shape[0]} Word Frequencies in Top {len(focus_area_order_for_heatmap)} Focus Areas')
                ax.set_ylabel('Focus Area')
                ax.set_xlabel(f'Top Words')
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                plt.tight_layout()
    except Exception as e:
        st.error(f"Could not generate word frequency heatmap: {e}")
        fig = None
    return fig

def generate_qlen_histogram_streamlit(df_processed):
    if df_processed.empty or 'question' not in df_processed.columns:
        return None
    fig = None
    try:
        df_copy = df_processed.copy()
        df_copy['question_length_words'] = df_copy['question'].astype(str).apply(lambda x: len(x.split()))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df_copy['question_length_words'], kde=True, bins=50, ax=ax)
        ax.set_title('Distribution of Question Lengths (Number of Words)')
        ax.set_xlabel('Number of Words')
        ax.set_ylabel('Frequency')
        plt.tight_layout()
    except Exception as e:
        st.error(f"Could not generate question length histogram: {e}")
        fig = None
    return fig

def train_and_evaluate_model_streamlit(X, y, label_encoder):
    results = {}
    if X is None or y is None or len(np.unique(y)) < 2:
        st.warning("Skipping model training: insufficient data or classes.")
        return results

    if len(X) <= len(label_encoder.classes_): # Basic check
        st.warning(f"Dataset too small ({len(X)} samples for {len(label_encoder.classes_)} classes). Skipping training.")
        return results

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1,2))
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    results['tfidf_train_shape'] = X_train_tfidf.shape
    results['tfidf_test_shape'] = X_test_tfidf.shape

    if X_test_tfidf.shape[0] == 0:
        st.warning("Test set is empty after splitting. Cannot evaluate model.")
        return results

    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(label_encoder.classes_),
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=7
    )
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)

    results['accuracy'] = accuracy_score(y_test, y_pred)
    
    try:
        present_labels_encoded = np.unique(np.concatenate((y_test, y_pred)))
        valid_indices_for_report = [l for l in present_labels_encoded if l < len(label_encoder.classes_)]
        target_names_report = [label_encoder.classes_[i] for i in valid_indices_for_report]
        results['classification_report'] = classification_report(y_test, y_pred, labels=valid_indices_for_report, target_names=target_names_report, zero_division=0)
    except Exception as e:
        st.error(f"Could not generate full classification report with names: {e}")
        results['classification_report'] = classification_report(y_test, y_pred, zero_division=0)

    # Feature Importance Plot
    fig_feat_imp = None
    if hasattr(model, 'feature_importances_'):
        try:
            importances = model.feature_importances_
            feature_names = tfidf_vectorizer.get_feature_names_out()
            if len(importances) == len(feature_names):
                feature_importances_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
                N_TOP_FEATURES = 20
                top_features_df = feature_importances_df.sort_values(by='importance', ascending=False).head(N_TOP_FEATURES)
                if not top_features_df.empty:
                    fig_feat_imp, ax = plt.subplots(figsize=(10, 8))
                    sns.barplot(x='importance', y='feature', data=top_features_df, palette="viridis", ax=ax)
                    ax.set_title(f'Top {min(N_TOP_FEATURES, len(top_features_df))} Feature Importances (XGBoost)')
                    ax.set_xlabel('Importance Score')
                    ax.set_ylabel('Feature (TF-IDF Term)')
                    plt.tight_layout()
            else: st.warning("Mismatch in feature importance/names for plot.")
        except Exception as e: st.error(f"Error in feature importance plot: {e}")
    results['feature_importance_fig'] = fig_feat_imp

    # Partial Dependence Plot
    fig_pdp = None
    try:
        if hasattr(model, 'feature_importances_') and hasattr(tfidf_vectorizer, 'get_feature_names_out'):
            feature_names = tfidf_vectorizer.get_feature_names_out()
            if len(feature_names) > 0:
                N_TOP_FEATURES_PDP = 3 # Reduced for faster display in Streamlit
                importances = model.feature_importances_
                feature_importances_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
                top_features_df_pdp = feature_importances_df.sort_values(by='importance', ascending=False).head(N_TOP_FEATURES_PDP)
                top_feature_indices_pdp = [list(feature_names).index(name) for name in top_features_df_pdp['feature']]

                if len(top_feature_indices_pdp) > 0:
                    # X_train_tfidf can be large, consider using a sample for PDP if slow
                    X_pdp_sample = X_train_tfidf
                    if X_train_tfidf.shape[0] > 1000: # Sample if too large
                         sample_indices = np.random.choice(X_train_tfidf.shape[0], 1000, replace=False)
                         X_pdp_sample = X_train_tfidf[sample_indices]
                    
                    pdp_display = plot_partial_dependence(
                        model, X_pdp_sample, features=top_feature_indices_pdp,
                        feature_names=feature_names,
                        n_cols=len(top_feature_indices_pdp), # one plot per row if n_cols=1
                        grid_resolution=20, # Reduced for speed
                        percentiles=(0.05, 0.95) 
                    )
                    fig_pdp = pdp_display.figure_
                    fig_pdp.suptitle(f'Partial Dependence Plots (Top {len(top_feature_indices_pdp)} Features)', fontsize=16)
                    fig_pdp.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
            else: st.warning("No feature names for PDP.")
        else: st.warning("PDP prerequisites not met.")
    except Exception as e: st.error(f"Error generating PDP: {e}")
    results['pdp_fig'] = fig_pdp
    
    # Confusion Matrix
    fig_cm = None
    try:
        num_classes_to_display_cm = min(len(label_encoder.classes_), 20)
        unique_labels_in_y_test = np.unique(y_test)
        valid_labels_for_cm_plot = [l for l in unique_labels_in_y_test if l < len(label_encoder.classes_)]

        if valid_labels_for_cm_plot:
            labels_to_plot_indices = valid_labels_for_cm_plot[:num_classes_to_display_cm]
            if labels_to_plot_indices:
                display_labels_names_cm = label_encoder.inverse_transform(labels_to_plot_indices)
                cm_subset = confusion_matrix(y_test, y_pred, labels=labels_to_plot_indices)
                
                fig_cm, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(cm_subset, annot=True, fmt='d', cmap='Blues',
                            xticklabels=display_labels_names_cm, yticklabels=display_labels_names_cm,
                            annot_kws={"size": 8}, ax=ax)
                ax.set_title(f'Confusion Matrix (Subset of up to {num_classes_to_display_cm} Focus Areas)')
                ax.set_ylabel('Actual Focus Area')
                ax.set_xlabel('Predicted Focus Area')
                plt.xticks(rotation=45, ha='right', fontsize=8)
                plt.yticks(rotation=0, fontsize=8)
                plt.tight_layout()
        else: st.warning("No valid labels for CM plot.")
    except Exception as e: st.error(f"Error in confusion matrix plot: {e}")
    results['confusion_matrix_fig'] = fig_cm
    
    return results

# --- Streamlit App UI ---
st.set_page_config(layout="wide")
st.title("🩺 AI-Driven Disease Diagnosis System")
st.markdown("""
This application analyzes medical questions from the MedQuAD dataset to predict focus areas (potential diseases or conditions) using an XGBoost model.
Press the button below to load data, preprocess it, train the model, and view evaluation results and insights.
""")

# Initialize session state for results if not already present
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'logs' not in st.session_state:
    st.session_state.logs = None
if 'df_processed_head' not in st.session_state:
    st.session_state.df_processed_head = None


if st.button("🚀 Run Full Analysis Pipeline", type="primary"):
    st.session_state.analysis_results = None # Reset previous results
    st.session_state.logs = None
    st.session_state.df_processed_head = None

    overall_status = st.empty() # Placeholder for overall status

    with st.spinner("Step 1/4: Loading data..."):
        overall_status.info("Step 1/4: Loading data...")
        df_medquad = download_and_load_data()
    if df_medquad is None:
        st.error("Data loading failed. Pipeline stopped.")
        st.stop()
    st.success("✅ Data loaded successfully!")

    with st.spinner("Step 2/4: Preprocessing data..."):
        overall_status.info("Step 2/4: Preprocessing data...")
        X_data, y_data, le, df_processed, logs = preprocess_data_streamlit(df_medquad)
        st.session_state.logs = logs # Store logs
        if df_processed is not None:
            st.session_state.df_processed_head = df_processed[['question', 'focus_area']].head()
            value_counts_processed = df_processed['focus_area'].value_counts() # For display
            st.session_state.top_10_focus_areas = value_counts_processed.head(10).to_frame("Counts")


    if X_data is None or y_data is None or le is None or df_processed is None:
        st.error("Data preprocessing failed or yielded insufficient data. Pipeline stopped.")
        st.stop()
    st.success("✅ Data preprocessing complete!")


    with st.spinner("Step 3/4: Generating EDA visualizations..."):
        overall_status.info("Step 3/4: Generating EDA visualizations...")
        # These will be stored in session_state if successful and displayed later
        st.session_state.heatmap_fig = generate_word_freq_heatmap_streamlit(df_processed)
        st.session_state.qlen_hist_fig = generate_qlen_histogram_streamlit(df_processed)
    st.success("✅ EDA visualizations generated.")


    with st.spinner("Step 4/4: Training model and evaluating... (This may take a moment)"):
        overall_status.info("Step 4/4: Training model and evaluating...")
        if len(X_data) > 10 and len(le.classes_) >= 2:
            st.session_state.analysis_results = train_and_evaluate_model_streamlit(X_data, y_data, le)
        else:
            st.warning("Not enough data or classes to proceed with model training.")
            st.session_state.analysis_results = {"error": "Insufficient data for training"}
    
    if st.session_state.analysis_results and "error" not in st.session_state.analysis_results:
        st.success("🎉 Full analysis pipeline complete!")
        overall_status.success("🎉 Full analysis pipeline complete!")
    elif st.session_state.analysis_results and "error" in st.session_state.analysis_results:
        st.error(f"Pipeline stopped: {st.session_state.analysis_results['error']}")
        overall_status.error(f"Pipeline stopped: {st.session_state.analysis_results['error']}")
    else:
        st.error("Model training and evaluation failed or was skipped.")
        overall_status.error("Model training and evaluation failed or was skipped.")

# Display Preprocessing Logs
if st.session_state.logs:
    st.subheader("📋 Data Preprocessing Summary")
    df_info_str, missing_before, missing_after, filtering_log_combined = st.session_state.logs
    
    col1, col2 = st.columns(2)
    with col1:
        st.text("Initial df.info():")
        st.text_area("df_info", df_info_str, height=200)
        st.text("Missing values before dropping NaNs:")
        st.dataframe(missing_before)
    with col2:
        st.text("Missing values after dropping NaNs:")
        st.dataframe(missing_after)
        st.text("Filtering and Encoding Log:")
        st.text_area("filtering_log", filtering_log_combined, height=150)
    
    if st.session_state.df_processed_head is not None:
        st.text("Sample of processed data (df_processed):")
        st.dataframe(st.session_state.df_processed_head)
    
    if 'top_10_focus_areas' in st.session_state and st.session_state.top_10_focus_areas is not None:
        st.text("Top 10 Focus Areas after all filtering:")
        st.dataframe(st.session_state.top_10_focus_areas)


# Display EDA Plots
st.subheader("📊 Exploratory Data Analysis (EDA) Visualizations")
eda_col1, eda_col2 = st.columns(2)
with eda_col1:
    if 'heatmap_fig' in st.session_state and st.session_state.heatmap_fig:
        st.pyplot(st.session_state.heatmap_fig)
    else:
        st.markdown("*Word frequency heatmap will appear here after running the analysis.*")
with eda_col2:
    if 'qlen_hist_fig' in st.session_state and st.session_state.qlen_hist_fig:
        st.pyplot(st.session_state.qlen_hist_fig)
    else:
        st.markdown("*Question length histogram will appear here after running the analysis.*")


# Display Model Results
if st.session_state.analysis_results:
    if "error" in st.session_state.analysis_results:
        st.error(f"Could not display model results: {st.session_state.analysis_results['error']}")
    else:
        st.subheader("🤖 Model Performance & Insights (XGBoost)")
        
        st.metric(label="Model Accuracy", value=f"{st.session_state.analysis_results.get('accuracy', 0)*100:.2f}%")

        st.text("TF-IDF Shapes:")
        st.write(f"Train data shape: {st.session_state.analysis_results.get('tfidf_train_shape', 'N/A')}")
        st.write(f"Test data shape: {st.session_state.analysis_results.get('tfidf_test_shape', 'N/A')}")
        
        st.text("Classification Report:")
        st.text_area("classification_report_output", st.session_state.analysis_results.get('classification_report', "Not generated."), height=400)

        model_plot_col1, model_plot_col2 = st.columns(2)
        with model_plot_col1:
            if st.session_state.analysis_results.get('feature_importance_fig'):
                st.pyplot(st.session_state.analysis_results['feature_importance_fig'])
            else:
                 st.markdown("*Feature importance plot will appear here.*")
        with model_plot_col2:
            if st.session_state.analysis_results.get('confusion_matrix_fig'):
                st.pyplot(st.session_state.analysis_results['confusion_matrix_fig'])
            else:
                st.markdown("*Confusion matrix will appear here.*")
        
        if st.session_state.analysis_results.get('pdp_fig'):
            st.subheader("Partial Dependence Plots")
            st.pyplot(st.session_state.analysis_results['pdp_fig'])
        else:
            st.markdown("*Partial dependence plots will appear here.*")

else:
    st.info("Click the 'Run Full Analysis Pipeline' button to see results.")

st.sidebar.header("About")
st.sidebar.info(
    "This app demonstrates an AI-driven disease diagnosis system "
    "using the MedQuAD dataset and XGBoost for classification. "
    "It performs data loading, preprocessing, model training, evaluation, and visualization."
)
st.sidebar.markdown("---")
st.sidebar.markdown("Student: Shyam Gokul S")
st.sidebar.markdown("Reg No: 715523104049")
