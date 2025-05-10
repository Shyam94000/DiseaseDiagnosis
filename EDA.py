if 'df_processed' in locals() and df_processed is not None and not df_processed.empty:
    plt.figure(figsize=(16, 10)) # Adjust figure size as needed
    sns.countplot(y=df_processed['focus_area'], order = df_processed['focus_area'].value_counts().index)
    plt.title(f'Distribution of Focus Areas (Total: {len(df_processed["focus_area"].unique())} classes)')
    plt.xlabel('Number of Questions')
    plt.ylabel('Focus Area')
    plt.tight_layout()
    plt.show()
else:
    print("df_processed is not available or is empty. Cannot generate focus area countplot.")




if 'df_processed' in locals() and df_processed is not None and not df_processed.empty:
    df_processed['question_length_words'] = df_processed['question'].apply(lambda x: len(str(x).split()))
    plt.figure(figsize=(10, 6))
    sns.histplot(df_processed['question_length_words'], kde=True, bins=50)
    plt.title('Distribution of Question Lengths (Number of Words)')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.show()

    # You can also print some descriptive statistics
    # print("\nDescriptive statistics for question length (words):")
    # print(df_processed['question_length_words'].describe())
else:
    print("df_processed is not available or is empty. Cannot generate question length histogram.")
