import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

# Page title
st.title("YKS Analysis")

# Tabs for each subject
tabs = st.tabs(['Türkçe', 'Matematik', 'Geometri', 'Fizik'])
subjects = ['Türkçe', 'Matematik', 'Geometri', 'Fizik']
csv_paths = {
    'Türkçe': 'turkce.csv',
    'Matematik': 'matematik.csv',
    'Geometri': 'geometri.csv',
    'Fizik': 'fizik.csv'
}

# Data preprocessing function
def preprocess_data(selected_subject):
    csv_file = csv_paths[selected_subject]
    try:
        df = pd.read_csv(csv_file, on_bad_lines='skip')
    except pd.errors.ParserError as e:
        st.error(f"Error reading the CSV file: {e}")
        st.stop()

    df.columns = [f"Column_{i}" if str(col).strip() == "" else str(col) for i, col in enumerate(df.columns)]
    df = df.loc[:, ~df.columns.duplicated()]
    df = df[['KONULAR'] + [col for col in df.columns if col.isdigit() and int(col) >= 2018]]
    df = df.set_index('KONULAR').T
    df = df.replace(['-', '', np.nan], 0).astype(int)
    return df

# Prediction function with GridSearchCV
def make_predictions(df):
    predictions = pd.DataFrame(index=['2025'], columns=df.columns)
    for KONULAR in df.columns:
        X = df.index.astype(int).values.reshape(-1, 1)
        y = df[KONULAR].values
        if len(y) < 3:
            continue
        model = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [10, None],
            'min_samples_split': [2, 5],
        }
        grid_search = GridSearchCV(model, param_grid, cv=min(3, len(y)), n_jobs=-1, verbose=0)
        grid_search.fit(X, y)
        best_model = grid_search.best_estimator_
        predictions.loc['2025', KONULAR] = best_model.predict([[2025]])[0]
    return predictions

# Visualization function with different plot styles
def create_plots(df, predictions, selected_subject):
    all_predictions = pd.concat([df, predictions])
    st.subheader(f"{selected_subject} 2025 Predictions")
    st.table(all_predictions)
    st.subheader(f"{selected_subject} Prediction Graphs")
    
    # Matplotlib Plot with improved styling
    fig, ax = plt.subplots(figsize=(12, 6))
    for KONULAR in df.columns:
        years = df.index.astype(int)
        values = df[KONULAR]
        ax.plot(years, values, marker='o', label=KONULAR, linestyle='-', linewidth=2)
        ax.plot(2025, predictions.loc['2025', KONULAR], marker='x', color='red', markersize=10, label=f'{KONULAR} 2025 Prediction')
    
    ax.set_title(f"{selected_subject} KONULAR Predictions", fontsize=16)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Question Count", fontsize=12)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xticks(rotation=45)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    st.pyplot(fig)

    # Plotly Interactive Plot
    fig_plotly = px.line(df.reset_index(), x=df.index, y=df.columns, title=f"{selected_subject} KONULAR Trends")
    fig_plotly.update_layout(showlegend=True, title_font_size=20, xaxis_title="Year", yaxis_title="Question Count")
    fig_plotly.add_scatter(x=[2025], y=[predictions.iloc[0]], mode='markers', name='2025 Predictions', marker=dict(size=12, color='red'))
    st.plotly_chart(fig_plotly)
    
    # Seaborn Bar Plot
    df_melt = df.reset_index().melt(id_vars='index')
    sns.set(style="whitegrid", palette="muted")
    fig_seaborn, ax_seaborn = plt.subplots(figsize=(12, 6))
    sns.barplot(x='index', y='value', hue='KONULAR', data=df_melt, ax=ax_seaborn)
    ax_seaborn.set_title(f"{selected_subject} Yearly Distribution", fontsize=16)
    ax_seaborn.set_xlabel("Year", fontsize=12)
    ax_seaborn.set_ylabel("Question Count", fontsize=12)
    st.pyplot(fig_seaborn)

# Process and plot data for each subject
for i, tab in enumerate(tabs):
    with tab:
        selected_subject = subjects[i]
        df = preprocess_data(selected_subject)
        predictions = make_predictions(df)
        create_plots(df, predictions, selected_subject)
