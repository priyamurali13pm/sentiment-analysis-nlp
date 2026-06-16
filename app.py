#import nltk
#nltk.download('stopwords')

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import pickle
import re

model = pickle.load(open("sentiment_pipeline.pkl", "rb"))

st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")

st.title("📊 Sentiment Analysis Dashboard")
st.markdown("Analyze user reviews, sentiment, and product insights")

# -------------------- TEXT CLEANING --------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text


# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    model = joblib.load("sentiment_pipeline.pkl")
    #from sentence_transformers import SentenceTransformer
    #embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return model

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_reviews.csv")

model = load_model()
df = load_data()
st.write(df.shape)


# -------------------- ADD PREDICTIONS --------------------
@st.cache_data
def add_predictions(df):
    #embeddings = embedder.encode(df['text'].tolist(), show_progress_bar=False)
    #preds = model.predict(embeddings)
    preds = model.predict(df['text'])
    #label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    df = df.copy()
    #df['predicted_sentiment'] = [label_map[p] for p in preds]
    df['predicted_sentiment'] = [str(p).lower() for p in preds]
    return df

df = add_predictions(df)

# -------------------- FILTERS --------------------
st.sidebar.header("🔍 Filters")

platform_filter = st.sidebar.multiselect(
    "Platform",
    options=df['platform'].unique(),
    default=df['platform'].unique()
)

location_filter = st.sidebar.multiselect(
    "Location",
    options=df['location'].unique(),
    default=df['location'].unique()
)

df_filtered = df[
    (df['platform'].isin(platform_filter)) &
    (df['location'].isin(location_filter))
]

# -------------------- KPI SECTION --------------------
st.subheader("📌 Key Metrics")
st.write(df_filtered['predicted_sentiment'].unique())

total_reviews = len(df_filtered)
avg_rating = df_filtered['rating'].mean()
positive_pct = (df_filtered['predicted_sentiment'] == 'positive').mean() * 100

col1, col2, col3 = st.columns(3)
col1.metric("Total Reviews", total_reviews)
col2.metric("Average Rating", f"{avg_rating:.2f}")
col3.metric("Positive %", f"{positive_pct:.1f}%")

st.divider()

# -------------------- CHARTS --------------------
st.subheader("📊 Analysis")

# Pie Chart
sentiment_counts = df_filtered['predicted_sentiment'].value_counts()
fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, title="Overall Sentiment")
st.plotly_chart(fig, use_container_width=True)

# Crosstab
ct = pd.crosstab(df_filtered['rating'], df_filtered['predicted_sentiment'])
st.dataframe(ct)

# Platform Rating
avg_rating_platform = df_filtered.groupby('platform')['rating'].mean().reset_index()
fig = px.bar(avg_rating_platform, x='platform', y='rating', color='platform')
st.plotly_chart(fig, use_container_width=True)

# Time Trend
df_filtered['date'] = pd.to_datetime(df_filtered['date'], errors='coerce')
df_time = df_filtered.dropna(subset=['date'])
df_time['month'] = df_time['date'].dt.to_period('M').astype(str)
avg_time = df_time.groupby('month')['rating'].mean().reset_index()

fig = px.line(avg_time, x='month', y='rating', markers=True)
st.plotly_chart(fig, use_container_width=True)

# Location
avg_location = df_filtered.groupby('location')['rating'].mean().reset_index()
fig = px.choropleth(avg_location, locations="location", locationmode="country names", color="rating")
st.plotly_chart(fig, use_container_width=True)

st.divider()

# -------------------- MODEL EVALUATION --------------------
@st.cache_data
def evaluate_model(df):
    df = df.dropna(subset=['sentiment', 'predicted_sentiment'])
    y_true = df['sentiment'].str.lower()
    y_pred = df['predicted_sentiment']
    #label_map_reverse = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    #y_true = df['sentiment'].map(label_map_reverse)
    #y_pred = df['predicted_sentiment'].map(label_map_reverse)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)

    return acc, prec, rec, f1, cm

st.subheader("📊 Model Evaluation")

acc, prec, rec, f1, cm = evaluate_model(df)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{acc:.2f}")
col2.metric("Precision", f"{prec:.2f}")
col3.metric("Recall", f"{rec:.2f}")
col4.metric("F1 Score", f"{f1:.2f}")

st.subheader("📉 Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Neg','Neu','Pos'],
            yticklabels=['Neg','Neu','Pos'])
st.pyplot(fig)

st.divider()

# -------------------- INSIGHTS --------------------
st.subheader("💡 Key Insights")

def generate_insights(df):
    insights = []
    insights.append(f"👉 Most reviews are **{df['predicted_sentiment'].value_counts().idxmax()}**.")
    insights.append(f"👉 Best platform: **{df.groupby('platform')['rating'].mean().idxmax()}**.")
    insights.append(f"👉 Worst platform: **{df.groupby('platform')['rating'].mean().idxmin()}**.")
    insights.append(f"👉 Top location: **{df.groupby('location')['rating'].mean().idxmax()}**.")
    insights.append(f"👉 Negative reviews: **{(df['predicted_sentiment']=='Negative').mean()*100:.1f}%**")
    return insights

for insight in generate_insights(df_filtered):
    st.markdown(f"✅ {insight}")

st.divider()

# -------------------- RECOMMENDATIONS --------------------
st.subheader("📌 Business Recommendations")
st.markdown("### 🎯 Actionable Suggestions")

def generate_recommendations(df):
    rec = []
    neg = (df['predicted_sentiment'] == 'Negative').mean() * 100

    if neg > 30:
        rec.append("🔴 Improve product quality & support")
    elif neg > 15:
        rec.append("🟡 Investigate user complaints")
    else:
        rec.append("🟢 Maintain current performance")

    rec.append(f"📉 Improve **{df.groupby('platform')['rating'].mean().idxmin()}** platform")

    return rec

for r in generate_recommendations(df_filtered):
    st.markdown(f"👉 {r}")

st.divider()

# -------------------- USER INPUT --------------------
st.subheader("🧠 Try Sentiment Prediction")

user_text = st.text_area("Enter a review text")

if st.button("Predict Sentiment"):
    if user_text.strip() != "":
        #emb = embedder.encode([user_text])
        #pred = model.predict(emb)[0]
        prediction = model.predict([user_text])[0]
        #label_map = {0:'Negative', 1:'Neutral', 2:'Positive'}
        #st.success(f"Predicted Sentiment: **{label_map[pred]}**")
        st.success(f"Predicted Sentiment: **{prediction}**")
    else:
        st.warning("Please enter some text.")