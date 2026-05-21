"""
Streamlit application for sentiment analysis of customer opinions.

This app allows users to upload a CSV file with customer opinions and their
corresponding class labels (e.g., positive/negative or star ratings). The app
performs text preprocessing (stopword removal and lemmatization), generates a
word cloud, displays the top 10 most frequent words in a bar chart, and
includes an additional chart showing sentiment distribution based on a
pretrained language model. Users can filter analyses by the original class
labels and can also enter new comments to receive a sentiment prediction.

To run the app locally:

    pip install streamlit pandas matplotlib wordcloud nltk transformers torch

Then execute:

    streamlit run sentiment_app.py

Make sure to download the NLTK stopwords and WordNet data before the first run:

    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')

"""

from collections import Counter
from typing import List, Tuple

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from transformers import pipeline
from wordcloud import WordCloud


SENTIMENT_MODEL = "pysentimiento/robertuito-sentiment-analysis"


def preprocess_text(text: str, language: str = "spanish") -> List[str]:
    """Preprocess a piece of text by removing stopwords and lemmatizing.

    Args:
        text: The input text.
        language: The language of the stopwords. Defaults to "spanish".

    Returns:
        A list of processed tokens.
    """
    stop_words = set(stopwords.words(language))
    stemmer = SnowballStemmer(language)
    tokens = [word.lower() for word in text.split() if word.isalpha()]
    return [stemmer.stem(word) for word in tokens if word not in stop_words]


def generate_wordcloud(tokens: List[str]) -> WordCloud:
    """Generate a WordCloud object from a list of tokens."""
    text = " ".join(tokens)
    return WordCloud(width=800, height=400, background_color="white").generate(text)


def top_frequent_words(tokens: List[str], n: int = 10) -> List[Tuple[str, int]]:
    """Return the top n most frequent words and their counts."""
    counter = Counter(tokens)
    return counter.most_common(n)


def classify_sentiments(texts: List[str]) -> List[str]:
    """Classify sentiments using a Spanish sentiment model from Hugging Face.

    Args:
        texts: List of input texts.

    Returns:
        List of sentiment labels (Positivo/Negativo/Neutral).
    """
    sentiment_pipeline = load_sentiment_pipeline()
    results = sentiment_pipeline(texts)
    label_map = {
        "POS": "Positivo",
        "NEG": "Negativo",
        "NEU": "Neutral",
        "POSITIVE": "Positivo",
        "NEGATIVE": "Negativo",
        "NEUTRAL": "Neutral",
    }
    return [label_map.get(result["label"], result["label"]) for result in results]


@st.cache_resource
def load_sentiment_pipeline():
    """Load the Hugging Face sentiment model once per Streamlit session."""
    return pipeline("sentiment-analysis", model=SENTIMENT_MODEL)


def display_bar_chart(top_words: List[Tuple[str, int]]):
    """Display a bar chart of top words using matplotlib."""
    words, counts = zip(*top_words)
    fig, ax = plt.subplots()
    ax.bar(words, counts, color="skyblue")
    ax.set_xlabel("Palabras")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Top palabras más frecuentes")
    plt.xticks(rotation=45)
    st.pyplot(fig)


def display_sentiment_distribution(labels: List[str]):
    """Display a pie chart of sentiment distribution."""
    count = Counter(labels)
    fig, ax = plt.subplots()
    ax.pie(count.values(), labels=count.keys(), autopct="%1.1f%%")
    ax.set_title("Distribución de sentimientos (modelo)")
    st.pyplot(fig)


def display_average_length_by_class(df: pd.DataFrame):
    """Display the average number of processed words by original class."""
    avg_lengths = df.groupby("clase")["tokens"].apply(lambda rows: sum(len(row) for row in rows) / len(rows))
    fig, ax = plt.subplots()
    ax.bar(avg_lengths.index, avg_lengths.values, color="mediumseagreen")
    ax.set_xlabel("Clase")
    ax.set_ylabel("Promedio de palabras")
    ax.set_title("Longitud promedio de opiniones por clase")
    st.pyplot(fig)


def display_new_comment_classifier():
    """Display an interactive sentiment classifier for a new comment."""
    st.subheader("Clasificar un nuevo comentario")
    st.caption("Escribe un comentario y presiona Enter o el botón para clasificarlo.")
    with st.form("new_comment_form"):
        new_comment = st.text_input("Comentario nuevo")
        submitted = st.form_submit_button("Clasificar comentario")

    if submitted:
        if not new_comment.strip():
            st.warning("Escribe un comentario antes de clasificar.")
            return
        try:
            new_label = classify_sentiments([new_comment])[0]
            st.success(f"Sentimiento detectado: {new_label}")
            st.dataframe(
                pd.DataFrame(
                    [{"comentario": new_comment, "sentimiento_modelo": new_label}]
                ),
                use_container_width=True,
            )
        except Exception as exc:
            st.error(
                "No se pudo clasificar el comentario. Intenta nuevamente o revisa los logs de Streamlit."
            )
            st.caption(f"Detalle tecnico: {exc}")


def main():
    st.title("Análisis de Opiniones de Clientes")
    st.write("""
    Cargue un archivo CSV con al menos 30 opiniones de clientes junto con su clase
    (por ejemplo, positivo/negativo o número de estrellas). El CSV debe tener
    una columna llamada `opinion` y otra llamada `clase`.
    """)

    # File uploader
    uploaded_file = st.file_uploader("Subir archivo CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if "opinion" not in df.columns or "clase" not in df.columns:
            st.error("El CSV debe contener las columnas 'opinion' y 'clase'.")
            return
        # Preprocess all texts
        df["tokens"] = df["opinion"].apply(lambda x: preprocess_text(str(x)))
        # Flatten all tokens for overall analysis
        all_tokens = [token for tokens_list in df["tokens"] for token in tokens_list]
        # Generate word cloud
        wc = generate_wordcloud(all_tokens)
        st.subheader("Nube de palabras (todas las opiniones)")
        st.image(wc.to_array())
        # Bar chart of top 10 words
        st.subheader("Top 10 palabras más frecuentes")
        top_words = top_frequent_words(all_tokens, n=10)
        display_bar_chart(top_words)
        # Additional chart: sentiment distribution (original classes)
        st.subheader("Distribución de opiniones por clase (datos originales)")
        class_counts = df["clase"].value_counts()
        fig, ax = plt.subplots()
        ax.pie(class_counts.values, labels=class_counts.index, autopct="%1.1f%%")
        ax.set_title("Distribución de opiniones originales")
        st.pyplot(fig)
        st.subheader("Longitud promedio de opiniones por clase")
        display_average_length_by_class(df)
        # Filter by class
        unique_classes = df["clase"].unique().tolist()
        selected_class = st.selectbox("Filtrar opiniones por clase", ["Todas"] + unique_classes)
        if selected_class != "Todas":
            df_filtered = df[df["clase"] == selected_class]
        else:
            df_filtered = df
        # Classification using LLM
        st.subheader("Clasificación de sentimientos utilizando modelo de lenguaje")
        st.caption(f"Modelo usado: {SENTIMENT_MODEL}.")
        # To avoid long runtime, process only first 100 comments; in this case we have 30
        try:
            labels = classify_sentiments(df_filtered["opinion"].tolist())
        except Exception as exc:
            st.error(
                "No se pudo clasificar con el modelo de sentimiento. Intenta nuevamente o revisa los logs de Streamlit."
            )
            st.caption(f"Detalle tecnico: {exc}")
            return
        df_filtered = df_filtered.copy()
        df_filtered["sentimiento_modelo"] = labels
        st.dataframe(df_filtered[["opinion", "clase", "sentimiento_modelo"]])
        display_sentiment_distribution(labels)
    else:
        st.info("Suba un archivo CSV para comenzar el análisis.")

    display_new_comment_classifier()


if __name__ == "__main__":
    nltk.download("stopwords")
    nltk.download("wordnet")
    main()
