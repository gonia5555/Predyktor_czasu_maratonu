import streamlit as st
import joblib
import json
import pandas as pd
from openai import OpenAI
from langfuse import Langfuse
from langfuse.decorators import observe
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
import time

# =========================
# KONFIGURACJA
# =========================

load_dotenv()

st.set_page_config(
    page_title="Half Marathon Time Predictor",
    layout="centered"
)

# =========================
# WYBÓR JĘZYKA
# =========================

LANG = st.selectbox(
    "🌍 Language / Język",
    ["PL", "EN"],
    index=0
)

TEXT = {
    "PL": {
        "title": "🏃 Predykcja czasu półmaratonu",
        "desc": "Aplikacja estymuje czas ukończenia półmaratonu (21.097 km) na podstawie danych wyłuskanych z języka naturalnego.",
        "how_title": "🔍 Jak to działa?",
        "how": """
1. Użytkownik opisuje się własnymi słowami  
2. Model LLM (OpenAI) wyłuskuje płeć, wiek i czas na 5 km  
3. Dane trafiają do modelu Machine Learning  
4. Model estymuje czas ukończenia półmaratonu  
""",
        "describe": "## 📝 Opisz się",
        "describe_help": "Podaj:\n- wiek\n- płeć\n- czas na 5 km",
        "placeholder": "Mam 41 lat, jestem kobietą, 5 km biegnę w 27 minut.",
        "load": "📥 Załaduj dane",
        "warn": "Wpisz opis",
        "missing": "Brakuje danych: ",
        "loaded": "✅ DANE ZAŁADOWANE",
        "gender": "👤 PŁEĆ",
        "age": "🎂 WIEK",
        "run5": "⏱️ 5 KM",
        "calc": "🏁 Oblicz czas półmaratonu",
        "spinner": "Model oblicza predykcję...",
        "success": "🏁 Predykcja zakończona sukcesem",
        "result": "⏱️ Szacowany czas półmaratonu",
        "compare": "### 📊 Porównanie dystansów",
        "others": "### 📈 Jak Ci poszło na tle innych zawodników?",
        "dist": "Dystans",
        "time_min": "Czas (minuty)",
        "xlabel": "Czas półmaratonu (minuty)",
        "ylabel": "Liczba zawodników",
        "hist_title": "Rozkład czasów półmaratonu",
        "others_label": "Inni zawodnicy",
        "you": "Twój wynik",
        "better": "🏆 Twój czas {time} jest lepszy niż **{pct:.1f}%** zawodników"
    },
    "EN": {
        "title": "🏃 Half Marathon Time Predictor",
        "desc": "This app estimates your half marathon (21.097 km) finish time based on natural language input.",
        "how_title": "🔍 How does it work?",
        "how": """
1. You describe yourself in your own words  
2. OpenAI extracts gender, age and 5 km time  
3. Data is passed to a Machine Learning model  
4. The model predicts your half marathon time  
""",
        "describe": "## 📝 Describe yourself",
        "describe_help": "Provide:\n- age\n- gender\n- 5 km time",
        "placeholder": "I am 41 years old, female, I run 5 km in 27 minutes.",
        "load": "📥 Load data",
        "warn": "Please enter a description",
        "missing": "Missing data: ",
        "loaded": "✅ DATA LOADED",
        "gender": "👤 GENDER",
        "age": "🎂 AGE",
        "run5": "⏱️ 5 KM",
        "calc": "🏁 Calculate half marathon time",
        "spinner": "Calculating prediction...",
        "success": "🏁 Prediction completed successfully",
        "result": "⏱️ Estimated half marathon time",
        "compare": "### 📊 Distance comparison",
        "others": "### 📈 How do you compare to other runners?",
        "dist": "Distance",
        "time_min": "Time (minutes)",
        "xlabel": "Half marathon time (minutes)",
        "ylabel": "Number of runners",
        "hist_title": "Half marathon time distribution",
        "others_label": "Other runners",
        "you": "Your result",
        "better": "🏆 Your time {time} is better than **{pct:.1f}%** of runners"
    }
}

T = TEXT[LANG]

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# HEADER
# =========================

st.markdown(f"""
<h1 style="text-align: center;">{T['title']}</h1>
<p style="text-align: center; font-size: 18px;">{T['desc']}</p>
""", unsafe_allow_html=True)

st.image("marathon.png", use_container_width=True)

st.sidebar.markdown("""
# **Technologies:**  
Machine Learning · OpenAI · Langfuse · Streamlit · Digital Ocean
---
""")

st.sidebar.markdown(f"""
### {T['how_title']}
{T['how']}
---
""")

# =========================
# MODEL
# =========================

@st.cache_resource
def load_model():
    return joblib.load("half_marathon_model.pkl")

model = load_model()

# =========================
# LLM → EKSTRAKCJA
# =========================

@observe(name="extract_user_data")
def extract_user_data(text: str) -> dict:
    prompt = """
Extract:
- gender (male/female)
- age
- 5 km time IN SECONDS (time_5km_seconds)

Return ONLY JSON.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Return only valid JSON."},
            {"role": "user", "content": prompt + "\n\nText:\n" + text}
        ],
        temperature=0
    )

    data = json.loads(response.choices[0].message.content)

    if data.get("gender") in ["female", "kobieta", "woman"]:
        data["gender"] = "kobieta"
    else:
        data["gender"] = "mężczyzna"

    return data


def validate_data(data):
    missing = []
    if "gender" not in data:
        missing.append("gender")
    if "age" not in data:
        missing.append("age")
    if "time_5km_seconds" not in data:
        missing.append("5 km time")
    return missing

def format_seconds(seconds: int) -> str:
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def display_gender(gender: str, lang: str) -> str:
    if lang == "EN":
        return "Female" if gender == "kobieta" else "Male"
    return gender.upper()

def predict_half_marathon(data: dict) -> int:
    X = pd.DataFrame([{
        "plec": data["gender"],
        "wiek": data["age"],
        "5 km czas": data["time_5km_seconds"]
    }])
    return int(model.predict(X)[0])

# =========================
# UI
# =========================

st.markdown(T["describe"])
st.markdown(T["describe_help"])

user_input = st.text_area("", placeholder=T["placeholder"])

if "data" not in st.session_state:
    st.session_state.data = None

if st.button(T["load"]):
    if not user_input.strip():
        st.warning(T["warn"])
    else:
        data = extract_user_data(user_input)
        missing = validate_data(data)

        if missing:
            st.error(T["missing"] + ", ".join(missing))
        else:
            st.session_state.data = data
            st.success(T["loaded"])

if st.session_state.data:
    data = st.session_state.data

    col1, col2, col3 = st.columns(3)
    col1.metric(T["gender"], display_gender(data["gender"], LANG))
    col2.metric(T["age"], data["age"])
    col3.metric(T["run5"], format_seconds(data["time_5km_seconds"]))

    st.markdown("---")

    if st.button(T["calc"]):
        with st.spinner(T["spinner"]):
            time.sleep(1)
            result = predict_half_marathon(data)

        st.success(T["success"])
        st.metric(T["result"], format_seconds(result))

        st.markdown(T["compare"])

        df = pd.DataFrame({
            T["dist"]: ["5 km", "Half Marathon"],
            T["time_min"]: [data["time_5km_seconds"] / 60, result / 60]
        })

        fig, ax = plt.subplots()
        ax.bar(df[T["dist"]], df[T["time_min"]])
        ax.set_ylabel(T["time_min"])
        st.pyplot(fig)

        st.markdown(T["others"])

        np.random.seed(42)
        others_seconds = np.random.normal(
            loc=1 * 3600 + 55 * 60,
            scale=15 * 60,
            size=5000
        )
        others_seconds = others_seconds[others_seconds > 0]

        fig, ax = plt.subplots()
        ax.hist(others_seconds / 60, bins=40, alpha=0.7, label=T["others_label"])
        ax.axvline(result / 60, color="red", linewidth=4, label=T["you"])
        ax.set_xlabel(T["xlabel"])
        ax.set_ylabel(T["ylabel"])
        ax.set_title(T["hist_title"])
        ax.legend()

        st.pyplot(fig)

        percentile = (others_seconds > result).mean() * 100

        st.success(
            T["better"].format(
                time=format_seconds(result),
                pct=percentile
            )
        )
