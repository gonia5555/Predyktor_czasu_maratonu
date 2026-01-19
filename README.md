# 🏃 Half Marathon Time Predictor

An AI-powered web application that estimates a runner’s **half marathon (21.097 km) finish time** using **natural language input**.

Instead of filling out forms, users simply describe themselves in plain text. A Large Language Model (LLM) extracts the relevant information, which is then passed to a Machine Learning model trained on real race data.

👉 **Live app:**  
https://predyktorczasumaratonugonia5555.streamlit.app/

---

## 🚀 Features

- Natural language input (no forms required)
- LLM-based extraction of age, gender, and 5 km time
- Machine Learning model for time prediction
- Multilingual support (English & Polish)
- Visual comparison of 5 km vs half marathon time
- Performance percentile vs other runners
- LLM observability with Langfuse

---

## 🧠 How It Works

1. User describes themselves in natural language  
2. OpenAI LLM extracts structured features  
3. Data is validated and preprocessed  
4. ML model predicts half marathon finish time  
5. Results are visualized and compared with other runners  

---

## 📊 Data

The prediction model was trained on **real-world race results** from:

- Half Marathon Wrocław 2023  
- Half Marathon Wrocław 2024  

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Scikit-learn
- OpenAI API
- Langfuse
- Pandas & NumPy
- Matplotlib
- Joblib
- DigitalOcean

---

## 🏗️ Architecture (High-Level)

- **Frontend:** Streamlit UI  
- **LLM Layer:** OpenAI (data extraction from text)  
- **ML Layer:** Regression model (joblib)  
- **Monitoring:** Langfuse (LLM traces)  

---

# 🏃 Predyktor czasu półmaratonu

Aplikacja webowa oparta na sztucznej inteligencji, która estymuje **czas ukończenia półmaratonu (21,097 km)** na podstawie **opisu użytkownika w języku naturalnym**.

Zamiast wypełniać formularze, użytkownik opisuje się własnymi słowami. Model językowy (LLM) wyłuskuje kluczowe informacje, które następnie trafiają do modelu Machine Learning wytrenowanego na rzeczywistych danych biegowych.

👉 **Aplikacja online:**  
https://predyktorczasumaratonugonia5555.streamlit.app/

---

## 🚀 Funkcjonalności

- Wprowadzanie danych w języku naturalnym (bez formularzy)
- Ekstrakcja wieku, płci i czasu na 5 km przy użyciu LLM
- Predykcja czasu półmaratonu za pomocą modelu Machine Learning
- Obsługa dwóch języków (polski i angielski)
- Wizualne porównanie czasu na 5 km i półmaraton
- Pozycja percentylowa na tle innych biegaczy
- Monitoring pracy LLM z wykorzystaniem Langfuse

---

## 🧠 Jak to działa?

1. Użytkownik opisuje się własnymi słowami  
2. Model OpenAI wyodrębnia dane strukturalne  
3. Dane są walidowane i przetwarzane  
4. Model ML przewiduje czas ukończenia półmaratonu  
5. Wyniki są prezentowane w formie wizualizacji  

---

## 📊 Dane

Model predykcyjny został wytrenowany na **rzeczywistych wynikach zawodów biegowych**:

- Półmaraton Wrocław 2023  
- Półmaraton Wrocław 2024  


---

## 🛠️ Stos technologiczny

- Python
- Streamlit
- Scikit-learn
- OpenAI API
- Langfuse
- Pandas & NumPy
- Matplotlib
- Joblib
- DigitalOcean

---

## 🏗️ Architektura (wysoki poziom)

- **Frontend:** Streamlit  
- **Warstwa LLM:** OpenAI (ekstrakcja danych z tekstu)  
- **Warstwa ML:** model regresyjny (joblib)  
- **Monitoring:** Langfuse (śledzenie zapytań do LLM)  

---




This project is intended for educational and portfolio purposes.
