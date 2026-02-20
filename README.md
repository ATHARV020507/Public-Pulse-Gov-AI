# 🏛️ Public Pulse  
## AI-Powered Government Triage System  

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Flask](https://img.shields.io/badge/Framework-Flask-lightgrey.svg)
![AI](https://img.shields.io/badge/AI-DistilBERT%20%7C%20LGBM-orange.svg)
![Security](https://img.shields.io/badge/Security-Double--Lock-green.svg)

---

## 📌 Overview  

**Public Pulse** is a secure, real-time analytics dashboard engineered specifically for government officials to process, analyze, and triage public e-consultation feedback.  

Built with a strict focus on data security and rapid response, the system utilizes a custom **3-Stage Hybrid AI Engine** to automatically categorize thousands of public comments by:

- 📊 Sentiment  
- 📄 Factual Basis  
- 🚨 Critical Urgency  

This allows human officers to bypass manual sorting and immediately address high-priority civilian issues.

---

# 🚀 Core Architecture & Features  

## 🧠 1. 3-Stage Hybrid AI Engine  

To balance processing speed with deep contextual accuracy, the NLP pipeline processes text through three distinct layers:

### 🔥 Stage 1: Urgency Protocol  
- O(1) keyword detection  
- Maps critical hazards (e.g., "fire", "crash", "leak", "hazard")  
- Enables immediate emergency triage  

### 📑 Stage 2: Factual Filter (VADER)  
- Heuristic sentiment analysis  
- Isolates highly neutral/factual statements  
- Compound scores near 0 with high neutrality  
- Separates objective incident reports from subjective opinions  

### 🤖 Stage 3: Deep Scan (DistilBERT + LightGBM)  

A dual-model fallback system:

- ⚡ TF-IDF + LightGBM handles standard processing  
- 🧠 Fine-tuned Hugging Face `distilbert-base-uncased` transformer processes complex or low-confidence inputs  

---

## 🔐 2. "Double-Lock" Security Protocol  

Designed for confidential GovTech deployment, the system employs strict access controls:

### 🏗 Infrastructure Layer  
- Deployment via private, whitelisted cloud environments  

### 🔑 Application Layer  
- Role-Based Access Control (RBAC) powered by Flask-Login  
- Secure password hashing using `werkzeug.security`  

### 👻 The "Ghost Admin" Protocol  
- Admin UI elements dynamically stripped from the DOM at the server level for non-root users  
- Forced unauthorized routing attempts (e.g., `/admin/dashboard`) trigger immediate session termination and security logging  

---

## 📊 3. Live Dashboard & Analytics  

### 📈 Real-Time Triage Matrix  
- Dynamically updates feedback priority  
- Based on live AI inferences and sentiment scoring  

### 🗂 Automated Topic Extraction  
- Custom NLP mapping routes issues to government sectors:  
  - Sanitation  
  - Infrastructure  
  - Law Enforcement  
  - Healthcare  

### ☁️ Live Word Cloud Generation  
- In-memory generation using `wordcloud` and `matplotlib`  
- Provides instant macro-level insights  

---

# 🛠️ Tech Stack  

### 🖥 Backend  
- Python  
- Flask  
- SQLite (SQLAlchemy)  
- ThreadPoolExecutor (for bulk CSV processing)  

### 🤖 Machine Learning  
- Hugging Face Transformers  
- LightGBM  
- NLTK  
- VADER  
- Scikit-learn  

### 🎨 Frontend  
- HTML5  
- CSS3  
- JavaScript  
- Chart.js  
- Jinja2 Templating  

---

# ⚙️ Local Setup & Installation  

> ⚠️ **Security Notice:**  
> This public repository represents the core computational engine.  
> The production SQLite database (`officers.db`) and secure environment configuration keys have been intentionally excluded from this repository to maintain data integrity and strict security compliance.

## 📥 Clone the Repository (this is the first step)

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/Public-Pulse-Gov-AI.git
cd Public-Pulse-Gov-AI
```
2. Install Dependencies:
   pip install -r requirements.txt

3. Initialize the Environment:
   Ensure your pre-trained models (sentiment_model.pkl and tfidf_vectorizer.pkl) are located in the /artifacts directory.

4. Run the Application:
   python app.py

👨‍💻 Author
Atharv Mishra B.Tech Data Science | Oriental Institute of Science and Technology (OIST), Bhopal. 

LinkedIn: https://www.linkedin.com/in/atharv-mishra-76834a2a3/
GitHub: github.com/ATHARV020507
   
   
