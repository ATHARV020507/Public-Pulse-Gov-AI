import pickle
import re
import os
import time
import io
import threading
import random
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash, render_template_string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from wordcloud import WordCloud
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.middleware.proxy_fix import ProxyFix 

# --- SECURITY IMPORTS ---
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from models import db, User 

# --- 1. INITIALIZE APP & CONFIG ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'sih-2025-secure-key-change-this' 
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///officers.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# --- CLOUD SECURITY HEADERS ---
app.config['SESSION_COOKIE_SECURE'] = True
app.config['REMEMBER_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'None'
app.config['SESSION_COOKIE_NAME'] = 'public_pulse_session'

app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login' 

def log_msg(msg):
    print(msg, flush=True)

# --- 2. SMART SECURITY INITIALIZATION ---
def initialize_system():
    """
    SECURITY UPGRADE:
    Only creates the default admin if the database is EMPTY.
    Never overwrites existing passwords.
    """
    with app.app_context():
        try:
            db.create_all()
            
            # Check if ANY user exists
            user_count = User.query.count()
            
            if user_count == 0:
                log_msg("⚠️ System Empty. Initializing Super Admin...")
                hashed_pw = generate_password_hash('admin123') 
                new_admin = User(username='admin', password_hash=hashed_pw)
                db.session.add(new_admin)
                db.session.commit()
                log_msg("✅ Super Admin Created: admin / admin123")
            else:
                log_msg(f"✅ System Verified. Found {user_count} registered officials.")
                
        except Exception as e:
            log_msg(f"❌ DB Error: {e}")

# Run once on import
initialize_system()

# --- NLTK SETUP ---
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except Exception: pass 

# --- 3. LOAD AI MODELS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts')

vectorizer = None
lgbm_model = None
tokenizer = None
transformer_model = None
summarizer = None
vader_analyzer = SentimentIntensityAnalyzer()

log_msg("--- SYSTEM STARTUP: Loading AI Models... ---")

try:
    vec_path = os.path.join(ARTIFACTS_DIR, 'tfidf_vectorizer.pkl')
    model_path = os.path.join(ARTIFACTS_DIR, 'sentiment_model.pkl')
    if os.path.exists(vec_path) and os.path.exists(model_path):
        vectorizer = pickle.load(open(vec_path, 'rb'))
        lgbm_model = pickle.load(open(model_path, 'rb'))
        log_msg("✅ Stage 1 (Fast Model) loaded!")
    else:
        log_msg("⚠️ Stage 1 files not found. Running in Fallback Mode.")
except Exception as e:
    log_msg(f"❌ Error loading Stage 1: {e}")

try:
    log_msg("⏳ Downloading/Loading Smart Model (Wait for it)...")
    model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    transformer_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")
    log_msg("✅ Stage 2 (Smart Model) loaded!")
except Exception as e:
    log_msg(f"❌ Error loading Stage 2: {e}")

# --- 4. HELPER FUNCTIONS ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

URGENT_KEYWORDS = [
    'urgent', 'fail', 'broken', 'unsafe', 'hazard', 'hazardous', 
    'immediately', 'immediate', 'dangerous', 'leak', 'fire', 
    'critical', 'severe', 'blocked', 'crash', 'down', 'error 404',
    'emergency', 'risk', 'expose', 'violation', 'threat'
]
CONFIDENCE_THRESHOLD = 0.80

GOV_TOPIC_MAP = {
    'road': 'Infrastructure', 'pothole': 'Infrastructure', 'highway': 'Transport',
    'water': 'Water Supply', 'leak': 'Water Supply', 'pipe': 'Sanitation',
    'garbage': 'Sanitation', 'waste': 'Sanitation', 'dirty': 'Sanitation',
    'electricity': 'Power & Energy', 'light': 'Power & Energy', 'power': 'Power & Energy',
    'school': 'Education', 'teacher': 'Education', 'class': 'Education',
    'hospital': 'Healthcare', 'doctor': 'Healthcare', 'medicine': 'Healthcare',
    'bribe': 'Corruption', 'money': 'Corruption', 'fraud': 'Corruption',
    'police': 'Law Enforcement', 'theft': 'Law Enforcement', 'crime': 'Law Enforcement',
    'bus': 'Public Transport', 'train': 'Public Transport', 'metro': 'Public Transport',
    'tax': 'Finance & Tax', 'refund': 'Finance & Tax', 'loan': 'Finance & Tax'
}

MASTER_COMMENT_LIST = []
word_cloud_buffer = None

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned_tokens)

def extract_key_topics(text_input, vectorizer=None, text_vector=None):
    extracted_topics = set()
    text_lower = text_input.lower()

    for keyword, category in GOV_TOPIC_MAP.items():
        if keyword in text_lower:
            extracted_topics.add(category)

    if vectorizer and text_vector is not None:
        try:
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = text_vector.toarray().flatten()
            top_indices = tfidf_scores.argsort()[-3:][::-1]
            for i in top_indices:
                if tfidf_scores[i] > 0:
                    extracted_topics.add(feature_names[i])
        except Exception: pass

    if len(extracted_topics) < 2:
        try:
            words = nltk.word_tokenize(text_lower)
            keywords = [word for word in words if word.isalnum() and word not in stop_words and len(word) > 3]
            freq = nltk.FreqDist(keywords)
            for w, c in freq.most_common(2):
                extracted_topics.add(w.capitalize())
        except Exception: pass

    final_topics = list(extracted_topics)
    return final_topics[:3] if final_topics else ["General Feedback"]

def create_word_cloud(comments_list):
    if not comments_list: return None
    text_data = " ".join([str(c['text']) for c in comments_list])
    if not text_data or len(text_data) < 5: return None
    try:
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text_data)
        img_buffer = io.BytesIO()
        wordcloud.to_image().save(img_buffer, format='PNG')
        img_buffer.seek(0)
        return img_buffer
    except Exception as e:
        return None

# --- 5. MASTER ANALYSIS FUNCTION ---
def analyze_comment_hybrid(text_input):
    analysis_source = "Basic Analysis"
    start_time = time.time()
    cleaned_text = preprocess_text(text_input)
    sentiment = None
    confidence = "0%"
    vectorized_text = None 
    
    models_ready = all([lgbm_model, vectorizer, tokenizer, transformer_model])
    
    if any(keyword in text_input.lower() for keyword in URGENT_KEYWORDS):
        sentiment = "Urgent Negative"
        confidence = "100%"
        analysis_source = "Urgency Protocol (Keyword)"
    elif len(text_input.split()) < 4 and ("no comment" in text_input.lower() or "none" in text_input.lower()):
        sentiment = "Neutral"
        confidence = "100%"
        analysis_source = "Filter (Non-Comment)"
    elif sentiment is None:
        vader_scores = vader_analyzer.polarity_scores(text_input)
        if vader_scores['neu'] > 0.85 and abs(vader_scores['compound']) < 0.1:
            sentiment = "Neutral"
            confidence = "100%"
            analysis_source = "Factual Detector (VADER)"
    
    if sentiment is None and models_ready:
        try:
            vectorized_text = vectorizer.transform([cleaned_text])
            prediction_proba = lgbm_model.predict_proba(vectorized_text)[0]
            confidence_val = prediction_proba.max()
            sentiment = lgbm_model.classes_[prediction_proba.argmax()]
            confidence = f"{confidence_val:.0%}"

            if confidence_val < CONFIDENCE_THRESHOLD:
                analysis_source = "Smart Model (Deep Scan)"
                inputs = tokenizer(text_input, return_tensors="pt", truncation=True, max_length=512)
                outputs = transformer_model(**inputs)
                scores = outputs.logits[0].softmax(0).detach().numpy()
                confidence_val = scores.max()
                confidence = f"{confidence_val:.0%}"
                model_sentiment = outputs.logits.argmax().item()
                sentiment = "Positive" if model_sentiment == 1 else "Negative"
                if confidence_val < 0.60: sentiment = "Neutral"
            else:
                analysis_source = "Fast Model (LGBM)"
        except Exception as e:
            sentiment = "Neutral"

    if sentiment is None:
        vs = vader_analyzer.polarity_scores(text_input)
        if vs['compound'] >= 0.05: sentiment = "Positive"
        elif vs['compound'] <= -0.05: sentiment = "Negative"
        else: sentiment = "Neutral"
        confidence = "80% (VADER Fallback)"
        analysis_source = "Backup System"

    topics = extract_key_topics(text_input, vectorizer, vectorized_text)

    summary = "Summary unavailable."
    try:
        if summarizer and len(text_input.split()) > 25:
            summary = summarizer(text_input, max_length=25, min_length=5, do_sample=False)[0]['summary_text']
        else:
            summary = text_input 
    except Exception: pass

    end_time = time.time()
    return {
        'sentiment': sentiment, 'confidence': confidence, 'key_topics': topics,
        'summary': summary, 'analysis_source': analysis_source,
        'analysis_time': f"{(end_time - start_time):.2f}s"
    }

# --- LIVE SIMULATOR ---
LIVE_STREAM_QUEUE = [
    "The portal is crashing when I try to upload my PDF.", "Section 4.2 is brilliant.",
    "Urgent: The deadline is tomorrow!", "I disagree with the new tax implication.",
    "Smooth experience submitting the form.", "Compliance is too strict for startups.",
    "Error 404 on the main page.", "Positive step towards modernization.",
    "Font size is too small.", "Critical security flaw found."
]

def background_feed_simulator():
    global MASTER_COMMENT_LIST, word_cloud_buffer
    while True:
        delay = random.randint(15, 30)
        time.sleep(delay)
        if LIVE_STREAM_QUEUE:
            try:
                text = random.choice(LIVE_STREAM_QUEUE)
                with app.app_context():
                    results = analyze_comment_hybrid(text)
                    results['id'] = f"Live #{len(MASTER_COMMENT_LIST) + 1}"
                    results['text'] = text
                    results['topics'] = results.pop('key_topics')
                    MASTER_COMMENT_LIST.insert(0, results)
                    word_cloud_buffer = create_word_cloud(MASTER_COMMENT_LIST)
            except Exception: pass

def startup_task():
    global MASTER_COMMENT_LIST, word_cloud_buffer
    demo_data = [
        {"id": "#8492", "text": "The new penalty for fraud in Section 2.1a is a joke."},
        {"id": "#8493", "text": "I fully support the new transparency guidelines."},
        {"id": "#8494", "text": "The proposal for the new park is nice but expensive."},
        {"id": "#8495", "text": "The data sharing clause is unacceptable."},
        {"id": "#8496", "text": "The road is full of potholes. Hazard."}
    ]
    temp_list = []
    log_msg("--- STARTING DATA POPULATION ---")
    for comment in demo_data:
        try:
            res = analyze_comment_hybrid(comment["text"])
            res['id'] = comment['id']
            res['text'] = comment['text']
            res['topics'] = res.pop('key_topics')
            temp_list.append(res)
        except Exception: pass
    MASTER_COMMENT_LIST = temp_list
    if MASTER_COMMENT_LIST:
        word_cloud_buffer = create_word_cloud(MASTER_COMMENT_LIST)
    log_msg(f"✅ Startup Complete. Loaded {len(MASTER_COMMENT_LIST)} items.")
    simulator_thread = threading.Thread(target=background_feed_simulator, daemon=True)
    simulator_thread.start()

# --- 6. ROUTES & NEW ADMIN PORTAL ---
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user, remember=True)
            return redirect(url_for('home'))
        else:
            flash('Invalid Password.')

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# --- NEW: SUPER ADMIN DASHBOARD ---
# Allows you to create new officials dynamically!
@app.route('/admin/dashboard', methods=['GET', 'POST'])
@login_required
def admin_dashboard():
    # SECURITY LOCK: Kick out anyone who isn't admin
    if current_user.username != 'admin':
        flash("⚠️ SECURITY ALERT: You are not authorized to access the Admin Panel.", "error")
        return redirect(url_for('home')) # <--- This is the key change: Redirects back to safety

    if request.method == 'POST':
        new_user = request.form.get('new_username')
        new_pass = request.form.get('new_password')
        
        if new_user and new_pass:
            existing = User.query.filter_by(username=new_user).first()
            if not existing:
                hashed = generate_password_hash(new_pass)
                official = User(username=new_user, password_hash=hashed)
                db.session.add(official)
                db.session.commit()
                flash(f"✅ Official '{new_user}' created successfully!")
            else:
                flash(f"❌ User '{new_user}' already exists.")

    # Get list of all officials
    users = User.query.all()
    
    # Simple embedded HTML for the Admin Panel
    html = """
    <html>
    <head><title>Super Admin Panel</title>
    <style>
        body { font-family: sans-serif; padding: 40px; max_width: 800px; margin: auto; background: #f4f6f9; }
        .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        input { padding: 10px; margin: 5px 0; width: 100%; border: 1px solid #ddd; border-radius: 4px; }
        button { background: #007bff; color: white; padding: 10px 20px; border: none; cursor: pointer; border-radius: 4px; }
        button:hover { background: #0056b3; }
        table { width: 100%; margin-top: 20px; border-collapse: collapse; }
        th, td { text-align: left; padding: 12px; border-bottom: 1px solid #ddd; }
        .back-btn { display: inline-block; margin-bottom: 20px; text-decoration: none; color: #333; }
    </style>
    </head>
    <body>
        <a href="/" class="back-btn">← Back to Dashboard</a>
        <h2>👮‍♂️ Super Admin: Manage Officials</h2>
        
        {% with messages = get_flashed_messages() %}
          {% if messages %}
            <div style="background: #d4edda; color: #155724; padding: 10px; margin-bottom: 20px; border-radius: 4px;">
              {{ messages[0] }}
            </div>
          {% endif %}
        {% endwith %}

        <div class="card">
            <h3>Register New Official</h3>
            <form method="POST">
                <label>Username</label>
                <input type="text" name="new_username" required placeholder="e.g. judge_01">
                <label>Password</label>
                <input type="password" name="new_password" required placeholder="Enter secure password">
                <button type="submit">Create Account</button>
            </form>
        </div>

        <div class="card" style="margin-top: 20px;">
            <h3>Active Officials</h3>
            <table>
                <tr><th>ID</th><th>Username</th><th>Status</th></tr>
                {% for u in users %}
                <tr>
                    <td>{{ u.id }}</td>
                    <td>{{ u.username }}</td>
                    <td><span style="color: green;">● Active</span></td>
                </tr>
                {% endfor %}
            </table>
        </div>
    </body>
    </html>
    """
    return render_template_string(html, users=users)
    
@app.route('/')
@login_required
def home():
    return render_template('index.html', user=current_user)

@app.route('/settings')
@login_required
def settings():
    return render_template('setting.html', user=current_user)

@app.route('/get_all_comments', methods=['GET'])
@login_required
def get_all_comments():
    return jsonify(MASTER_COMMENT_LIST)

@app.route('/wordcloud.png')
@login_required
def get_word_cloud():
    filter_type = request.args.get('sentiment', 'all')
    if filter_type == 'all':
        filtered_list = MASTER_COMMENT_LIST
    else:
        target = filter_type.replace('-', ' ') 
        filtered_list = [c for c in MASTER_COMMENT_LIST if c['sentiment'].lower() == target]
    if not filtered_list: return "No data", 404
    img_buffer = create_word_cloud(filtered_list)
    if img_buffer: return send_file(img_buffer, mimetype='image/png')
    return "Error", 500

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    data = request.get_json()
    text_input = data.get('text', '')
    if not text_input: return jsonify({'error': 'No text.'}), 400
    try:
        results = analyze_comment_hybrid(text_input)
        new_entry = results.copy()
        new_entry['id'] = f"New #{len(MASTER_COMMENT_LIST) + 1}"
        new_entry['text'] = text_input
        new_entry['topics'] = new_entry.pop('key_topics')
        MASTER_COMMENT_LIST.insert(0, new_entry)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload_csv', methods=['POST'])
@login_required
def upload_csv():
    global MASTER_COMMENT_LIST, word_cloud_buffer
    if 'file' not in request.files: return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({'error': 'No file selected'}), 400
    try:
        if file.filename.endswith('.csv'): df = pd.read_csv(file)
        elif file.filename.endswith(('.xls', '.xlsx')): df = pd.read_excel(file)
        else: return jsonify({'error': 'Invalid file type.'}), 400
        possible_cols = ['text', 'comment', 'feedback', 'suggestion']
        text_col = next((col for col in df.columns if col.lower() in possible_cols), None)
        if not text_col: return jsonify({'error': 'Column missing (text/comment).'}), 400
        comments_to_process = df[text_col].fillna('').astype(str).tolist()
        comments_to_process = [c for c in comments_to_process if len(c.strip()) > 3][:500] 
        processed_results = []
        def process_single_row(comment_text):
            try:
                res = analyze_comment_hybrid(comment_text)
                res['id'] = "Upload"
                res['text'] = comment_text
                res['topics'] = res.pop('key_topics')
                return res
            except: return None
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = executor.map(process_single_row, comments_to_process)
            for res in results:
                if res: processed_results.append(res)
        for i, res in enumerate(processed_results):
            res['id'] = f"Upload #{len(MASTER_COMMENT_LIST) + i + 1}"
        for res in reversed(processed_results):
            MASTER_COMMENT_LIST.insert(0, res)
        if MASTER_COMMENT_LIST:
            word_cloud_buffer = create_word_cloud(MASTER_COMMENT_LIST)
        return jsonify({'message': f'Successfully analyzed {len(processed_results)} comments!', 'count': len(processed_results)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# EXECUTE STARTUP
startup_task()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)