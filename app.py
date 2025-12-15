import streamlit as st
import os, shutil, zipfile, re
import pdfplumber
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

# ---------------------------------------------------
# PAGE CONFIG (MUST BE FIRST)
# ---------------------------------------------------
st.set_page_config(
    page_title="HireSense AI",
    layout="wide",
    page_icon="🧠"
)

# ---------------------------------------------------
# 🧪 TRIAL CONFIG
# ---------------------------------------------------
TRIAL_MODE = True
MAX_RUNS = 1
USAGE_FILE = "usage_count.txt"

# ---------------------------------------------------
# 🔐 PASSWORD PROTECTION
# ---------------------------------------------------
def check_password():
    def password_entered():
        if st.session_state["password"] == st.secrets["app_password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("🔐 Enter Access Password", type="password",
                      on_change=password_entered, key="password")
        st.stop()
    elif not st.session_state["password_correct"]:
        st.text_input("🔐 Enter Access Password", type="password",
                      on_change=password_entered, key="password")
        st.error("❌ Incorrect Password")
        st.stop()

check_password()

# ---------------------------------------------------
# TRIAL USAGE FUNCTIONS
# ---------------------------------------------------
def get_usage_count():
    if not os.path.exists(USAGE_FILE):
        return 0
    try:
        with open(USAGE_FILE, "r") as f:
            c = f.read().strip()
            return int(c) if c.isdigit() else 0
    except:
        return 0

def increment_usage():
    with open(USAGE_FILE, "w") as f:
        f.write(str(get_usage_count() + 1))

# ---------------------------------------------------
# INITIAL SETUP
# ---------------------------------------------------
nltk.download("stopwords")
client = Groq(api_key=st.secrets["groq_api_key"])

# ---------------------------------------------------
# UI STYLE
# ---------------------------------------------------
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #005bea, #00c6fb);
    padding: 25px;
    border-radius: 15px;
}
.stButton>button {
    background: #005bea;
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HELPERS
# ---------------------------------------------------
def clear_resume_folder(path="resumes"):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def extract_zip(zip_file, extract_to="resumes"):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    for f in os.listdir(extract_to):
        if not f.lower().endswith(".pdf"):
            os.remove(os.path.join(extract_to, f))

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for p in pdf.pages:
            if p.extract_text():
                text += p.extract_text() + " "
    return text

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
    stop_words = set(stopwords.words("english"))
    return " ".join(w for w in text.split() if w not in stop_words)

def groq_generate(prompt):
    return client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content

def extract_skills_from_jd(jd):
    result = groq_generate(
        f"Extract ONLY technical job skills. Return comma-separated list.\n{jd}"
    )
    return [s.strip().lower() for s in result.split(",") if s.strip()]

def extract_experience(text):
    yrs = re.findall(r'(\d+)\s*(?:years?|yrs?)', text.lower())
    return max(map(int, yrs)) if yrs else 0

def extract_email(text):
    m = re.findall(r"[\\w\\.-]+@[\\w\\.-]+", text)
    return m[0] if m else "Not Found"

def extract_phone(text):
    m = re.findall(r"\+?\d[\d\s\-()]{8,}\d", text)
    return m[0] if m else "Not Found"

# ---------------------------------------------------
# MAIN UI
# ---------------------------------------------------
st.markdown("<div class='main-header'><h1 style='color:white'>🧠 HireSense AI</h1></div>",
            unsafe_allow_html=True)

if TRIAL_MODE:
    st.info(f"🧪 Trial Mode — {MAX_RUNS - get_usage_count()} screening(s) remaining")

c1, c2 = st.columns(2)
zip_file = c1.file_uploader("Upload Resume ZIP (.zip)", type=["zip"])
jd_file = c2.file_uploader("Upload Job Description (.txt)", type=["txt"])

# ---------------------------------------------------
# RUN SCREENING
# ---------------------------------------------------
if st.button("🚀 Run Screening"):

    if TRIAL_MODE and get_usage_count() >= MAX_RUNS:
        st.error("🚫 Trial limit reached. Upgrade to Pro.")
        st.stop()

    if not zip_file or not jd_file:
        st.error("Please upload both files.")
        st.stop()

    clear_resume_folder()
    extract_zip(zip_file)

    jd_text = jd_file.read().decode("utf-8")
    cleaned_jd = clean_text(jd_text)
    skills_list = extract_skills_from_jd(jd_text)

    resumes = {f: extract_text_from_pdf(f"resumes/{f}") for f in os.listdir("resumes")}
    cleaned = {k: clean_text(v) for k, v in resumes.items()}

    vec = TfidfVectorizer()
    mat = vec.fit_transform([cleaned_jd] + list(cleaned.values()))
    scores = cosine_similarity(mat[0:1], mat[1:]).flatten()

    df = pd.DataFrame({
        "Candidate": list(cleaned.keys()),

        "Match Score": (scores * 100).round(2),

        "Skills Found": [
            ", ".join(skill for skill in skills_list if skill in cleaned[k])
            if cleaned.get(k) else "None"
            for k in cleaned
        ],

        "Missing Skills": [
            ", ".join(skill for skill in skills_list if skill not in cleaned[k])
            if cleaned.get(k) else "None"
            for k in cleaned
        ],

        "Experience": [
            extract_experience(resumes[k]) if resumes.get(k) else 0
            for k in cleaned
        ],

        "Email": [
            extract_email(resumes[k]) if resumes.get(k) else "Not Found"
            for k in cleaned
        ],

        "Phone": [
            extract_phone(resumes[k]) if resumes.get(k) else "Not Found"
            for k in cleaned
        ],
    })

    df["Overall Score"] = (
        0.5 * df["Match Score"]/100 +
        0.3 * (df["Skills Found"].str.count(",") + 1) / max(len(skills_list),1) +
        0.2 * (df["Experience"]/10).clip(upper=1)
    ) * 100

    df = df.sort_values("Overall Score", ascending=False)
    increment_usage()

    # ---------------------------------------------------
    # DASHBOARD (OLD STYLE)
    # ---------------------------------------------------
    st.header("🏆 AI Candidate Ranking Dashboard")

    def color_score(v):
        if v >= 25: return "background-color:#4ade80"
        if v >= 15: return "background-color:#fb923c"
        return "background-color:#f87171"

    st.dataframe(df.style.applymap(color_score, subset=["Overall Score"]))

    # ---------------------------------------------------
    # INTERVIEW QUESTIONS
    # ---------------------------------------------------
    st.subheader("🎤 AI-Generated Interview Questions")
    st.write(groq_generate(f"Generate 8 interview questions:\n{jd_text}"))

    # ---------------------------------------------------
    # RESUME SUMMARIES
    # ---------------------------------------------------
    st.subheader("📄 AI Resume Summaries")
    for c, t in resumes.items():
        st.markdown(f"### {c}")
        st.write(groq_generate(f"Summarize this resume:\n{t}"))

    # ---------------------------------------------------
    # BAR CHART
    # ---------------------------------------------------
    st.subheader("📊 Candidate Ranking Bar Chart")
    st.bar_chart(df.set_index("Candidate")["Overall Score"])
    
    # Heatmap
    #st.subheader("🟩 Skills Heatmap")
    #heatmap_data = [
        #[1 if s in cleaned_resumes[candidate] else 0 for s in skills_list]
        #for candidate in df["Candidate"]
    #]
    #heatmap_df = pd.DataFrame(heatmap_data, columns=skills_list, index=df["Candidate"])
    #st.dataframe(heatmap_df.style.background_gradient(cmap='Blues'))