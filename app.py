import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os

from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="Smart Recruitment Intelligence Platform",
    layout="wide"
)

st.title("Smart Recruitment Intelligence Platform")

# -----------------------------------
# LOAD FILES
# -----------------------------------

@st.cache_resource
def load_models():

    model = tf.keras.models.load_model(
        "attention_resume_model.keras"
    )

    tokenizer = joblib.load(
        "tokenizer.pkl"
    )

    label_encoder = joblib.load(
        "label_encoder.pkl"
    )

    tfidf = joblib.load(
        "tfidf.pkl"
    )

    return (
        model,
        tokenizer,
        label_encoder,
        tfidf
    )

model, tokenizer, label_encoder, tfidf = load_models()

# -----------------------------------
# CLEANING
# -----------------------------------

import re

def clean_text(text):

    text = str(text)

    text = re.sub(
        r'http\\S+',
        ' ',
        text
    )

    text = re.sub(
        r'www\\S+',
        ' ',
        text
    )

    text = re.sub(
        r'[^a-zA-Z ]',
        ' ',
        text
    )

    text = text.lower()

    return text

# -----------------------------------
# SKILLS DATABASE
# -----------------------------------

skills_db = [

    "python",
    "sql",
    "machine learning",
    "deep learning",
    "tensorflow",
    "keras",
    "pytorch",
    "nlp",
    "tableau",
    "power bi",
    "aws",
    "azure",
    "docker",
    "kubernetes",
    "java",
    "javascript",
    "html",
    "css",
    "react",
    "nodejs"

]

def extract_skills(text):

    text = text.lower()

    found = []

    for skill in skills_db:

        if skill in text:

            found.append(skill)

    return list(set(found))

# -----------------------------------
# SKILL MATCH
# -----------------------------------

def calculate_skill_match(

    resume_skills,

    jd_skills

):

    if len(jd_skills) == 0:

        return 0

    matched = len(

        set(resume_skills)

        &

        set(jd_skills)

    )

    return round(

        (matched / len(jd_skills)) * 100,

        2

    )

# -----------------------------------
# SIDEBAR
# -----------------------------------

st.sidebar.header("Upload Files")

jd_file = st.sidebar.file_uploader(
    "Upload Job Description (.txt)",
    type=["txt"]
)

resume_files = st.sidebar.file_uploader(
    "Upload Resumes (.txt)",
    type=["txt"],
    accept_multiple_files=True
)

# -----------------------------------
# MAIN
# -----------------------------------

if jd_file and resume_files:

    jd_text = jd_file.read().decode()

    jd_clean = clean_text(jd_text)

    jd_skills = extract_skills(
        jd_clean
    )

    jd_vector = tfidf.transform(
        [jd_clean]
    )

    results = []

    progress = st.progress(0)

    for idx, file in enumerate(resume_files):

        resume_text = file.read().decode()

        resume_clean = clean_text(
            resume_text
        )

        # CATEGORY

        seq = tokenizer.texts_to_sequences(
            [resume_clean]
        )

        seq = pad_sequences(
            seq,
            maxlen=300,
            padding="post",
            truncating="post"
        )

        pred = model.predict(
            seq,
            verbose=0
        )

        category = label_encoder.inverse_transform(
            [np.argmax(pred)]
        )[0]

        # SIMILARITY

        resume_vector = tfidf.transform(
            [resume_clean]
        )

        similarity = cosine_similarity(
            jd_vector,
            resume_vector
        )[0][0]

        # SKILLS

        resume_skills = extract_skills(
            resume_clean
        )

        skill_match = calculate_skill_match(
            resume_skills,
            jd_skills
        )

        # FINAL SCORE

        final_score = (

            similarity * 70

            +

            skill_match * 0.30

        )

        results.append({

            "Resume": file.name,

            "Predicted Category": category,

            "Similarity Score":

                round(
                    similarity * 100,
                    2
                ),

            "Skill Match %":

                skill_match,

            "Final Score":

                round(
                    final_score,
                    2
                ),

            "Matched Skills":

                ", ".join(

                    list(

                        set(resume_skills)

                        &

                        set(jd_skills)

                    )

                )

        })

        progress.progress(

            (idx + 1)

            /

            len(resume_files)

        )

    result_df = pd.DataFrame(
        results
    )

    result_df = result_df.sort_values(

        by="Final Score",

        ascending=False

    )

    st.success(
        "Ranking Completed"
    )

    st.subheader(
        "Top Candidates"
    )

    st.dataframe(
        result_df,
        use_container_width=True
    )

    # DOWNLOAD

    csv = result_df.to_csv(
        index=False
    )

    st.download_button(

        label="Download Hiring Report",

        data=csv,

        file_name="hiring_report.csv",

        mime="text/csv"

    )

    # TOP CANDIDATE

    st.subheader(
        "Best Candidate"
    )

    st.write(
        result_df.iloc[0]
    )

# -----------------------------------
# HEATMAPS
# -----------------------------------

st.sidebar.header(
    "Project Assets"
)

if os.path.exists(
    "positional_heatmap.png"
):

    st.sidebar.image(
        "positional_heatmap.png",
        caption="Positional Encoding"
    )

if os.path.exists(
    "candidate_scores.png"
):

    st.sidebar.image(
        "candidate_scores.png",
        caption="Candidate Scores"
    )