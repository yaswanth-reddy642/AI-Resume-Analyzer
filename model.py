from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(resume_text, job_description):

    documents = [resume_text, job_description]

    tfidf = TfidfVectorizer()

    tfidf_matrix = tfidf.fit_transform(documents)

    similarity = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:2]
    )

    ats_score = round(similarity[0][0] * 100, 2)

    # Missing keywords detection
    resume_words = set(resume_text.lower().split())
    jd_words = set(job_description.lower().split())

    missing_keywords = list(jd_words - resume_words)

    return ats_score, missing_keywords[:10]