import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Muat model SVM
model_svm = joblib.load('best_svm_model.pkl')

# Fungsi untuk membaca file dan mengembalikan list
def read_file_to_list(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file]

# Membaca data dari file
list_data_training = read_file_to_list("data_training.txt")
list_data_lda = read_file_to_list("data_lda.txt")
stopwords = read_file_to_list("streamlit_stopwords.txt")

# BOW dan TF-IDF Vectorizer
vectorizer = CountVectorizer(stop_words=stopwords, min_df=5, ngram_range=(1, 2))
tfidf = TfidfVectorizer(min_df=2, max_df=0.8, sublinear_tf=True, use_idf=True, ngram_range=(1, 2))
tfidf.fit_transform(pd.Series(list_data_training))

# Pemetaan aspek
aspek_mapping = {
    0: 'Kualitas Makanan',
    1: 'Lingkungan Fisik',
    2: 'Harga',
    3: 'Kualitas Pelayanan',
}

# Fungsi prediksi
def predict(teks):
    # Transformasi teks ke TF-IDF
    X_test_tfidf = tfidf.transform(pd.Series(teks))
    predicted = model_svm.predict(X_test_tfidf)

    # Tambahkan teks ke LDA data list
    list_data_lda.append(teks)
    bow_matrix = vectorizer.fit_transform(list_data_lda)

    # LDA Model
    lda_4 = LatentDirichletAllocation(n_components=4, doc_topic_prior=0.25, topic_word_prior=0.25, random_state=42)
    topic_distributions = lda_4.fit_transform(bow_matrix)

    # Temukan topik dengan probabilitas tertinggi
    data_lda_terbaru = topic_distributions[-1]
    index_highest_prob = np.argmax(data_lda_terbaru)

    # Menentukan sentimen
    sentimen = "Sentimen Positif" if np.round(predicted) == 1 else "Sentimen Negatif"
    aspek = aspek_mapping[index_highest_prob]

    return sentimen, aspek