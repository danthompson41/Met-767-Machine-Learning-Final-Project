# Dependency management and python versions are found in pyproject.toml. this was run with phttps://python-poetry.org/

print("Loading imports, this can take a while")
import os
print(".", end =" ")
import pandas as pd
print(".", end =" ")
from sklearn.datasets import fetch_20newsgroups
print(".", end =" ")
from sklearn.metrics.pairwise import cosine_similarity
print(".", end =" ")
import umap
print(".", end =" ")
import hdbscan
print(".", end =" ")
import re
print(".", end =" ")
from nltk.corpus import stopwords
print(".", end =" ")
import numpy as np
print(".", end =" ")
from sklearn.feature_extraction.text import CountVectorizer
print(".", end =" ")
from sentence_transformers import SentenceTransformer
print(".", end =" ")

# Load the data, from ../../data/2_raw_text
data = []
names = []
print("Loading Data")
for root, dirs, files in os.walk("../../data/2_raw_text"):
    print(f"-Dir: {root}")
    for dir in dirs:
        print(f"--{dir}")
    for filename in files:
        with open(f"{root}/{filename}") as f:
            data.append(f.read())
            names.append(filename)
# Create Embeddings from the raw data, using distilbert
print("Files Loading, loading distilbert sentence transformer")
model = SentenceTransformer('distilbert-base-nli-mean-tokens')
embeddings = model.encode(data, show_progress_bar=True)

# Use UMAP to lower the dimensions of the data for HDBScan. 
# DBScan based algorithms don't work well with high dimensional data
reduced_embeddings = umap.UMAP(n_neighbors=5, 
                               n_components=5, 
                               metric='cosine').fit_transform(embeddings)

# Run HDBScan on the reduced embeddings
cluster = hdbscan.HDBSCAN(min_cluster_size=3,
                          metric='euclidean',                      
                          cluster_selection_method='eom').fit(reduced_embeddings)

# Load the bills into a data frame, with the topic numbers and some metadata
df = pd.DataFrame(data, columns=["Bill"])
df['Topic'] = cluster.labels_
df['Bill_Id'] = range(len(df))
bills_per_topic = df.groupby(['Topic'], as_index = False).agg({'Bill': ' '.join})
df["Bill_Number"] = [i.split("_")[0] for i in names]
df["File_Name"] = names

# Sanitize the bills, removing non-alphabetical characters and stopwords
def bill_sanitizer(raw_bill):
    letters_only = re.sub('[^a-zA-Z]', ' ', raw_bill)
    words = letters_only.lower().split()
    stopwords_set = set(stopwords.words("english"))
    nonstopwords = [x for x in words if not x in stopwords_set]
    
    # Combine words into a paragraph again
    output_strings = ' '.join(nonstopwords)
    return(output_strings)
cleaned_bills_per_topic = [bill_sanitizer(i) for i in bills_per_topic.Bill.values]

def extract_top_words(tf_idf, count, bills_per_topic, n=5):
    words = count.get_feature_names()
    labels = list(bills_per_topic.Topic)
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
    return top_n_words

def extract_cluster_sizes(df):
    topic_sizes = (df.groupby(['Topic'])
                     .Bill
                     .count()
                     .reset_index()
                     .rename({"Topic": "Topic", "Bill": "Size"}, axis='columns')
                     .sort_values("Size", ascending=False))
    return topic_sizes

def cluster_tf_idf(documents, m, ngram_range=(1, 1)):
    count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)

    return tf_idf, count
  
tf_idf, count = cluster_tf_idf(cleaned_bills_per_topic, m=len(data))

top_n_words = extract_top_words(tf_idf, count, bills_per_topic, n=10)
topic_sizes = extract_cluster_sizes(df)

# Drop the bill content and ID, since we only 
# care about topic and bill number / file name
df2 = df.drop("Bill",axis=1)
df2 = df2.drop("Bill_Id",axis=1)
print(df2)

# Write out a map of what bills map to what topics
json_data = df2.to_json(orient="split")
import json
parsed = json.loads(json_data)
with open ("../../data/4_clustering/clustering_bills.json", "w") as f:
    json.dump(parsed, f)
    
# Write out a map of key words that map to each topic
with open ("../../data/4_clustering/topic_map.json", "w") as f:
    json.dump(top_n_words, f)
