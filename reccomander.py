import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from rapidfuzz import process



df = pd.read_csv("anime_dataset_cleaned.csv")


train_df = pd.DataFrame(df, columns=['title_english', 'description'])
train_df = train_df.set_index('title_english')


"""lemmatization"""

lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return 'a'
    elif tag.startswith('V'):
        return 'v'
    elif tag.startswith('N'):
        return 'n'
    elif tag.startswith('R'):
        return 'r'
    else:
        return 'n'


def lemmatize_description(description):
  tokens = word_tokenize(description)
  tagged_tokens = pos_tag(tokens)
  lemmatized_sentence = []
  for word, tag in tagged_tokens:
      if word.lower() == 'are' or word.lower() in ['is', 'am']:
          lemmatized_sentence.append(word)
      else:
          lemmatized_sentence.append(
              lemmatizer.lemmatize(word, get_wordnet_pos(tag)))

  return ' '.join(lemmatized_sentence)

train_df['description'] = train_df['description'].apply(lambda x: lemmatize_description(x))


"""tf-idf vectorization"""


tfidfvec = TfidfVectorizer()
desc_vect = tfidfvec.fit_transform((train_df["description"]))


cos_sim = cosine_similarity(desc_vect, desc_vect)




indices = pd.Series(range(len(train_df)), index=train_df.index)
titles = train_df.index

def find_closest_title(user_title, all_titles, threshold=70):
    match = process.extractOne(user_title, all_titles)
    return match[0] if match and match[1] >= threshold else None


def recommendations(title, cosine_sim=cos_sim, range_nr=10, lambda_param=0.7, candidate_pool=50):
    real_title = find_closest_title(title, titles)

    if real_title is None:
        return []

    query_idx = indices[real_title]

    # Get similarity scores from cosine matrix
    sims = cosine_sim[query_idx]

    # Get top candidates (excluding itself)
    candidates = list(np.argsort(sims)[::-1][1:candidate_pool+1])

    selected = []

    while len(selected) < range_nr and candidates:
        mmr_scores = {}

        for c in candidates:
            relevance = sims[c]

            if selected:
                diversity = max(cosine_sim[c][s] for s in selected)
            else:
                diversity = 0

            mmr_scores[c] = lambda_param * relevance - (1 - lambda_param) * diversity

        next_best = max(mmr_scores, key=mmr_scores.get)
        selected.append(next_best)
        candidates.remove(next_best)

    return titles[selected].tolist()