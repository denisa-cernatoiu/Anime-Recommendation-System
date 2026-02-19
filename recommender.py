import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import nltk

for resource in ['punkt_tab', 'wordnet', 'omw-1.4', 'averaged_perceptron_tagger_eng']:
    nltk.download(resource, quiet=True)


class AnimeRecommendation():
    def __init__(self):
        self.tfidf_vect = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = None
        self.lemmatizer = WordNetLemmatizer()

    def get_wordnet_pos(self, tag):
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

    def lemmatize_description(self, description_col: str):
        tokens = word_tokenize(description_col)
        tagged_tokens = pos_tag(tokens)
        lemmatized_sentence = []
        for word, tag in tagged_tokens:
            if word.lower() in ['are', 'is', 'am']:
                lemmatized_sentence.append(word)
            else:
                lemmatized_sentence.append(
                    self.lemmatizer.lemmatize(word, self.get_wordnet_pos(tag))
                )
        return ' '.join(lemmatized_sentence)

    def fit(self, train_df: pd.DataFrame, description_col: str, title_col: str):
        self.train_df = train_df
        self.description_col = description_col
        self.title_col = title_col

        self.train_df = train_df.set_index(title_col)
        self.train_df[self.description_col] = self.train_df[self.description_col].apply(self.lemmatize_description)
        self.tfidf_matrix = self.tfidf_vect.fit_transform(self.train_df[self.description_col])
     

    def get_recommendations(self, title: str):
        if self.tfidf_matrix is None:
            raise RuntimeError("Model is not fitted yet.")
        
        idx = self.train_df.index.get_loc(title)
        cos_sim = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        similarity_scores = pd.Series(cos_sim, index=self.train_df.index)
        top_10 = similarity_scores.sort_values(ascending=False).iloc[1:11].index
        
        return list(top_10)

    def save(self, path='model.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path='model.pkl'):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        instance = cls.__new__(cls)
        instance.__dict__.update(obj.__dict__)
        return instance


# training the model 
if __name__ == "__main__":
    import recommender
    df = pd.read_csv("anime_dataset_cleaned.csv")
    train_df = pd.DataFrame(df, columns=['title_english', 'description'])
    
    rec = recommender.AnimeRecommendation()
    rec.fit(train_df,title_col='title_english', description_col='description')
    rec.save()
    print('Model trained!')
