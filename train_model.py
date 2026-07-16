import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

print("1. Učitavanje i čišćenje podataka...")
df = pd.read_csv('products.csv')
df.columns = df.columns.str.strip()
df_model = df[['Product Title', ' Category Label']].copy()
df_model = df_model.dropna(subset=['Product Title', ' Category Label'])

df_model['Product Title'] = df_model['Product Title'].str.lower()
df_model[' Category Label'] = df_model[' Category Label'].str.title()

mapiranje_kategorija = {
    'Mobile Phone': 'Mobile Phones',
    'Cpu': 'CPUs',
    'Fridge': 'Fridges'
}
df_model[' Category Label'] = df_model[' Category Label'].replace(mapiranje_kategorija)

X = df_model['Product Title']
y = df_model[' Category Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("2. Vektorizacija teksta (TF-IDF)...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)

print("3. Treniranje Logističke regresije...")
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)

# Cuvamo model i vektorizer na disk da bi predict.category.py mogao da ih koristi
print("4. Čuvanje modela na disk...")
with open('model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Sinhronizacija završena! Model i vektorizer su uspešno sačuvani.")