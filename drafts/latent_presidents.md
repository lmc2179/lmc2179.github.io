```python
from nltk.corpus import inaugural
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import pandas as pd

# https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html

rows = []

for fileid in inaugural.fileids():
  year = int(fileid[:4])
  president = fileid[5:-4]
  words = ' '.join(list(inaugural.words(fileid)))
  rows.append([year, president, words])

speech_df = pd.DataFrame(rows, columns=['year', 'president', 'speech'])

vectorizer = TfidfVectorizer()
tf_idf_speeches = vectorizer.fit_transform(speech_df['speech'])
tf_idf_speeches_df = pd.DataFrame(tf_idf_speeches.todense(), columns=vectorizer.get_feature_names())

nmf = NMF(n_components=3)
nmf_speeches = nmf.fit_transform(tf_idf_speeches_df.values)
```
