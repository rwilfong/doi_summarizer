# DOI Summarizer and Recommender 

This is live on Flask: rwilfong.pythonanywhere.com

This program takes a list of DOI's from user input, retrieves their abstracts, and generates a pair of summaries and keywords based on the text in the abstract. It has a content-based recommender system based on the generated keywords for each paper. It was created for BIOL 595: Practical Biocomputing at Purdue University.

### Required Libraries
1. ``Flask:`` 
  - Flask
  - render_template
  - request
  - redirect
  - url_for
  - session 
2. ``Datetime:`` 
  - datetime
3. ``Collections:``
  - defaultdict 
4. ``scikit-learn:``
  - sklearn.feature_extraction.text: CountVectorizer, TfidfVectorizer
  - sklearn.metrics.pairwise: cosine_similarity
5. ``NLTK:``
  - nltk.tokenize: sent_tokenize
  - nltk.collocations: BigramAssocMeasures, BigramCollocationFinder
  - nltk.corpus import stopwords
6. ``NetworkX``
7. ``NumPy``
8. ``xml.etree.ElementTree``
9. ``sqlite3``
10. ``Biopython:``
  - Entrez
11. ``string:``
  - punctuation 

