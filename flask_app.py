"""=================================================================================================
Flask application that will return a html page with input for a list of DOIs, email address, and table name.
Given the list of DOIs, it will use Entrez to search for the matching PMID, return the article. It will then parse the
XML to find the abstract and the title. Using the abstract, it will use two methods: nltk and networkx to return two
summaries of the paper to give the user a better understanding of their paper based on the two results. It will then use
scikit-learn and NLTK to return the top 5 keywords associated with the paper.
This will then write their results to a database and return the output in a new html file formatted as a table.
Then it will recommend papers based on both sets of keywords and MeSH terms and return them below the summarizations.

Rose Wilfong & Wenxuan Dong         05/05/2023
================================================================================================="""
# import libraries
from flask import Flask, render_template, request, redirect, url_for, session
from datetime import datetime
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import networkx as nx
import numpy as np
import xml.etree.ElementTree as ET
import sqlite3
from Bio import Entrez
from string import punctuation
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import urllib.parse

app = Flask(__name__)
app.secret_key = 'pickles_rules'  # set key for Flask session


@app.route('/')
def index():
    return render_template('starting_page.html')  # the HTML file to return when Flask is first deployed


@app.route('/summarizer', methods=['POST', 'GET'])  # summarization portion of Flask
def summarizer():
    if request.method == 'POST':
        # when the form is submitted, analyze the input
        # retrieve entries from the HTML form
        email = request.form['email']
        table_name = request.form['table_name']
        api_key = request.form.get('api_key')
        if api_key:
            pass
        else:
            api_key = 'd881ac932ddb3b61dcb88feac3fcc450af09'  # if the user leaves the API key empty, use this
        # clean input DOIs to remove any whitespaces, new lines, etc.
        doi_list = [doi.strip() for doi in request.form['links'].split(',')]
        # parse DOIs for PMIDS, MeSH terms, abstracts, summaries, and keywords
        cleaned_abstracts = abstract_analysis(doi_list, email, table_name, api_key)
        # get recommendations based on keywords (scikit-learn, NLTK, and MeSH terms)
        recommended = recommend_similar_articles(cleaned_abstracts, email, api_key)
        # save session
        session['cleaned_abstracts'] = cleaned_abstracts
        session['recommended'] = recommended
        # render output when submitted
        return render_template('output.html', results=cleaned_abstracts, recs=recommended)
        # results and recs will be used to return the table in output.html
    else:
        # else, if the form hasn't been submitted, just return the form HTMl file
        return render_template('doi_summarizer.html')


@app.route('/return_data', methods=['GET', 'POST'])
# route for the return your data portion (not the one where you're looking up DOIs and summarizing them)
def return_data():
    if request.method == 'POST':
        # retrieve the form data
        email = request.form['email']
        table_name = request.form['table_name']
        date = request.form['date']
        # connect to the database
        conn = sqlite3.connect('database.db')
        # create a cursor
        cursor = conn.cursor()
        # create the SQL query to return the DOI, title, summary, keywords, and MeSH terms based on the email and table name
        query = "SELECT doi, title, summary, scikit_keywords, nltk_keywords, mesh_terms FROM papers WHERE email = ? AND table_name = ?"
        if date:
            query += " AND date = ?"

        # execute the query based on if the date is used or not
        if date:
            cursor.execute(query, (email, table_name, date))
            # if implemented, it will return the data past a certain date
        else:
            # else, just use the email and table name from user input
            cursor.execute(query, (email, table_name))

        # fetch the results
        results = cursor.fetchall()

        # close the database connection
        cursor.close()
        conn.close()

        return render_template('return_data_table.html', results=results)
        # then return the data in a table format where results are those collected
    else:
        # else, if not submitted, just render the input form to return data
        return render_template('return_data.html')


@app.route('/delete_entries_database', methods=['POST'])  # used to delete entries from the return_data_table.html file
def delete_entries_database():
    # users can use this to delete any entries in the database
    # Retrieve the form data
    # find the entries where the delete_row variable is seleted using the checkbox in html
    checked_entries = request.form.getlist('delete_row')

    # Connect to the database
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()

    # Delete the checked entries from the database based on the DOI
    for doi in checked_entries:
        cursor.execute("DELETE FROM papers WHERE doi = ?", (doi,))
        conn.commit()  # commit changes

    # Close the database connection
    cursor.close()
    conn.close()

    # Redirect back to the return_data route
    # tried routing back to the table, but saving a session returned the same table, so user will have to look up again
    return redirect(url_for('return_data'))


@app.route('/main')
# when the "DOI Summarizer and Recommender" in the top left will return to the starting page so users can nav back
def main():
    return render_template('starting_page.html')  # return the original page


###### Functions ######
# below are the functions used in the flask application
def extract_doi(url):
    """
    This removes the doi.org prefix if it exists in the list of DOIs from the user
    :param url: each string  in the doi list
    :return: the cleaned string
    """
    prefix = "https://doi.org/"
    if url.startswith(prefix):
        return url[len(prefix):]
    else:
        return url


def abstract_analysis(doi_list, email, table_name, api_key):
    """
    This is the main function of the script. It will find the PMIDs associated with the DOIs, extract the title and
    abstracts. Generate the two sets of summaries and the keywords. It will return it in a list of dictionaries called
    cleaned_abstracts and this will be written to the table seen in the output.html file.

    :param doi_list: list of comma-separated values of dois.
    :param email: the user's email address, str.
    :param table_name: the name the user wants for this search. can be used more than once, str.
    :param api_key: the API key from the if/else statement, str.
    :return: list of dictionaries called cleaned_abstracts.
    """
    Entrez.email = email  # email for Entrez
    Entrez.api_key = api_key  # api key for Entrez

    # Given a list of DOIs, find the PMIDs and article titles
    # initialize empty lists and dictionaries
    id_list = []
    titles = {}
    abstracts = {}
    mesh_terms = {}
    # clean dois before iterating
    dois = [extract_doi(doi) for doi in doi_list if extract_doi(doi)]
    for doi in dois:
        # for each entry:
        handle = Entrez.esearch(db='pubmed', term=doi)  # search pubmed with doi
        record = Entrez.read(handle)
        handle.close()
        if len(record['IdList']) > 0:  # if the IdList has a recording, continue
            pmid = record['IdList'][0]
            handle = Entrez.efetch(db='pubmed', id=pmid, rettype='xml', retmode='text')  # return the xml
            xml = handle.read()
            handle.close()  # close the efetch handle
            root = ET.fromstring(xml)  # parse through the xml
            article = root.find('.//PubmedArticle/MedlineCitation/Article')  # find the title
            title = article.find('ArticleTitle').text  # extract title
            id_list.append(pmid)  # add the PMID to the list
            titles[pmid] = title  # add to dictionary
            abstract_elem = root.find('.//AbstractText')  # extract abstract
            abstract = abstract_elem.text.strip()  # remove any new lines, white spaces, etc.
            abstracts[pmid] = abstract  # add to dictionary
            mesh_heading_list = root.findall('.//PubmedArticle/MedlineCitation/MeshHeadingList/MeshHeading')
            # extract MeSH terms
            mesh_terms[pmid] = [mesh.find('DescriptorName').text for mesh in mesh_heading_list]
            # add MeSH to dictionary

    # parse the abstracts
    # initialize empty list to write the dictionaries to
    cleaned_abstracts = []
    for pmid, abstract in abstracts.items():
        # abstracts contains the PMID and abstract
        # make set of summaries based on the abstract using two methods
        # first up, Networkx
        sentences = sent_tokenize(abstract)  # get every sentence in the abstract
        count_vectorizer = CountVectorizer()  # apply count vectorize, or tally the number of words in each sentence
        X = count_vectorizer.fit_transform(sentences)  # apply to the sentences
        # create a graph of sentence similarity
        graph = nx.Graph()
        # construct the graph
        for i, sentence_i in enumerate(sentences):
            for j, sentence_j in enumerate(sentences):
                if i == j:
                    continue
                similarity = cosine_similarity(X[i], X[j])[0][0]  # calculate the cosine similarity between the nodes
                graph.add_edge(i, j, weight=similarity)  # make the edges the similarity calculation

        # compute PageRank scores for each sentence
        scores = nx.pagerank(graph)

        # get the indices of the top two sentences
        top_indices = np.array(sorted(scores, key=scores.get, reverse=True)[:2])

        # construct the summary by joining the top two sentences
        nx_summary = ' '.join([sentences[i] for i in sorted(top_indices)])

        # next summary
        # second set of summaries using NLTK. This uses the same sentences variable from above
        stop_words = set(stopwords.words('english') + list(punctuation))

        # calculate the frequency of each word in the abstract
        # generate the word frequencies with a count
        word_frequencies = {}
        for word in abstract.split():
            if word.lower() not in stop_words:
                if word not in word_frequencies:
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] += 1

        # calculate the score of each sentence based on the frequency of its words using word_frequencies
        sentence_scores = {}
        for sentence in sentences:
            for word in sentence.split():
                if word.lower() in word_frequencies:
                    if len(sentence.split()) < 30:
                        if sentence not in sentence_scores:
                            sentence_scores[sentence] = word_frequencies[word]
                        else:
                            sentence_scores[sentence] += word_frequencies[word]

        # sort the sentences by their score and return the top 2
        summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:2]
        # join the top 2 sentences to create the summary
        nltk_summary = ' '.join(summary_sentences)
        # Combine the summaries
        total_summary = (f"Summary 1: {nx_summary} Summary 2: {nltk_summary}")
        # maybe insert new line in the output tbl
        # add dictionary to final list
        cleaned_abstracts.append(
            {'doi': doi_list[id_list.index(pmid)], 'pmid': pmid, 'title': titles[pmid], 'abstract': abstract,
             'summary': total_summary, 'mesh_terms': mesh_terms[pmid]})

    # keyword generating
    # now from the abstracts, return the keywords using scikit-learn
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')  # create a TF-IDF vectorizer
    corpus = [a['abstract'] for a in cleaned_abstracts]  # extract each abstract in the list of dictionaries
    # iterate through corpus
    for i, abstract in enumerate(corpus):
        tfidf_x = tfidf_vectorizer.fit_transform([abstract])  # apply the vectorizer to the abstract
        feature_names = tfidf_vectorizer.get_feature_names_out()  # return feature names
        idf_scores = tfidf_vectorizer.idf_  # calculate the IDF scores
        keyword_scores = defaultdict(float)  # dictionary for the scores as a float
        for j in range(tfidf_x.shape[1]):
            keyword_scores[feature_names[j]] += tfidf_x[0, j] * idf_scores[j]  # save the scores
        top_keywords = sorted(keyword_scores, key=keyword_scores.get, reverse=True)[:5]  # top 5
        # to return with the scores associated:
        # top_keywords = sorted(keyword_scores.items(), key=itemgetter(1), reverse=True)[:5]
        cleaned_abstracts[i]['scikit_keywords'] = top_keywords  # return the scikit-learn keywords

        # now create keywords that are pairs using NLTK since the scikit-learn keywords are generally one word
        # initialize the BigramAssocMeasure, which measures the relationship between pairs of words
        bigram_measures = BigramAssocMeasures()
        words = word_tokenize(abstract.lower())  # starting with the most common keywords
        stop_words = (stopwords.words("english"))
        # add some custom stop words that are common in research papers
        custom_stop_words = ['instead', 'study', 'results', 'analysis', 'method', 'data', 'experiment', 'figure',
                             'table', 'author', 'et al.', 'conclusion', 'discussion', 'findings', 'significant',
                             'difference', 'effect', 'increase', 'decrease', 'reduction', 'however', 'moreover',
                             'thus', 'therefore', 'also', 'similarly', 'hence', 'namely', 'cm', 'mm', 'mL', 'kg',
                             'accounts', 'approximation', 'across', 'research', 'approach', 'approaches']
        # extend the keywords by adding the custom stopwords from above
        stop_words.extend(custom_stop_words)
        # filter through the words in the abstract and only return those that are not stopwords
        filtered_tokens = [token for token in words if token not in stop_words]
        # apply the BigramCollocationFinder to find similar word pairs in the filtered words/tokens
        finder = BigramCollocationFinder.from_words(filtered_tokens)
        # get the measurements and return the best 5 pairs of words
        keywords = finder.nbest(bigram_measures.pmi, 5)
        # combine the keywords together in a list as strings
        keywords = [' '.join(keyword) for keyword in keywords]
        # return the NLTK keywords in the cleaned abstracts dictionary as a key, value pair
        cleaned_abstracts[i]['nltk_keywords'] = keywords

    # add the date time to the dictionary that will be written to the database
    time = datetime.now()
    for entry in cleaned_abstracts:
        entry['time'] = time.strftime("%Y-%m-%d %H:%M:%S")

        # insert into database
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        # add the list of dictionaries to the database as individual rows
        c.execute('''INSERT INTO papers (email, table_name, doi, pmid, title, abstract, summary, mesh_terms, 
        scikit_keywords, nltk_keywords, time)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (email, table_name, entry['doi'], entry['pmid'], entry['title'], entry['abstract'], entry['summary'],
                   ', '.join(entry['mesh_terms']), ', '.join(entry['scikit_keywords']),
                   ', '.join(entry['nltk_keywords']), entry['time']))
        conn.commit()
        conn.close()

    return cleaned_abstracts


# end abstract_analysis

def recommend_similar_articles(articles, email, api_key):
    """
    content-based recommender for each research paper.
    :param articles: input dictionary to the code. Should have the appropriate key, value pairs from abstract_analysis
    :param email: users email to find a NCBI account, str.
    :param api_key: users api key for NCBI from the if/else statement, str.
    :return: returns the dictionary with three new key, value pairs for recommendations based on keywords.
    """
    recommended_articles = []
    # abstracts = [article['abstract'] for article in articles]

    for i, article in enumerate(articles):
        # iterate through the articles in the dictionary
        try:
            # create a query from the list of keywords and mesh terms
            # get the scikit and NLTK keywords in each dictionary as well as the MeSH terms
            scikit_keywords = article.get("scikit_keywords", [])
            nltk_keywords = article.get('nltk_keywords', [])
            mesh_terms = article.get("mesh_terms", [])
            # combine the keywords together to generate a query
            sk_keyword_query = " ".join(scikit_keywords)
            nltk_keyword_query = " ".join(nltk_keywords)
            mesh_query = " ".join(mesh_terms)

            # collect the data for keywords
            Entrez.email = email  # email for Entrez
            Entrez.api_key = api_key  # api key for Entrez

            # scikit keywords
            # search entrez for the scikit keywords as the criteria, return 5 entries
            keyword_handle = Entrez.esearch(db="pubmed", term=sk_keyword_query, retmax=5)
            keyword_record = Entrez.read(keyword_handle)  # create a record
            keyword_ids = keyword_record["IdList"]  # get the ID list
            keyword_handle = Entrez.efetch(db="pubmed", id=keyword_ids, retmode="xml")  # fetch the XML for the PMID
            keyword_records = Entrez.read(keyword_handle)
            # return the title from the XML
            keyword_titles = [record["MedlineCitation"]["Article"]["ArticleTitle"] for record in
                              keyword_records["PubmedArticle"]]
            # NLTK keywords
            # repeat the same process as above but with the NLTK keywords as input (keyword pairs)
            nltk_keyword_handle = Entrez.esearch(db="pubmed", term=nltk_keyword_query, retmax=5)
            nltk_keyword_record = Entrez.read(nltk_keyword_handle)
            nltk_keyword_ids = nltk_keyword_record["IdList"]
            nltk_keyword_handle = Entrez.efetch(db="pubmed", id=nltk_keyword_ids, retmode="xml")
            nltk_keyword_records = Entrez.read(nltk_keyword_handle)
            nltk_keyword_titles = [record["MedlineCitation"]["Article"]["ArticleTitle"] for record in
                                   nltk_keyword_records["PubmedArticle"]]

            # Step 3: Collect the data for mesh terms
            # repeat the same process from scikit learn but with the MeSH terms as input
            mesh_handle = Entrez.esearch(db="pubmed", term=mesh_query, retmax=5)
            mesh_record = Entrez.read(mesh_handle)
            mesh_ids = mesh_record["IdList"]
            mesh_handle = Entrez.efetch(db="pubmed", id=mesh_ids, retmode="xml")
            mesh_records = Entrez.read(mesh_handle)
            mesh_titles = [record["MedlineCitation"]["Article"]["ArticleTitle"] for record in
                           mesh_records["PubmedArticle"]]

            # add the recommended articles to the dictionary
            article["scikit_keywords_recs"] = keyword_titles
            article['nltk_keywords_recs'] = nltk_keyword_titles
            article["mesh_recs"] = mesh_titles

            recommended_articles.append(article)

        except Exception as e:
            # raise an exception if there is nothing for the recommendations and just return an empty list
            article["scikit_keywords_recs"] = []
            article["nltk_keywords_recs"] = []
            article["mesh_recs"] = []

            recommended_articles.append(article)  # append the results to the final list

    return recommended_articles


# end recommended_articles

if __name__ == '__main__':
    # main function, deploy the above application
    app.run(debug=True)
