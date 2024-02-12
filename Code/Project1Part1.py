#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import praw
import time
import csv
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import re

#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('omw-1.4')

file_path = 'newsapi_key.txt' 

with open(file_path, 'r') as file:
    api_key = file.readline().strip()


# In[2]:


#Getting data via NewsAPI 
url = "https://newsapi.org/v2/everything"
articles = []

for page in range(1, 4):  
    params = {
        "q": "blockchain",
        "apiKey": api_key,
        "page": page  
    }

    response = requests.get(url, params=params)
    data = response.json()

    if response.status_code == 200:
        articles.extend(data['articles'])

print(f"Total articles fetched: {len(articles)}")


# In[3]:


news_df = pd.DataFrame(articles)
news_df['title'] = news_df['title'].fillna('')
news_df['description'] = news_df['description'].fillna('')

news_df.head()


# In[4]:


# Function to lemmatize text
def clean_text_lem(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert text to lowercase
    text = text.lower()
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    
    LEMMER = WordNetLemmatizer()
    lemmatized_tokens = [LEMMER.lemmatize(token) for token in filtered_tokens]
    
    # Reconstruct the text
    clean_text_lem = ' '.join(lemmatized_tokens)
    return clean_text_lem

# Apply the cleaning function to the DataFrame
news_df['title_lem'] = news_df['title'].apply(clean_text_lem)
news_df['description_lem'] = news_df['description'].apply(clean_text_lem)
news_df_lemmatized = news_df[['title_lem', 'description_lem']]


# In[5]:


news_df_lemmatized.head()


# In[6]:


# Function to stem text
def clean_text_stem(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert text to lowercase
    text = text.lower()
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    
    STEMMER = PorterStemmer()  
    stemmed_tokens = [STEMMER.stem(token) for token in filtered_tokens]  
    
    # Reconstruct the text
    clean_text_stem = ' '.join(stemmed_tokens)  
    return clean_text_stem

# Apply the cleaning function to the DataFrame
news_df['title_stem'] = news_df['title'].apply(clean_text_stem)
news_df['description_stem'] = news_df['description'].apply(clean_text_stem)
news_df_stemmed = news_df[['title_stem', 'description_stem']]


# In[7]:


news_df_stemmed.head()


# In[8]:


# Function to generate and display a word cloud
def generate_word_cloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color ='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()


# In[9]:


# Generating word clouds for uncleaned data
title_text = ' '.join(news_df['title'])
description_text = ' '.join(news_df['description'])
generate_word_cloud(title_text, 'Titles Word Cloud')
generate_word_cloud(description_text, 'Descriptions Word Cloud')


# In[10]:


# Generating word clouds for lemmatized data
lem_title_text = ' '.join(news_df_lemmatized['title_lem'])
lem_description_text = ' '.join(news_df_lemmatized['description_lem'])
generate_word_cloud(lem_title_text, 'Lemmatized Titles Word Cloud')
generate_word_cloud(lem_description_text, 'Lemmatized Descriptions Word Cloud')


# In[11]:


# Generating word clouds for stemmed data
stem_title_text = ' '.join(news_df_stemmed['title_stem'])
stem_description_text = ' '.join(news_df_stemmed['description_stem'])
generate_word_cloud(stem_title_text, 'Stemmed Titles Word Cloud')
generate_word_cloud(stem_description_text, 'Stemmed Descriptions Word Cloud')


# In[12]:


# Function to apply CountVectorizer
def apply_count_vectorizer(df, column_name, max_features=None, max_df=0.85, min_df=3):
    vectorizer = CountVectorizer(max_df=max_df, min_df=min_df, max_features=max_features)
    X = vectorizer.fit_transform(df[column_name])
    return pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

# Function to apply TfidfVectorizer
def apply_tfidf_vectorizer(df, column_name, max_features=None, max_df=0.85, min_df=3):
    vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df, max_features=max_features)
    X = vectorizer.fit_transform(df[column_name])
    return pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

# Usage with the lemmatized titles
title_countvec_lem = apply_count_vectorizer(news_df_lemmatized, 'title_lem', max_features=85)
title_tfidf_lem = apply_tfidf_vectorizer(news_df_lemmatized, 'title_lem', max_features=85)
# Usage with the stemmed titles
title_countvec_stem = apply_count_vectorizer(news_df_stemmed, 'title_stem', max_features=85)
title_tfidf_stem = apply_tfidf_vectorizer(news_df_stemmed, 'title_stem', max_features=85)

# Usage with the lemmatized descriptions
description_countvec_lem = apply_count_vectorizer(news_df_lemmatized, 'description_lem', max_features=110)
description_tfidf_lem = apply_tfidf_vectorizer(news_df_lemmatized, 'description_lem', max_features=110)
# Usage with the stemmed descriptions
description_countvec_stem = apply_count_vectorizer(news_df_stemmed, 'description_stem', max_features=110)
description_tfidf_stem = apply_tfidf_vectorizer(news_df_stemmed, 'description_stem', max_features=110)


# In[13]:


#for i in description_countvec_lem:
    #print(i)


# In[14]:


#for i in title_countvec_lem:
    #print(i)


# In[15]:


title_tfidf_lem.head()


# In[16]:


description_tfidf_stem.head()


# In[17]:


title_countvec_stem.head()


# In[18]:


# Generating word clouds 


# In[19]:


# Saving dataframes to CSV files
title_countvec_lem.to_csv('title_countvec_lem.csv', index=False)
title_tfidf_lem.to_csv('title_tfidf_lem.csv', index=False)
title_countvec_stem.to_csv('title_countvec_stem.csv', index=False)
title_tfidf_stem.to_csv('title_tfidf_stem.csv', index=False)
description_countvec_lem.to_csv('description_countvec_lem.csv', index=False)
description_tfidf_lem.to_csv('description_tfidf_lem.csv', index=False)
description_countvec_stem.to_csv('description_countvec_stem.csv', index=False)
description_tfidf_stem.to_csv('description_tfidf_stem.csv', index=False)


# In[ ]:





# In[20]:


#Getting Reddit data via API wrapper
def read_file(file_path):
    file = {}
    with open(file_path, 'r') as i:
        for line in i:
            key, value = line.strip().split('=', 1)
            file[key] = value
    return file

file = read_file('reddit_key.txt')

reddit = praw.Reddit(client_id=file['client_id'],
                     client_secret=file['client_secret'],
                     user_agent=file['user_agent'])

def scrape_reddit(term, filename='reddit_data.csv'):
    posts_data = []
    # Search posts on Reddit
    for submission in reddit.subreddit("all").search(term, limit=200):
        submission.comments.replace_more(limit=0)
        comments = [comment.body for comment in submission.comments.list()]
        posts_data.append({
            'title': submission.title,
            'selftext': submission.selftext,
            'comments': ' | '.join(comments)  # Join comments with a separator
        })
    
    # Save to CSV
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['title', 'selftext', 'comments'])
        writer.writeheader()
        for post in posts_data:
            writer.writerow(post)


# In[21]:


#term = 'blockchain'
#posts_data = scrape_reddit(term)


# In[22]:


reddit_df = pd.read_csv('reddit_data.csv')
reddit_df


# In[23]:


reddit_df['title'] = reddit_df['title'].fillna('')
reddit_df['selftext'] = reddit_df['selftext'].fillna('')
reddit_df['comments'] = reddit_df['comments'].fillna('')
reddit_df.head()


# In[24]:


# Reddit Lemmitization

# Apply the cleaning function to the DataFrame
reddit_df['title_lem'] = reddit_df['title'].apply(clean_text_lem)
reddit_df['selftext_lem'] = reddit_df['selftext'].apply(clean_text_lem)
reddit_df['comments_lem'] = reddit_df['comments'].apply(clean_text_lem)
reddit_df_lemmatized = reddit_df[['title_lem', 'selftext_lem', 'comments_lem']]

reddit_df_lemmatized.head()


# In[25]:


# Reddit Stemming

# Apply the cleaning function to the DataFrame
reddit_df['title_stem'] = reddit_df['title'].apply(clean_text_stem)
reddit_df['selftext_stem'] = reddit_df['selftext'].apply(clean_text_stem)
reddit_df['comments_stem'] = reddit_df['comments'].apply(clean_text_stem)
reddit_df_stemmed = reddit_df[['title_stem', 'selftext_stem', 'comments_stem']]

reddit_df_stemmed.head()


# In[26]:


# Function to apply CountVectorizer
def apply_count_vectorizer(df, column_name, max_features=None, max_df=0.85, min_df=3):
    vectorizer = CountVectorizer(max_df=max_df, min_df=min_df, max_features=max_features)
    X = vectorizer.fit_transform(df[column_name])
    return pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

# Function to apply TfidfVectorizer
def apply_tfidf_vectorizer(df, column_name, max_features=None, max_df=0.85, min_df=3):
    vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df, max_features=max_features)
    X = vectorizer.fit_transform(df[column_name])
    return pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())


# Vectorization with the lemmatized comments
reddit_countvec_lem = apply_count_vectorizer(reddit_df_lemmatized, 'comments_lem', max_features=200)
reddit_tfidf_lem = apply_tfidf_vectorizer(reddit_df_lemmatized, 'comments_lem', max_features=200)
# Vectorization with the stemmed comments
reddit_countvec_stem = apply_count_vectorizer(reddit_df_stemmed, 'comments_stem', max_features=200)
reddit_tfidf_stem = apply_tfidf_vectorizer(reddit_df_stemmed, 'comments_stem', max_features=200)

# Vectorization with the lemmatized title
reddit_countvec_lem_title = apply_count_vectorizer(reddit_df_lemmatized, 'title_lem', max_features=75)
reddit_tfidf_lem_title = apply_tfidf_vectorizer(reddit_df_lemmatized, 'title_lem', max_features=75)
# Vectorization with the stemmed title
reddit_countvec_stem_title = apply_count_vectorizer(reddit_df_stemmed, 'title_stem', max_features=75)
reddit_tfidf_stem_title = apply_tfidf_vectorizer(reddit_df_stemmed, 'title_stem', max_features=75)

# Vectorization with the lemmatized selftext
reddit_countvec_lem_selftext = apply_count_vectorizer(reddit_df_lemmatized, 'selftext_lem', max_features=100)
reddit_tfidf_lem_selftext = apply_tfidf_vectorizer(reddit_df_lemmatized, 'selftext_lem', max_features=100)
# Vectorization with the stemmed selftext
reddit_countvec_stem_selftext = apply_count_vectorizer(reddit_df_stemmed, 'selftext_stem', max_features=100)
reddit_tfidf_stem_selftext = apply_tfidf_vectorizer(reddit_df_stemmed, 'selftext_stem', max_features=100)


# In[27]:


reddit_countvec_stem.head()


# In[28]:


#for i in reddit_countvec_stem:
    #print(i)


# In[29]:


reddit_countvec_lem.head()


# In[30]:


# Saving the vectorized data
reddit_countvec_lem.to_csv('reddit_countvec_lem.csv', index=False)
reddit_tfidf_lem.to_csv('reddit_tfidf_lem.csv', index=False)
reddit_countvec_stem.to_csv('reddit_countvec_stem.csv', index=False)
reddit_tfidf_stem.to_csv('reddit_tfidf_stem.csv', index=False)
reddit_countvec_lem_title.to_csv('reddit_countvec_lem_title.csv', index=False)
reddit_tfidf_lem_title.to_csv('reddit_tfidf_lem_title.csv', index=False)
reddit_countvec_stem_title.to_csv('reddit_countvec_stem_title.csv', index=False)
reddit_tfidf_stem_title.to_csv('reddit_tfidf_stem_title.csv', index=False)
reddit_countvec_lem_selftext.to_csv('reddit_countvec_lem_selftext.csv', index=False)
reddit_tfidf_lem_selftext.to_csv('reddit_tfidf_lem_selftext.csv', index=False)
reddit_countvec_stem_selftext.to_csv('reddit_countvec_stem_selftext.csv', index=False)
reddit_tfidf_stem_selftext.to_csv('reddit_tfidf_stem_selftext.csv', index=False)


# In[ ]:





# In[ ]:





# In[31]:


# Using Selenium to scrape Medium articles' titles and descriptions
#url = 'https://medium.com/tag/blockchain/recommended'
#options = webdriver.ChromeOptions()
#driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
#driver.get(url)

#SCROLL_PAUSE_TIME = 5 

#last_height = driver.execute_script("return document.body.scrollHeight")
#articles_with_tags = []

#try:
    #while True:
        # Scroll down to bottom
        #driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # Wait to load page
        #time.sleep(SCROLL_PAUSE_TIME)

        # Find new set of articles
        #articles = driver.find_elements(By.CSS_SELECTOR, 'h2, h3')
        #for article in articles:
            #title = article.text.strip()
            #tag = article.tag_name
            #if title and (title, tag) not in articles_with_tags:
                #articles_with_tags.append((title, tag))

        # Break after number of articles
        #if len(articles_with_tags) >= 500:
            #break 
#finally:
    #driver.quit()

#print(f"Collected {len(articles_with_tags)} articles")


# In[32]:


#articles_with_tags

#(title, h2), (description, h3), .... format

articles_with_tags = pd.read_csv('medium_articles.csv')
articles_with_tags.head()


# In[33]:


titles = []
descriptions = []

current_title = ""
current_description = ""

for _, row in articles_with_tags.iterrows():
    title, tag = row['Title'], row['Tag']
    if tag == "h2":
        if current_title:  # If there's already a title save it and its description
            titles.append(current_title)
            descriptions.append(current_description)
            current_description = ""  # Reset description for the next title
        current_title = title
    elif tag == "h3" and current_title:
        if current_description:
            current_description += " " + title  # Separate concatenatation with a space
        else:
            current_description = title

# Append the last title and description if any
if current_title:
    titles.append(current_title)
    descriptions.append(current_description)

# Create DataFrame
medium_df = pd.DataFrame({
    "Title": titles,
    "Description": descriptions
})


# In[34]:


medium_df['Title'] = medium_df['Title'].fillna('')
medium_df['Description'] = medium_df['Description'].fillna('')
medium_df


# In[35]:


# Medium Lemmitization

# Apply the cleaning function to the DataFrame
medium_df['title_lem'] = medium_df['Title'].apply(clean_text_lem)
medium_df['description_lem'] = medium_df['Description'].apply(clean_text_lem)
medium_df_lemmatized = medium_df[['title_lem', 'description_lem']]

medium_df_lemmatized.head()


# In[36]:


# Medium Stemming

# Apply the cleaning function to the DataFrame
medium_df['title_stem'] = medium_df['Title'].apply(clean_text_stem)
medium_df['description_stem'] = medium_df['Description'].apply(clean_text_stem)
medium_df_stemmed = medium_df[['title_stem', 'description_stem']]

medium_df_stemmed.head()


# In[37]:


# Function to apply CountVectorizer
def apply_count_vectorizer(df, column_name, max_features=None, max_df=0.85, min_df=2):
    vectorizer = CountVectorizer(max_df=max_df, min_df=min_df, max_features=max_features)
    X = vectorizer.fit_transform(df[column_name])
    return pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

# Function to apply TfidfVectorizer
def apply_tfidf_vectorizer(df, column_name, max_features=None, max_df=0.85, min_df=2):
    vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df, max_features=max_features)
    X = vectorizer.fit_transform(df[column_name])
    return pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())


# Vectorization with the lemmatized Title
medium_countvec_lem_title = apply_count_vectorizer(medium_df_lemmatized, 'title_lem', max_features=60)
medium_tfidf_lem_title = apply_tfidf_vectorizer(medium_df_lemmatized, 'title_lem', max_features=60)
# Vectorization with the stemmed Title
medium_countvec_stem_title = apply_count_vectorizer(medium_df_stemmed, 'title_stem', max_features=60)
medium_tfidf_stem_title = apply_tfidf_vectorizer(medium_df_stemmed, 'title_stem', max_features=60)

# Vectorization with the lemmatized description
medium_countvec_lem_description = apply_count_vectorizer(medium_df_lemmatized, 'description_lem', max_features=85)
medium_tfidf_lem_description = apply_tfidf_vectorizer(medium_df_lemmatized, 'description_lem', max_features=85)
# Vectorization with the stemmed description
medium_countvec_stem_description = apply_count_vectorizer(medium_df_stemmed, 'description_stem', max_features=85)
medium_tfidf_stem_description = apply_tfidf_vectorizer(medium_df_stemmed, 'description_stem', max_features=85)


# In[38]:


medium_tfidf_lem_description.head()


# In[39]:


#for i in medium_tfidf_lem_description:
    #print(i)


# In[40]:


medium_tfidf_lem_title.head()


# In[41]:


#for i in medium_tfidf_lem_title:
    #print(i)


# In[42]:


# Saving the vectorized data
medium_countvec_lem_title.to_csv('medium_countvec_lem_title.csv', index=False)
medium_tfidf_lem_title.to_csv('medium_tfidf_lem_title.csv', index=False)
medium_countvec_stem_title.to_csv('medium_countvec_stem_title.csv', index=False)
medium_tfidf_stem_title.to_csv('medium_tfidf_stem_title.csv', index=False)
medium_countvec_lem_description.to_csv('medium_countvec_lem_description.csv', index=False)
medium_tfidf_lem_description.to_csv('medium_tfidf_lem_description.csv', index=False)
medium_countvec_stem_description.to_csv('medium_countvec_stem_description.csv', index=False)
medium_tfidf_stem_description.to_csv('medium_tfidf_stem_description.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:




