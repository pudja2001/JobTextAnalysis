from turtle import color
import warnings
warnings.simplefilter(action='ignore')
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from collections import  Counter
import matplotlib.pyplot as plt
plt.style.use('ggplot')
stop=set(stopwords.words('english'))
import re
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import gensim
import string
import pandas as pd
import numpy as np
import seaborn as sns

# Convert PostDate Format into Datetime
def convertpostDate(df):
    TempDf = df.copy()
    TempDf["PostDate"] = TempDf["PostDate"].apply(lambda x: re.sub('[A-Za-z]', '', x))
    TempDf.loc[TempDf["PostDate"].str.contains("30+"), "PostDate"] = "30"
    TempDf.loc[TempDf["PostDate"]==" ", "PostDate"] = "0"
    TempDf["PostDate"] = TempDf["PostDate"].apply(lambda x: re.sub(' ', '', x))
    TempDf["PostDate"] = TempDf["PostDate"].astype(int)
    TempDf['PostDate'] = pd.to_datetime(TempDf['ScrapeDate'], format='%d-%m-%Y') -  pd.to_timedelta(TempDf['PostDate'], unit='d')
    return TempDf

def clean_text(text):
    text = re.sub('[.?\",()|+*#&]+', '', text)
    text = re.sub('[:/]+', ' ', text)
    text = re.sub('job description', '', text)
    text = re.sub('job description ', '', text)
    text = re.sub('job description: ', '', text)
    text = re.sub('location ', '', text)
    text = re.sub('work type ', '', text)
    text = re.sub(r"it'll", "it will", text)
    text = re.sub(r"we'll", "we will", text)
    text = re.sub(r"wouldn\x89Ûªt", "would not", text)
    text = re.sub(r"We've", "We have", text)
    text = re.sub(r"he'll", "he will", text)
    text = re.sub(r"Y'all", "You all", text)
    text = re.sub(r"Weren't", "Were not", text)
    text = re.sub(r"Didn't", "Did not", text)
    text = re.sub(r"they'll", "they will", text)
    text = re.sub(r"they'd", "they would", text)
    text = re.sub(r"DON'T", "DO NOT", text)
    text = re.sub(r"That\x89Ûªs", "That is", text)
    text = re.sub(r"they've", "they have", text)
    text = re.sub(r"i'd", "I would", text)
    text = re.sub(r"should've", "should have", text)
    text = re.sub(r"Let's", "Let us", text)
    text = re.sub(r"you'd", "You would", text)
    text = re.sub(r"It's", "It is", text)
    text = re.sub(r"Ain't", "am not", text)
    text = re.sub(r"Haven't", "Have not", text)
    text = re.sub(r"Could've", "Could have", text)
    text = re.sub(r"youve", "you have", text)  
    text = re.sub(r"donå«t", "do not", text)
    
    return text

def gen_wordcloud(df, position='All'):
    dfNew = df
    if position == 'All':
        words_per_jobdesc = dfNew["JobDesc"]
    else:
        words_per_jobdesc = dfNew[dfNew["JobTitle"].str.contains(position)]["JobDesc"]
    all_jobdesc = []

    for row in words_per_jobdesc:
        row = np.array(row)
        mask = row != ''
        row = row[mask]
        all_jobdesc.append(row)
        
    jobdesc = [" ".join(text) for text in all_jobdesc]
    final_jobdesc = " ".join(jobdesc)
    
    wordcloud_jobdesc = WordCloud(background_color="black").generate(final_jobdesc)

    plt.figure(figsize = (20,20))
    plt.imshow(wordcloud_jobdesc, interpolation='None')
    plt.axis("off")
    plt.show()

def generate_ngrams(text, n_gram=1):
    token = [token for token in text.lower().split(' ') if token != '' if token not in STOPWORDS]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [' '.join(ngram) for ngram in ngrams]

def generate_ngrams_bar(df, position='All', n_gram=1):
    dfNew = df
    if position == 'All':
        dfNewMasked = dfNew.copy()
    else:
        dfNewMasked = dfNew[df["JobTitle"].str.contains(position)]
    # Unigrams
    if n_gram == 1:
        n = 'uni'
        jobdesc_unigrams = defaultdict(int)

        for text in dfNewMasked['JobDesc']:
            for word in generate_ngrams(text):
                jobdesc_unigrams[word] += 1

        dfNew_jobdesc_ngrams = pd.DataFrame(sorted(jobdesc_unigrams.items(), key=lambda x: x[1])[::-1])
    # Bigrams
    elif n_gram == 2:
        n = 'bi'
        jobdesc_bigrams = defaultdict(int)

        for text in dfNewMasked['JobDesc']:
            for word in generate_ngrams(text, n_gram=2):
                jobdesc_bigrams[word] += 1

        dfNew_jobdesc_ngrams = pd.DataFrame(sorted(jobdesc_bigrams.items(), key=lambda x: x[1])[::-1])
    # Trigrams
    elif n_gram == 3:
        n = 'tri'
        jobdesc_trigrams = defaultdict(int)

        for text in dfNewMasked['JobDesc']:
            for word in generate_ngrams(text, n_gram=3):
                jobdesc_trigrams[word] += 1

        dfNew_jobdesc_ngrams = pd.DataFrame(sorted(jobdesc_trigrams.items(), key=lambda x: x[1])[::-1])
        
    dfNew_jobdesc_ngrams = dfNew_jobdesc_ngrams.iloc[:50]    
    dfNew_jobdesc_ngrams.columns = ['ngrams', 'counts']
    
    return dfNew_jobdesc_ngrams

def JobsOpenedOverTime(df, position="All"):
    dfNew = df
    fig = plt.figure(figsize=(10,7))
    fig.patch.set_facecolor('black')
    fig.patch.set_alpha(0.6)

    ax = fig.add_subplot(111)
    ax.patch.set_facecolor('green')
    ax.patch.set_alpha(1.0)

    if position == 'All':
        job_per_date = pd.DataFrame(dfNew["PostDate"].value_counts(ascending=False)).reset_index()
        job_per_date.columns = ["Date", "NumOfPositions"]
        job_per_date = job_per_date.sort_values(by="Date")

    else:
        maskedDf = dfNew.copy()
        maskedDf = maskedDf.loc[maskedDf["JobTitle"].str.contains(position),:]
        job_per_date = pd.DataFrame(maskedDf["PostDate"].value_counts(ascending=False)).reset_index()
        job_per_date.columns = ["Date", "NumOfPositions"]
        job_per_date = job_per_date.sort_values(by="Date")

    return job_per_date    
    # plt.plot(job_per_date["Date"], job_per_date["NumOfPositions"], color='green')
    # plt.xticks(rotation=45, color="green")
    # plt.xlabel("Date", color="green")
    # plt.yticks(color="green")
    # plt.ylabel("Number of Positions Opened", color="green")
    # plt.show()

def NumofAllPosition(df, company='All', limitNum=30):

    dfNew = df
    plt.figure(figsize=(7,8))
    
    if company == 'All':
        sumOfEachPos = dfNew["JobTitle"].value_counts(ascending=False)[:limitNum]
        
    else:
        maskedDf = dfNew.copy()
        sumOfEachPos = maskedDf.loc[maskedDf["Company"]==company, "JobTitle"].value_counts(ascending=False)[:limitNum]

    sumOfEachPositions = pd.DataFrame(sumOfEachPos).reset_index()
    sumOfEachPositions.columns = ["Position", "Num"]  

    return sumOfEachPositions

def numOfpositionEachCompany(df, position="All", limitNum=30):
    """Generate Bar Plot of Distributions of Num of Specific Position in each Company"""
    
    dfNew = df
    plt.figure(figsize=(7,8))
    if position == "All":  
        sum_of_each_company = dfNew["Company"].value_counts(ascending=False)[:limitNum]

    else:
        Tempdf = dfNew.copy()
        Tempdf["mask"] = Tempdf["JobTitle"].str.contains(position)
        masked_df = Tempdf.loc[Tempdf["mask"] == True]
    
        sum_of_each_company = masked_df["Company"].value_counts(ascending=False)[:limitNum]
        
    sum_of_each_company = pd.DataFrame(sum_of_each_company).reset_index()
    sum_of_each_company.columns = ["Company", "Num"]    

    return sum_of_each_company
    # return sum_of_each_company
    # sns.barplot(x=sum_of_each_company.values, y=sum_of_each_company.index, color='green')
    # plt.xlabel("Sum of Position Opened")
    # plt.ylabel("Company")
    # plt.title("Number of {} Positions Opened in each Company".format(position))
    # plt.show()

def data_cleaning(df):
    df = df.drop(columns=["Unnamed: 0"])
    dfNew = df.drop_duplicates()
    
    dfNew["JobSummary"] = dfNew["JobSummary"].str.lower()
    dfNew["JobSummary"] = dfNew["JobSummary"].apply(lambda x: re.sub('\n', '', x))
    dfNew["JobDesc"] = dfNew["JobDesc"].str.lower()
    dfNew["JobDesc"] = dfNew["JobDesc"].apply(lambda x: re.sub('\n', "", x))
    
    dfNew["JobSummary"] = dfNew["JobSummary"].str.lower()
    dfNew["JobSummary"] = dfNew["JobSummary"].apply(lambda x: re.sub('\n', '', x))
    dfNew["JobDesc"] = dfNew["JobDesc"].str.lower()
    dfNew["JobDesc"] = dfNew["JobDesc"].apply(lambda x: re.sub('\n', "", x))
    
    dfNew["UncleanedJobDesc"] = dfNew["JobDesc"]
    dfNew["UncleanedJobSummary"] = dfNew["JobSummary"]
    
    dfNew["JobDesc"] = dfNew["UncleanedJobDesc"].apply(lambda x: clean_text(x))
    dfNew["JobSummary"] = dfNew["UncleanedJobSummary"].apply(lambda x: clean_text(x))

    dfNew = convertpostDate(dfNew)
    
    return dfNew


    


