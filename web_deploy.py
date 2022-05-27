import streamlit as st
import pandas as pd
import altair as alt
import cleaning_analysis

st.set_page_config(layout="wide")

# Define the container
header = st.container()
dataset = st.container()
eda = st.container()
features = st.container()

# Read the Data and cleaning the data
df = pd.read_csv("datajobs.csv")
df = cleaning_analysis.data_cleaning(df)

# Slidebar 
pos_list = ["Data Scientist", "Data Analyst", "Business Intelligence", "Business Analyst", "Data Engineer"]
pos_list.insert(0, "All")
list_company = list(df["Company"].unique())
list_company.insert(0, "All")

title_sidebar = st.sidebar.title("Positions and Company you wanna see")
position = st.sidebar.selectbox('Choose the Positions!', pos_list)
company = st.sidebar.selectbox('Choose the Company!', list_company)

with header:
    st.title("Text Analysis on Data Field Job in indeed.com")
    st.markdown("This website is designed to present some visualizations about Exploration Data Analysis (EDA) of Job Description text in data field position (Data Scientist/Analyst, Data Engineer, etc) and does topic modelling on overall job description text data (on going) to know what's the most and important topic in the job description of data field position.")
    st.markdown("***Made by M Pudja Gemilang***")


with dataset:
    st.header("Data ")
    st.markdown("Data was scraped from job platform indeed.com using BeautifoulSoup in Pyhton. Data has 89 null values. Here is the data after some cleaning :")
    st.write(df.head(5))
    

with eda:
    # Bar Plot of  NUmber of Total Specific Jobs on each company
    st.header("Exploratory Data Analysis (EDA)")
    st.markdown("**Number of {} Jobs on Each Company**".format(position))
    st.set_option('deprecation.showPyplotGlobalUse', False)
    NumOfPos = cleaning_analysis.numOfpositionEachCompany(df, position)
 
    fig1 = alt.Chart(NumOfPos).mark_bar().encode(x = alt.X('Num', title="Number of Position"), y = alt.Y('Company', title="Company", sort='-x')).configure_mark(opacity=0.8, color='blue')
    st.altair_chart(fig1, use_container_width = True)

    # Bar Plot of  NUmber of Total Specific Jobs on each company
    st.markdown("**Number of Each Position in {} Company**".format(company))
    st.set_option('deprecation.showPyplotGlobalUse', False)
    NumOfComp = cleaning_analysis.NumofAllPosition(df, company)

    fig2 = alt.Chart(NumOfComp).mark_bar().encode(x = alt.X('Num', title="Number of Company"), y = alt.Y('Position', title="Position", sort='-x')).configure_mark(opacity=0.8, color='red')
    st.altair_chart(fig2, use_container_width = True)

    # Number Of Positions Opened Over Time
    st.markdown("**Number of {} Positions Job Descriptions Over Time**".format(position))
    st.set_option('deprecation.showPyplotGlobalUse', False)
    PosOverTime = cleaning_analysis.JobsOpenedOverTime(df, position)

    fig3 = alt.Chart(PosOverTime).mark_line().encode(x = alt.X('Date', title="Date"), y = alt.Y('NumOfPositions', title="Number of Position", sort='-x')).configure_mark(opacity=0.8, color='purple')
    # fig3 = alt.Chart(PosOverTime).mark_line().encode(x = 'Date', y = 'NumOfPositions').configure_mark(opacity=0.8, color='purple')
    st.altair_chart(fig3, use_container_width = True)

    # N-Grams Count 
    # Unigram
    st.markdown("**Counts of Unigrams of {} Job Descriptions**".format(position))
    st.set_option('deprecation.showPyplotGlobalUse', False)
    unigrams = cleaning_analysis.generate_ngrams_bar(df, position, 1)
    fig4 = alt.Chart(unigrams).mark_bar().encode(x = alt.X('counts', title="Counts"), y = alt.Y('ngrams', title="Unigram", sort='-x')).configure_mark(opacity=0.8, color='red')
    st.altair_chart(fig4, use_container_width = True)

    # Bigram
    st.markdown("**Counts of Bigrams of {} Job Descriptions**".format(position))
    st.set_option('deprecation.showPyplotGlobalUse', False)
    bigrams = cleaning_analysis.generate_ngrams_bar(df, position, 2)
    fig5 = alt.Chart(bigrams).mark_bar().encode(x = alt.X('counts', title="Counts"), y = alt.Y('ngrams', title="Unigram", sort='-x')).configure_mark(opacity=0.8, color='green')
    st.altair_chart(fig5, use_container_width = True)

    # Trigram
    st.markdown("**Counts of Trigrams of {} Job Descriptions**".format(position))
    st.set_option('deprecation.showPyplotGlobalUse', False)
    trigrams = cleaning_analysis.generate_ngrams_bar(df, position, 3)
    fig6 = alt.Chart(trigrams).mark_bar().encode(x = alt.X('counts', title="Counts"), y = alt.Y('ngrams', title="Unigram", sort='-x')).configure_mark(opacity=0.8, color='blue')
    st.altair_chart(fig6, use_container_width = True)

    # Wordcloud
    st.markdown("**Wordcloud of {} Positions**".format(position))
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(cleaning_analysis.gen_wordcloud(df, position))



