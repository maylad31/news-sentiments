import pandas as pd
from vaderSentimentGER import SentimentIntensityAnalyzer
import tqdm
from sklearn import preprocessing
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import argparse
from nltk.corpus import stopwords
import cleantext



    
def getwords_content_pre(df: pd.DataFrame, filename: str, german_stop_words: list[str]) -> None:
    """
    wordcloud for pre covid using content
    :param df: dataframe
    :param filename: input filename used to generate output filename

    """
    
    
    df = df.copy()
    #drop post covid data
    for index, row in df.iterrows():
        if int(row["YEAR"])>2020:
            df.drop(index, inplace=True)
    df.dropna(inplace = True)
    
    
    #print(german_stop_words)
    
    
    #get text
    text_list=df["CONTENT"].tolist()
    text=" ".join(text_list)
    text=text.lower()
   
    text=cleantext.clean(text, extra_spaces=True, lowercase=True, numbers=True, punct=True)
    
    text_list=[i for i in text.split() if i not in german_stop_words]
    text=" ".join(text_list)
    

    #generate wordcloud
    word_cloud = WordCloud(collocations = False, background_color = 'white').generate(text)
    #save wordcloud
    word_cloud.to_file(filename+"_wordcloud_content_pre_covid"+".png")
   
    

def getwords_content_post(df: pd.DataFrame ,filename: str, german_stop_words: list[str]) -> None:
    """
    wordcloud for post covid using content
    :param df: dataframe
    :param filename: input filename used to generate output filename

    """
    
    df=df.copy()
    
    #drop pre covid data
    for index, row in df.iterrows():
        if int(row["YEAR"])<2021:
            df.drop(index, inplace=True)
    
    df.dropna(inplace = True)
    
    #get text
    text_list=df["CONTENT"].tolist()
    text=" ".join(text_list)
    text=text.lower()
    text=cleantext.clean(text, extra_spaces=True, lowercase=True, numbers=True, punct=True)
    text_list=[i for i in text.split() if i not in german_stop_words]
    text=" ".join(text_list)
    
    #generate wordcloud
    word_cloud = WordCloud(collocations = False, background_color = 'white').generate(text)
    #save wordcloud 
    word_cloud.to_file(filename+"_wordcloud_content_post_covid"+".png")

def getwords_headline_pre(df: pd.DataFrame, filename: str, german_stop_words: list[str]) -> None:
    """
    wordcloud for pre covid using headline
    :param df: dataframe
    :param filename: input filename used to generate output filename

    """
    
    df=df.copy()
    #drop post covid data
    for index, row in df.iterrows():
        if int(row["YEAR"])>2020:
            df.drop(index, inplace=True)
    df.dropna(inplace = True)
    #get text
    text_list=df["HEADLINE"].tolist()
    text=" ".join(text_list)
    text=text.lower()
    text=cleantext.clean(text, extra_spaces=True, lowercase=True, numbers=True, punct=True)
    text_list=[i for i in text.split() if i not in german_stop_words]
    text=" ".join(text_list)
    
    #generate wordcloud
    word_cloud = WordCloud(collocations = False, background_color = 'white').generate(text)
    
    word_cloud.to_file(filename+"_wordcloud_headline_pre_covid"+".png")
    
def getwords_headline_post(df: pd.DataFrame, filename: str, german_stop_words: list[str]) -> None:
    """
    wordcloud for post covid using headline
    :param df: dataframe
    :param filename: input filename used to generate output filename

    """
    
    df=df.copy()
    #drop pre covid data
    for index, row in df.iterrows():
        if int(row["YEAR"])<2021:
            df.drop(index, inplace=True)
    df.dropna(inplace = True)
    
    #get text
    text_list=df["HEADLINE"].tolist()
    text=" ".join(text_list)
    text=text.lower()
    text=cleantext.clean(text, extra_spaces=True, lowercase=True, numbers=True, punct=True)
    text_list=[i for i in text.split() if i.strip() not in german_stop_words]
    text=" ".join(text_list)
    
    #generate wordcloud
    word_cloud = WordCloud(collocations = False, background_color = 'white').generate(text)
    
    # res_file = filename.rsplit('.', 1)[0] 
    #save wordcloud
    word_cloud.to_file(filename+"_wordcloud_headline_post_covid"+".png")    



        

def get_wordcloud(inputfiles: list[str], name: str or None = None, ignore_words: list[str] = ["spiegel","faz","die","mehr","sei","sagte"]) -> None:
    """
    generate wordcloud
    :param inputfile: input filename(.xlsx)

    """
    
    # generate name to save files
    if name is None:
        starts = [file.split('/')[-1][:3] for file in inputfiles]
    name = f"{'_'.join([s for s in starts])}"
    
    # append all the passed dataframes into one big one
    df = pd.read_excel(inputfiles[0],nrows=3000)
    for file in inputfiles[1:]:
        df = df.append(pd.read_excel(file,nrows=3000),ignore_index=True)

    # convert dates from string to datetime
    df['DATE'] = pd.to_datetime(df['DATE'])
    df['YEAR'] = df['DATE'].dt.year    

    
    # create list of stopwords and add all the words that should be ignored
    german_stop_words = list(stopwords.words('german'))+ignore_words
    for i in range(len(german_stop_words)):
        german_stop_words[i]=german_stop_words[i].lower()
    # generate the plots
    getwords_content_pre(df,name,german_stop_words)
    getwords_content_post(df,name,german_stop_words)
    getwords_headline_pre(df,name,german_stop_words)
    getwords_headline_post(df,name,german_stop_words)
    
    
    
    
if __name__=="__main__":
    # parser = argparse.ArgumentParser(description='Generate wordcloud')
    # parser.add_argument('-f','--file', help='input file name(with sentiments)', default="SPIEGEL_SCRAPING_BEREINIGT_12.07.2022_sentiments.xlsx")
    # args = vars(parser.parse_args())
    get_wordcloud(inputfiles = ["FAZ_SCRAPING_BEREINIGT_12.07.2022.xlsx","BILD_SCRAPING_BEREINIGT_12.07.2022.xlsx","SUEDDEUTSCHE_SCRAPING_BEREINIGT_12.07.2022.xlsx","SPIEGEL_SCRAPING_BEREINIGT_12.07.2022.xlsx"])
    
       
