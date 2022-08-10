import pandas as pd
from vaderSentimentGER import SentimentIntensityAnalyzer
import tqdm
from sklearn import preprocessing
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import argparse
from nltk.corpus import stopwords



    
def getwords_content_pre(df,filename,german_stop_words):
    """
    wordcloud for pre covid using content
    :param df: dataframe
    :param filename: input filename used to generate output filename

    """
    
    #df=pd.read_excel("SPIEGEL_SCRAPING_BEREINIGT_12.07.2022_sentiments.xlsx")
    df=df[df['ERA'] == "pre"].copy()
    df.dropna(inplace = True)
    
    
    print(german_stop_words)
    
    
    #get text
    text_list=df["CONTENT"].tolist()
    text=" ".join(text_list)
    text=text.lower()
    
    text_list=[i for i in text.split() if i not in german_stop_words]
    text=" ".join(text_list)
    
    #generate wordcloud
    word_cloud = WordCloud(collocations = False, background_color = 'white').generate(text)
    
    res_file = filename.rsplit('.', 1)[0] 
    word_cloud.to_file(res_file+"_wordcloud_content_pre_covid"+".png")
   
    

def getwords_content_post(df,filename,german_stop_words):
    """
    wordcloud for post covid using content
    :param df: dataframe
    :param filename: input filename used to generate output filename

    """
    
    df=df[df['ERA'] == "post"].copy()
    df.dropna(inplace = True)
    
    #get text
    text_list=df["CONTENT"].tolist()
    text=" ".join(text_list)
    text=text.lower()
    
    text_list=[i for i in text.split() if i not in german_stop_words]
    text=" ".join(text_list)
    
    #generate wordcloud
    word_cloud = WordCloud(collocations = False, background_color = 'white').generate(text)
    
    res_file = filename.rsplit('.', 1)[0] 
    word_cloud.to_file(res_file+"_wordcloud_content_post_covid"+".png")

def getwords_headline_pre(df,filename,german_stop_words):
    """
    wordcloud for pre covid using headline
    :param df: dataframe
    :param filename: input filename used to generate output filename

    """
    
    df=df[df['ERA'] == "pre"].copy()
    df.dropna(inplace = True)
    
    #get text
    text_list=df["HEADLINE"].tolist()
    text=" ".join(text_list)
    text=text.lower()
    
    text_list=[i for i in text.split() if i not in german_stop_words]
    text=" ".join(text_list)
    
    #generate wordcloud
    word_cloud = WordCloud(collocations = False, background_color = 'white').generate(text)
    
    res_file = filename.rsplit('.', 1)[0] 
    word_cloud.to_file(res_file+"_wordcloud_headline_pre_covid"+".png")
    
def getwords_headline_post(df,filename,german_stop_words):
    """
    wordcloud for post covid using headline
    :param df: dataframe
    :param filename: input filename used to generate output filename

    """
    
    df=df[df['ERA'] == "post"].copy()
    df.dropna(inplace = True)
    
    #get text
    text_list=df["HEADLINE"].tolist()
    text=" ".join(text_list)
    text=text.lower()
    
    text_list=[i for i in text.split() if i not in german_stop_words]
    text=" ".join(text_list)
    
    #generate wordcloud
    word_cloud = WordCloud(collocations = False, background_color = 'white').generate(text)
    
    res_file = filename.rsplit('.', 1)[0] 
    word_cloud.to_file(res_file+"_wordcloud_headline_post_covid"+".png")    



        

def get_wordcloud(inputfile):
    """
    generate wordcloud
    :param inputfile: input filename(.xlsx)

    """
    
    df=pd.read_excel(inputfile)
    german_stop_words = stopwords.words('german')
    getwords_content_pre(df,inputfile,german_stop_words)
    getwords_content_post(df,inputfile,german_stop_words)
    getwords_headline_pre(df,inputfile,german_stop_words)
    getwords_headline_post(df,inputfile,german_stop_words)
    
    
    
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Generate wordcloud')
    parser.add_argument('-f','--file', help='input file name(with sentiments)', default="SPIEGEL_SCRAPING_BEREINIGT_12.07.2022_sentiments.xlsx")
    args = vars(parser.parse_args())
    get_wordcloud(args['file'])
    
       
