import pandas as pd
from vaderSentimentGER import SentimentIntensityAnalyzer
import tqdm
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import argparse
import matplotlib as mpl
label_size = 8
mpl.rcParams['xtick.labelsize'] = label_size 


def getrolling_content_pre(df,filename):
    """
    get pre covid analysis considering content  
    :param df: dataframe
    :param filename: input filename used to generate output filename

    """
    #df=pd.read_excel("SPIEGEL_SCRAPING_BEREINIGT_12.07.2022_sentiments.xlsx")
    df=df[df['ERA'] == "pre"].copy()
    
    res_file = filename.rsplit('.', 1)[0] 
    df["DATE"] = pd.to_datetime(df["DATE"])
    df=df.sort_values('DATE')
    df1=df[["DATE","PUBLISHER"]]
    #groupby
    res = df1.groupby(['DATE']).value_counts(normalize=False)
    res=res.reset_index()
    res.columns = ['DATE', 'PUBLISHER','COUNTS']
    average_per_day=res['COUNTS'].mean()
    
        
    df1=df[["DATE","CONTENT_SENTIMENT_SCORE"]]
    
    #get rolling mean
    
    #res = df1.groupby('DATE', as_index=False, sort=False)['CONTENT_SENTIMENT_SCORE'].mean()
    res = df1.groupby('DATE', as_index=False, sort=False)['CONTENT_SENTIMENT_SCORE'].mean()
    res['rolling_mean'] = res['CONTENT_SENTIMENT_SCORE'].rolling(28).mean()
    avg_sentiment_list=[0]+res["rolling_mean"].tolist()[1:]
    avg_sentiment_list=avg_sentiment_list[::28]
    date_list=["week"+str(4*i) for i in range(len(avg_sentiment_list))]
    
    #plotting average sentiment
    plt.clf()
    plt.title("Average sentiment pre covid \n considering content")
    plt.xlabel("Time")
    plt.ylabel("Score")
    plt.xticks(rotation=90)
    plt.xticks(fontsize=8)
    plt.plot(date_list,avg_sentiment_list)
    plt.savefig(res_file+"_avg_sentiments_precovid_content"+".png") 
    plt.xticks(rotation=0)
    plt.clf()
    return average_per_day

def getrolling_content_post(df,filename):
    """
    get post covid analysis considering content  
    :param df: dataframe
    :param filename: input filename used to generate output filename

    """
    #df=pd.read_excel("SPIEGEL_SCRAPING_BEREINIGT_12.07.2022_sentiments.xlsx")
    df=df[df['ERA'] == "post"].copy()
    res_file = filename.rsplit('.', 1)[0] 
    df["DATE"] = pd.to_datetime(df["DATE"])
    df=df.sort_values('DATE')
    df1=df[["DATE","PUBLISHER"]]
    #groupby
    res = df1.groupby(['DATE']).value_counts(normalize=False)
    res=res.reset_index()
    res.columns = ['DATE', 'PUBLISHER','COUNTS']
    average_per_day=res['COUNTS'].mean()
    
        
    df1=df[["DATE","CONTENT_SENTIMENT_SCORE"]]
    
    #get rolling mean
    res = df1.groupby('DATE', as_index=False, sort=False)['CONTENT_SENTIMENT_SCORE'].mean()
    res['rolling_mean'] = res['CONTENT_SENTIMENT_SCORE'].rolling(28).mean()
    avg_sentiment_list=[0]+res["rolling_mean"].tolist()[1:]
    avg_sentiment_list=avg_sentiment_list[::28]
    date_list=["week"+str(4*i) for i in range(len(avg_sentiment_list))]
    
    #plotting average sentiment
    plt.clf()
    plt.title("Average sentiment post covid \n considering content")
    plt.xlabel("Time")
    plt.ylabel("Score")
    plt.xticks(rotation=90)
    plt.xticks(fontsize=8)
    plt.plot(date_list,avg_sentiment_list)
    plt.savefig(res_file+"_avg_sentiments_postcovid_content"+".png") 
    plt.xticks(rotation=0)
    plt.clf()
    return average_per_day

def getrolling_headline_pre(df,filename):
    """
    get pre covid analysis considering headline
    :param df: dataframe
    :param filename: input filename used to generate output filename

    """
    
    df=df[df['ERA'] == "pre"].copy()
    res_file = filename.rsplit('.', 1)[0] 
    df["DATE"] = pd.to_datetime(df["DATE"])
    df=df.sort_values('DATE')
    #get rolling mean   
    df1=df[["DATE","HEADLINE_SENTIMENT_SCORE"]]
    res = df1.groupby('DATE', as_index=False, sort=False)['HEADLINE_SENTIMENT_SCORE'].mean()
    res['rolling_mean'] = res['HEADLINE_SENTIMENT_SCORE'].rolling(28).mean()
    avg_sentiment_list=[0]+res["rolling_mean"].tolist()[1:]
    avg_sentiment_list=avg_sentiment_list[::28]
    date_list=["week"+str(4*i) for i in range(len(avg_sentiment_list))]
    
    #plotting average sentiment
    plt.clf()
    plt.title("Average sentiment pre covid \n considering headline")
    plt.xlabel("Time")
    plt.ylabel("Score")
    plt.xticks(rotation=90)
    plt.xticks(fontsize=8)
    plt.plot(date_list,avg_sentiment_list)
    plt.savefig(res_file+"_avg_sentiments_precovid_headline"+".png") 
    plt.xticks(rotation=0)
    plt.clf()
    
    
def getrolling_headline_post(df,filename):
    """
    get post covid analysis considering headline
    :param df: dataframe
    :param filename: input filename used to generate output filename

    """
    df=df[df['ERA'] == "post"].copy()
    res_file = filename.rsplit('.', 1)[0] 
    df["DATE"] = pd.to_datetime(df["DATE"])
    df=df.sort_values('DATE')    
    #get rolling mean    
    df1=df[["DATE","HEADLINE_SENTIMENT_SCORE"]]
    res = df1.groupby('DATE', as_index=False, sort=False)['HEADLINE_SENTIMENT_SCORE'].mean()
    res['rolling_mean'] = res['HEADLINE_SENTIMENT_SCORE'].rolling(28).mean()
    avg_sentiment_list=[0]+res["rolling_mean"].tolist()[1:]
    avg_sentiment_list=avg_sentiment_list[::28]
    date_list=["week"+str(4*i) for i in range(len(avg_sentiment_list))]
    plt.clf()
    
    #plotting average sentiment
    plt.title("Average sentiment post covid \n considering headline")
    plt.xlabel("Time")
    plt.ylabel("Score")
    plt.xticks(rotation=90)
    plt.xticks(fontsize=8)
    plt.plot(date_list,avg_sentiment_list)
    plt.savefig(res_file+"_avg_sentiments_postcovid_headline"+".png") 
    plt.xticks(rotation=0)
    plt.clf()
    


def getrolling_headline_overall(df,filename):
    """
    get positive, negative, neutral, total articles along with average sentiment overtime considering headline    
    :param df: dataframe
    :param filename: input filename used to generate output filename

    """
    
    #df=pd.read_excel("SPIEGEL_SCRAPING_BEREINIGT_12.07.2022_sentiments.xlsx")
    
    df1=df[["DATE","HEADLINE_SENTIMENT"]]
    
    #get rolling count
        
    out = (
        df1
        .groupby(['DATE', 'HEADLINE_SENTIMENT']) # set the columns as index
        .size() # aggregate by row count
        .unstack(fill_value=0) # move 'HEADLINE_SENTIMENT' index level to columns
        .sort_index()
    )

    out = out.rolling('28D').sum()
     
     
    neutral_list=[0]+out["neutral"].tolist()[1:]
    positive_list=[0]+out["positive"].tolist()[1:]
    negative_list=[0]+out["negative"].tolist()[1:]
    
    #calculating total no of articles
    
   
    out["total"]=0
    for index,row in out.iterrows():
        neutral=int(row["neutral"])
        positive=int(row["positive"])
        negative=int(row["negative"])
        total=neutral+positive+negative
        out.loc[index,'total'] = total
        
   
    total_list=[0]+out["total"].tolist()[1:]   
       
    #choose every 28th for easy plotting
    neutral_list=neutral_list[::28]
    positive_list=positive_list[::28]
    negative_list=negative_list[::28]
    total_list=total_list[::28]
    date_list=["week"+str(4*i) for i in range(len(neutral_list))]
    
    
    
    #plotting positive negative neutral and total articles overtime
    fig, ax = plt.subplots()
    fig.suptitle('Total positive negative and neutral \n articles published overtime \n considering Headline \n (rolling count)')
    fig.supxlabel('Time')
    fig.supylabel('Count')
    for Y in [(neutral_list,"neutral","red"),(positive_list,"positive","blue"),(negative_list,"negative","green"),(total_list,"total","orange")]:
        ax.plot(date_list, Y[0],color=Y[2], label=Y[1])
        ax.legend(loc="upper right")
    fig.autofmt_xdate()
    res_file = filename.rsplit('.', 1)[0] 
    plt.savefig(res_file+"_positive_negative_total_overtime_overall_headline"+".png") 
    
    df1=df[["DATE","HEADLINE_SENTIMENT_SCORE"]]
    res = df1.groupby('DATE', as_index=False, sort=False)['HEADLINE_SENTIMENT_SCORE'].mean()
    res['rolling_mean'] = res['HEADLINE_SENTIMENT_SCORE'].rolling(28).mean()
    avg_sentiment_list=[0]+res["rolling_mean"].tolist()[1:]
    avg_sentiment_list=avg_sentiment_list[::28]
    
    #plotting average sentiment
    plt.clf()
    plt.title("Average sentiment over time")
    plt.xticks(rotation=90)
    plt.xticks(fontsize=8)
    plt.xlabel("Time")
    plt.ylabel("Score")
    plt.plot(date_list,avg_sentiment_list)
    plt.savefig(res_file+"_avg_sentiments_overtime_overall_headline"+".png") 
    plt.xticks(rotation=0)
    plt.clf()


def getrolling_content_overall(df,filename):
    """
    get positive, negative, neutral, total articles along with average sentiment overtime considering content    
    :param df: dataframe
    :param filename: input filename used to generate output filename

    """
    #df=pd.read_excel("SPIEGEL_SCRAPING_BEREINIGT_12.07.2022_sentiments.xlsx")
    
    df1=df[["DATE","CONTENT_SENTIMENT"]]
    
    #df1=df1.head(10000) 
    
    #get rolling count   
    out = (
        df1
        .groupby(['DATE', 'CONTENT_SENTIMENT']) # set the columns as index
        .size() # aggregate by row count
        .unstack(fill_value=0) # move 'Variable' index level to columns
        .sort_index()
    )

    out = out.rolling('28D').sum()
     
    #make first value as 0  
    neutral_list=[0]+out["neutral"].tolist()[1:]
    positive_list=[0]+out["positive"].tolist()[1:]
    negative_list=[0]+out["negative"].tolist()[1:]
    
    #calculating total no of articles
    
   
    out["total"]=0
    for index,row in out.iterrows():
        neutral=int(row["neutral"])
        positive=int(row["positive"])
        negative=int(row["negative"])
        total=neutral+positive+negative
        out.loc[index,'total'] = total
        
   
    total_list=[0]+out["total"].tolist()[1:]   
    
    
    #choose every 28th for easy plotting
    neutral_list=neutral_list[::28]
    positive_list=positive_list[::28]
    negative_list=negative_list[::28]
    total_list=total_list[::28]
    date_list=["week"+str(4*i) for i in range(len(neutral_list))]
        
    #plotting positive negative neutral and total articles overtime
    fig, ax = plt.subplots()
    fig.suptitle('Total positive negative and neutral \n articles published overtime \n considering Content \n(rolling count)')
    fig.supxlabel('Time')
    fig.supylabel('Count')
    for Y in [(neutral_list,"neutral","red"),(positive_list,"positive","blue"),(negative_list,"negative","green"),(total_list,"total","orange")]:
        ax.plot(date_list, Y[0],color=Y[2], label=Y[1])
        ax.legend(loc="upper right")
    fig.autofmt_xdate()
    res_file = filename.rsplit('.', 1)[0] 
    plt.savefig(res_file+"_positive_negative_total_overtime_overall_content"+".png") 
    
    df1=df[["DATE","CONTENT_SENTIMENT_SCORE"]]
    res = df1.groupby('DATE', as_index=False, sort=False)['CONTENT_SENTIMENT_SCORE'].mean()
    res['rolling_mean'] = res['CONTENT_SENTIMENT_SCORE'].rolling(28).mean()
    avg_sentiment_list=[0]+res["rolling_mean"].tolist()[1:]
    avg_sentiment_list=avg_sentiment_list[::28]
    
    #plotting average sentiment overtime
    plt.clf()
    plt.title("Average sentiment over time")
    plt.xlabel("Time")
    plt.ylabel("Score")
    plt.xticks(rotation=90)
    plt.xticks(fontsize=8)
    plt.plot(date_list,avg_sentiment_list)
    plt.savefig(res_file+"_avg_sentiments_overtime_overall_content"+".png") 
    plt.xticks(rotation=0)
    plt.clf()
    


def getrolling_headline_topics(dataframe,topics,filename):
    """
    get positive, negative, neutral, total articles along with average sentiment overtime considering headline for each topic
    :param df: dataframe
    :param topics: list of topics
    :param filename: input filename used to generate output filename

    """
    #df=pd.read_excel("SPIEGEL_SCRAPING_BEREINIGT_12.07.2022_sentiments.xlsx")
    for topic in topics:
        df=dataframe[dataframe['TOPICS'] == topic].copy()
        df["DATE"] = pd.to_datetime(df["DATE"])
        df=df.sort_values('DATE')
        df1=df[["DATE","HEADLINE_SENTIMENT"]]
        
        #df1=df1.head(10000)    
        out = (
            df1
            .groupby(['DATE', 'HEADLINE_SENTIMENT']) # set the columns as index
            .size() # aggregate by row count
            .unstack(fill_value=0) # move 'Variable' index level to columns
            .sort_index()
        )
        
        #rolling=> 28 days
        out = out.rolling('28D').sum()
        
        #make first value as 0  
        neutral_list=[0]+out["neutral"].tolist()[1:]
        positive_list=[0]+out["positive"].tolist()[1:]
        negative_list=[0]+out["negative"].tolist()[1:]
        
        #calculating total no of articles
        
        
        out["total"]=0
        for index,row in out.iterrows():
            neutral=int(row["neutral"])
            positive=int(row["positive"])
            negative=int(row["negative"])
            total=neutral+positive+negative
            out.loc[index,'total'] = total
            
        
        total_list=[0]+out["total"].tolist()[1:]   
        #out=out.reset_index()
        
        
        #date_list=out["DATE"].tolist() 
        #print(date_list)
        #date_list=date_list[::28]
        
        #choose every 28th for easy plotting
        neutral_list=neutral_list[::28]
        positive_list=positive_list[::28]
        negative_list=negative_list[::28]
        total_list=total_list[::28]
        date_list=["week"+str(4*i) for i in range(len(neutral_list))]
        
        
        
        #plotting
        fig, ax = plt.subplots()
        fig.suptitle('Total positive negative and \n neutral articles published \n overtime considering Headline \n for '+str(topic)+" (rolling count)")
        fig.supxlabel('Time')
        fig.supylabel('Count')
        for Y in [(neutral_list,"neutral","red"),(positive_list,"positive","blue"),(negative_list,"negative","green"),(total_list,"total","orange")]:
            ax.plot(date_list, Y[0],color=Y[2], label=Y[1])
            ax.legend(loc="upper right")
        fig.autofmt_xdate()
        res_file = filename.rsplit('.', 1)[0] 
        plt.savefig(res_file+"_positive_negative_total_headline_"+str(topic)+".png") 
        
        df1=df[["DATE","HEADLINE_SENTIMENT_SCORE"]]
        res = df1.groupby('DATE', as_index=False, sort=False)['HEADLINE_SENTIMENT_SCORE'].mean()
        res['rolling_mean'] = res['HEADLINE_SENTIMENT_SCORE'].rolling(28).mean()
        avg_sentiment_list=[0]+res["rolling_mean"].tolist()[1:]
        avg_sentiment_list=avg_sentiment_list[::28]
        
        #plotting average sentiment
        plt.clf()
        plt.title("Average sentiment over time \n considering headline for \n "+str(topic))
        plt.xlabel("Time")
        plt.ylabel("Score")
        plt.xticks(rotation=90)
        plt.xticks(fontsize=8)
        plt.plot(date_list,avg_sentiment_list)
        plt.savefig(res_file+"_avg_sentiments_headline_"+str(topic)+".png") 
        plt.xticks(rotation=0)
        plt.clf()

    

def getrolling_content_topics(dataframe,topics,filename):
    """
    get positive, negative, neutral, total articles along with average sentiment overtime considering content for each topic
    :param df: dataframe
    :param topics: list of topics
    :param filename: input filename used to generate output filename

    """
    #df=pd.read_excel("SPIEGEL_SCRAPING_BEREINIGT_12.07.2022_sentiments.xlsx")
    for topic in topics:
        df=dataframe[dataframe['TOPICS'] == topic].copy()
        df["DATE"] = pd.to_datetime(df["DATE"])
        df=df.sort_values('DATE')
        df1=df[["DATE","CONTENT_SENTIMENT"]]
        
        #df1=df1.head(10000)    
        out = (
            df1
            .groupby(['DATE', 'CONTENT_SENTIMENT']) # set the columns as index
            .size() # aggregate by row count
            .unstack(fill_value=0) # move 'Variable' index level to columns
            .sort_index()
        )
        
        out = out.rolling('28D').sum()
        
        #make first value as 0  
        neutral_list=[0]+out["neutral"].tolist()[1:]
        positive_list=[0]+out["positive"].tolist()[1:]
        negative_list=[0]+out["negative"].tolist()[1:]
        
        #calculating total no of articles
        
        
        out["total"]=0
        for index,row in out.iterrows():
            neutral=int(row["neutral"])
            positive=int(row["positive"])
            negative=int(row["negative"])
            total=neutral+positive+negative
            out.loc[index,'total'] = total
            
        
        total_list=[0]+out["total"].tolist()[1:]   
        #out=out.reset_index()
        
        
        #date_list=out["DATE"].tolist() 
        #print(date_list)
        #date_list=date_list[::28]
        
        #choose every 28th for easy plotting
        neutral_list=neutral_list[::28]
        positive_list=positive_list[::28]
        negative_list=negative_list[::28]
        total_list=total_list[::28]
        date_list=["week"+str(4*i) for i in range(len(neutral_list))]
        
        
        
        #plotting positive, negative, neutral, total
        fig, ax = plt.subplots()
        fig.suptitle('Total positive negative and \n neutral articles published overtime \n considering content for \n '+str(topic)+" (rolling count)")
        fig.supxlabel('Time')
        fig.supylabel('Count')
        for Y in [(neutral_list,"neutral","red"),(positive_list,"positive","blue"),(negative_list,"negative","green"),(total_list,"total","orange")]:
            ax.plot(date_list, Y[0],color=Y[2], label=Y[1])
            ax.legend(loc="upper right")
        fig.autofmt_xdate()
        res_file = filename.rsplit('.', 1)[0] 
        plt.savefig(res_file+"_positive_negative_total_content_"+str(topic)+".png") 
        
        df1=df[["DATE","CONTENT_SENTIMENT_SCORE"]]
        res = df1.groupby('DATE', as_index=False, sort=False)['CONTENT_SENTIMENT_SCORE'].mean()
        
        res['rolling_mean'] = res['CONTENT_SENTIMENT_SCORE'].rolling(28).mean()
        avg_sentiment_list=[0]+res["rolling_mean"].tolist()[1:]
        avg_sentiment_list=avg_sentiment_list[::28]
        
        #plotting average sentiment
        plt.clf()
        plt.title("Average sentiment over time considering content for "+str(topic))
        plt.xlabel("Time")
        plt.ylabel("Score")
        plt.xticks(rotation=90)
        plt.xticks(fontsize=8)
        plt.plot(date_list,avg_sentiment_list)
        plt.savefig(res_file+"_avg_sentiments_content_"+str(topic)+".png") 
        plt.xticks(rotation=0)
        plt.clf()

        

def get_sentiments(inputfile):
    """
    :param inputfile: input filename(.xlsx)

    """
    analyzer = SentimentIntensityAnalyzer()
    df=pd.read_excel(inputfile)
    #df=df.head(500)
    
    #create columns
    df['HEADLINE_SENTIMENT'] = ''
    df['CONTENT_SENTIMENT'] = ''
    df['HEADLINE_SENTIMENT_SCORE'] = 0
    df['CONTENT_SENTIMENT_SCORE'] = 0
    df['ERA']='' #pre-covid ot post-covid
    
    #make sure columns are string
    df['HEADLINE'] = df['HEADLINE'].astype(str)
    df['CONYTENT'] = df['CONTENT'].astype(str)
    df['HEADLINE']  = df['HEADLINE'].fillna('')
    df['CONTENT']  = df['CONTENT'].fillna('')
    df['YEAR']=df['DATE'].dt.year
    
    for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
        
       
        # calculate headline sentiment and store
        
        if int(row["YEAR"])<2021:
            era="pre"
        else:
            era="post"    
        df.loc[index,'ERA']=era
        vs = analyzer.polarity_scores(row["HEADLINE"])
        score = vs['compound']
        df.loc[index,'HEADLINE_SENTIMENT_SCORE'] = score
        if score >= 0.05:
            score = 'positive'
            df.loc[index,'HEADLINE_SENTIMENT'] = score
            
        elif score <= -0.05:
            score = 'negative'
            df.loc[index,'HEADLINE_SENTIMENT'] = score
           
        else:
            score = 'neutral'
            df.loc[index,'HEADLINE_SENTIMENT'] = score
           
            
        #calculate content sentiment  and store     
        vs = analyzer.polarity_scores(row["CONTENT"])
        score = vs['compound']
        df.loc[index,'CONTENT_SENTIMENT_SCORE'] = score
        if score >= 0.05:
            score = 'positive'
            df.loc[index,'CONTENT_SENTIMENT'] = score
            
        elif score <= -0.05:
            score = 'negative'
            df.loc[index,'CONTENT_SENTIMENT'] = score
            
        else:
            score = 'neutral'
            df.loc[index,'CONTENT_SENTIMENT'] = score
    res = inputfile.rsplit('.', 1)[0]         
    df.to_excel(str(res)+"_sentiments.xlsx",index=False)  
    df["DATE"] = pd.to_datetime(df["DATE"])
    df=df.sort_values('DATE')
    topics=df['TOPICS'].unique()
    
    #get analysis
    getrolling_headline_overall(df,inputfile)   
    getrolling_headline_topics(df,topics,inputfile)
    getrolling_content_overall(df,inputfile)
    getrolling_content_topics(df,topics,inputfile)
    avg_per_day_pre=int(getrolling_content_pre(df,inputfile))
    avg_per_day_post=int(getrolling_content_post(df,inputfile))
    getrolling_headline_pre(df,inputfile)
    getrolling_headline_post(df,inputfile)
    
    #plotting 
    plt.clf()
    data = {'Pre-Covid':avg_per_day_pre, 'Post-Covid':avg_per_day_post,}
    keys = list(data.keys())
    values = list(data.values())
   
    fig, ax = plt.subplots()
     # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Articles per day')
    ax.set_title('Articles per day before \n and after covid')
    ax.bar(keys, values, color ='maroon',width = 0.4)
    
    
    plt.savefig(str(res)+"_average_articles_per_day_before_after_covid.png")
    
    
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Analyse news data')
    parser.add_argument('-f','--file', help='input file name', default="SPIEGEL_SCRAPING_BEREINIGT_12.07.2022.xlsx")
    args = vars(parser.parse_args())
    get_sentiments(args['file'])
    
       
