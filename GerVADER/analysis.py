import pandas as pd
from vaderSentimentGER import SentimentIntensityAnalyzer
import tqdm
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import argparse
import matplotlib as mpl
import datetime
label_size = 12
mpl.rcParams['xtick.labelsize'] = label_size 
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "lmodern",
    "font.size": 14,
})

def format_plot(x: None or list = None, y: list = None, 
                title: str = None, x_label: str = None,
                y_label: str = None, name: str = None):
    """Formats a plot properly for all the different functions and saves 
    it under the specified filename.

    Args:
        x (list, optional): x values of plot. Defaults to None.
        y (list, optional): y values of plot. Defaults to None.
        title (str, optional): title of plot. Defaults to None.
        x_label (str, optional): x axis label of plot. Defaults to None.
        y_label (str, optional): y axis label of plot. Defaults to None.
        name (str, optional): name of file in which plot is saved. Defaults to None.
    """
    date_list=["week "+str(i//7) for i in range(len(y))] # assuming data exists for each day
    plt.clf()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=90)
    plt.xticks(ticks=np.arange(len(y))[::28], labels=date_list[::28])
    plt.xticks(fontsize=8)
    if x is not None:
        plt.plot(x,y)
    else:
        plt.plot(y)
    plt.savefig(f"{name}.png", bbox_inches='tight') 
    plt.xticks(rotation=0)
    plt.clf()

def getrolling_content_pre(df: pd.DataFrame, filename: str) -> None:
    """
    get pre covid analysis considering content
    :param df: dataframe
    :param filename: input filename used to generate output filename
    """

    df=df[df['ERA'] == "pre"].copy()
    
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
    
    title = "Average sentiment pre covid \n considering content" 
    name = f"{filename}_avg_sentiments_precovid_content"
     
    # create and save plot
    format_plot(y=avg_sentiment_list, title=title, x_label="Time", y_label="Score", name=name)

    return average_per_day



def getrolling_content_post(df: pd.DataFrame, filename: str) -> None:
    """
    get post covid analysis considering content  
    :param df: dataframe
    :param filename: input filename used to generate output filename
    """

    df=df[df['ERA'] == "post"].copy()
    df["DATE"] = pd.to_datetime(df["DATE"])
    df=df.sort_values('DATE')
    df1=df[["DATE","PUBLISHER"]]
    
    # group by
    res = df1.groupby(['DATE']).value_counts(normalize=False)
    res=res.reset_index()
    res.columns = ['DATE', 'PUBLISHER','COUNTS']
    average_per_day=res['COUNTS'].mean()
    
        
    df1=df[["DATE","CONTENT_SENTIMENT_SCORE"]]
    
    #get rolling mean
    res = df1.groupby('DATE', as_index=False, sort=False)['CONTENT_SENTIMENT_SCORE'].mean()
    res['rolling_mean'] = res['CONTENT_SENTIMENT_SCORE'].rolling(28).mean()
    avg_sentiment_list=[0]+res["rolling_mean"].tolist()[1:]
    # avg_sentiment_list=avg_sentiment_list
    
    #plotting average sentiment
    title = "Average sentiment post covid \n considering content"
    name= f"{filename}_avg_sentiments_postcovid_content"
    format_plot(y=avg_sentiment_list, title=title, x_label="Time", y_label="Score", name=name)
    
    return average_per_day

def getrolling_headline_pre(df: pd.DataFrame, filename: str) -> None:
    """
    get pre covid analysis considering headline
    :param df: dataframe
    :param filename: input filename used to generate output filename

    """
    
    df=df[df['ERA'] == "pre"].copy()
    df["DATE"] = pd.to_datetime(df["DATE"])
    df=df.sort_values('DATE')
    #get rolling mean   
    df1=df[["DATE","HEADLINE_SENTIMENT_SCORE"]]
    res = df1.groupby('DATE', as_index=False, sort=False)['HEADLINE_SENTIMENT_SCORE'].mean()
    res['rolling_mean'] = res['HEADLINE_SENTIMENT_SCORE'].rolling(28).mean()
    avg_sentiment_list=[0]+res["rolling_mean"].tolist()[1:]
    
    #plotting average sentiment
    title = "Average sentiment pre covid \n considering headline"
    name= f"{filename}_avg_sentiments_precovid_headline"
    format_plot(y=avg_sentiment_list, title=title, x_label="Time", y_label="Score", name=name)
    
    
def getrolling_headline_post(df: pd.DataFrame, filename: str) -> None:
    """
    get post covid analysis considering headline
    :param df: dataframe
    :param filename: input filename used to generate output filename

    """
    df=df[df['ERA'] == "post"].copy()
    df["DATE"] = pd.to_datetime(df["DATE"])
    df=df.sort_values('DATE')    
    #get rolling mean    
    df1=df[["DATE","HEADLINE_SENTIMENT_SCORE"]]
    res = df1.groupby('DATE', as_index=False, sort=False)['HEADLINE_SENTIMENT_SCORE'].mean()
    res['rolling_mean'] = res['HEADLINE_SENTIMENT_SCORE'].rolling(28).mean()
    avg_sentiment_list=[0]+res["rolling_mean"].tolist()[1:]
    
    #plotting average sentiment
    title = "Average sentiment post covid \n considering headline"
    name= f"{filename}_avg_sentiments_postcovid_headline"
    format_plot(y=avg_sentiment_list, title=title, x_label="Time", y_label="Score", name=name)
    


def getrolling_headline_overall(df: pd.DataFrame, filename: str) -> None:
    """
    get positive, negative, neutral, total articles along with average sentiment overtime considering headline    
    :param df: dataframe
    :param filename: input filename used to generate output filename

    """
    
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
       
    date_list=["week "+str(i//7) for i in range(len(neutral_list))]
    
    
    #plotting positive negative neutral and total articles overtime
    fig, ax = plt.subplots()
    fig.suptitle('Total positive negative and neutral \n articles published overtime \n considering Headline \n (rolling count)')
    fig.supxlabel('Time')
    fig.supylabel('Count')
    for Y in [(neutral_list,"neutral","red"),(positive_list,"positive","blue"),(negative_list,"negative","green"),(total_list,"total","orange")]:
        ax.plot(np.arange(len(neutral_list)), Y[0],color=Y[2], label=Y[1])
        ax.legend(loc="upper right")
    fig.autofmt_xdate()
    plt.xticks(ticks=np.arange(len(neutral_list))[::28], labels=date_list[::28])
    plt.savefig(filename+"_positive_negative_total_overtime_overall_headline"+".png", bbox_inches='tight') 
    
    df1=df[["DATE","HEADLINE_SENTIMENT_SCORE"]]
    res = df1.groupby('DATE', as_index=False, sort=False)['HEADLINE_SENTIMENT_SCORE'].mean()
    res['rolling_mean'] = res['HEADLINE_SENTIMENT_SCORE'].rolling(28).mean()
    avg_sentiment_list=[0]+res["rolling_mean"].tolist()[1:]
    
    #plotting average sentiment
    title = "Average sentiment over time based on Headlines"
    name= f"{filename}_avg_sentiments_overtime_overall_headline"
    format_plot(y=avg_sentiment_list, title=title, x_label="Time", y_label="Score", name=name)


def getrolling_content_overall(df: pd.DataFrame, filename: str) -> None:
    """
    get positive, negative, neutral, total articles along with average sentiment overtime considering content    
    :param df: dataframe
    :param filename: input filename used to generate output filename

    """
    
    df1=df[["DATE","CONTENT_SENTIMENT"]]
    
    
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
    
    
    date_list=["week"+str(i//7) for i in range(len(neutral_list))]
        
    #plotting positive negative neutral and total articles overtime
    fig, ax = plt.subplots()
    fig.suptitle('Total positive negative and neutral \n articles published overtime \n considering Content \n(rolling count)')
    fig.supxlabel('Time')
    fig.supylabel('Count')
    for Y in [(neutral_list,"neutral","red"),(positive_list,"positive","blue"),(negative_list,"negative","green"),(total_list,"total","orange")]:
        plt.xticks(ticks=np.arange(len(Y[0]))[::28], labels=date_list[::28])
        ax.plot(Y[0],color=Y[2], label=Y[1])
        ax.legend(loc="upper right")
    fig.autofmt_xdate()
    plt.savefig(filename+"_positive_negative_total_overtime_overall_content"+".png", bbox_inches='tight') 
    
    df1=df[["DATE","CONTENT_SENTIMENT_SCORE"]]
    res = df1.groupby('DATE', as_index=False, sort=False)['CONTENT_SENTIMENT_SCORE'].mean()
    res['rolling_mean'] = res['CONTENT_SENTIMENT_SCORE'].rolling(28).mean()
    avg_sentiment_list=[0]+res["rolling_mean"].tolist()[1:]
    avg_sentiment_list=avg_sentiment_list
    
    #plotting average sentiment overtime
    title = "Average sentiment over time based on Content"
    name= f"{filename}_avg_sentiments_overtime_overall_content"
    format_plot(y=avg_sentiment_list, title=title, x_label="Time", y_label="Score", name=name)
    


def getrolling_headline_topics(dataframe,topics,filename):
    """
    get positive, negative, neutral, total articles along with average sentiment overtime considering headline for each topic
    :param df: dataframe
    :param topics: list of topics
    :param filename: input filename used to generate output filename

    """
    for topic in topics:
        df=dataframe[dataframe['TOPICS'] == topic].copy()
        df["DATE"] = pd.to_datetime(df["DATE"])
        df=df.sort_values('DATE')
        df1=df[["DATE","HEADLINE_SENTIMENT"]]
        
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
        date_list=["week"+str(i//7) for i in range(len(neutral_list))]
        
        
        
        #plotting
        fig, ax = plt.subplots()
        fig.suptitle('Total positive negative and \n neutral articles published \n overtime considering Headline \n for '+str(topic)+" (rolling count)")
        fig.supxlabel('Time')
        fig.supylabel('Count')
        for Y in [(neutral_list,"neutral","red"),(positive_list,"positive","blue"),(negative_list,"negative","green"),(total_list,"total","orange")]:
            plt.xticks(ticks=np.arange(len(Y[0]))[::28], labels=date_list[::28])
            ax.plot(Y[0],color=Y[2], label=Y[1])
            ax.legend(loc="upper right")
        fig.autofmt_xdate()
        plt.savefig(filename+"_positive_negative_total_headline_"+str(topic)+".png", bbox_inches='tight') 
        
        df1=df[["DATE","HEADLINE_SENTIMENT_SCORE"]]
        res = df1.groupby('DATE', as_index=False, sort=False)['HEADLINE_SENTIMENT_SCORE'].mean()
        res['rolling_mean'] = res['HEADLINE_SENTIMENT_SCORE'].rolling(28).mean()
        avg_sentiment_list=[0]+res["rolling_mean"].tolist()[1:]
        # avg_sentiment_list=avg_sentiment_list
        
        #plotting average sentiment
        title = f"Average sentiment over time \n considering headline for \n {str(topic)}"
        name= f"{filename}_avg_sentiments_headline_{str(topic)}"
        format_plot(y=avg_sentiment_list, title=title, x_label="Time", y_label="Score", name=name)

    

def getrolling_content_topics(dataframe,topics,filename):
    """
    get positive, negative, neutral, total articles along with average sentiment overtime considering content for each topic
    :param df: dataframe
    :param topics: list of topics
    :param filename: input filename used to generate output filename

    """
    for topic in topics:
        df=dataframe[dataframe['TOPICS'] == topic].copy()
        df["DATE"] = pd.to_datetime(df["DATE"])
        df=df.sort_values('DATE')
        df1=df[["DATE","CONTENT_SENTIMENT"]]
        
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
        # neutral_list=neutral_list
        # positive_list=positive_list
        # negative_list=negative_list
        # total_list=total_list
        date_list=["week"+str(i//7) for i in range(len(neutral_list))]
        
        #plotting positive, negative, neutral, total
        fig, ax = plt.subplots()
        fig.suptitle('Total positive negative and \n neutral articles published overtime \n considering content for \n '+str(topic)+" (rolling count)")
        fig.supxlabel('Time')
        fig.supylabel('Count')
        for Y in [(neutral_list,"neutral","red"),(positive_list,"positive","blue"),(negative_list,"negative","green"),(total_list,"total","orange")]:
            plt.xticks(ticks=np.arange(len(Y[0]))[::28], labels=date_list[::28])
            ax.plot(Y[0],color=Y[2], label=Y[1])
            ax.legend(loc="upper right")
        fig.autofmt_xdate()
        plt.savefig(filename+"_positive_negative_total_content_"+str(topic)+".png", bbox_inches='tight') 
        
        df1=df[["DATE","CONTENT_SENTIMENT_SCORE"]]
        res = df1.groupby('DATE', as_index=False, sort=False)['CONTENT_SENTIMENT_SCORE'].mean()
        
        res['rolling_mean'] = res['CONTENT_SENTIMENT_SCORE'].rolling(28).mean()
        avg_sentiment_list=[0]+res["rolling_mean"].tolist()[1:]
        avg_sentiment_list=avg_sentiment_list
        
        #plotting average sentiment
        title = f"Average sentiment over time considering content for {str(topic)}"
        name= f"{filename}_avg_sentiments_content_{str(topic)}"
        format_plot(y=avg_sentiment_list, title=title, x_label="Time", y_label="Score", name=name)

        

def get_sentiments(inputfiles, name=None):
    """
    :param inputfile: input filename(.xlsx)
    :param name: name of the file to save
    """
    
    if name is None:
        starts = [file.split('/')[-1][:3] for file in inputfiles]
        name = f"{'_'.join([s for s in starts])}"
    analyzer = SentimentIntensityAnalyzer()
    
    df = pd.read_excel(inputfiles[0])
    for file in inputfiles[1:]:
        df = df.append(pd.read_excel(file), ignore_index=True)
    print(len(df))
    print(df)
        
    df.dropna(subset=['DATE']) 

    #create columns
    df['HEADLINE_SENTIMENT'] = ''
    df['CONTENT_SENTIMENT'] = ''
    df['HEADLINE_SENTIMENT_SCORE'] = 0
    df['CONTENT_SENTIMENT_SCORE'] = 0
    df['ERA']='' #pre-covid ot post-covid
    
    # make sure columns are string, if not typecast
    df['HEADLINE'] = df['HEADLINE'].astype(str)
    df['CONYTENT'] = df['CONTENT'].astype(str)
    df['HEADLINE']  = df['HEADLINE'].fillna('')
    df['CONTENT']  = df['CONTENT'].fillna('')
    df['DATE'] = pd.to_datetime(df['DATE'])
    df['YEAR'] = df['DATE'].dt.year
    
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
    df.to_excel(str(name)+"_sentiments.xlsx",index=False)  
    df["DATE"] = pd.to_datetime(df["DATE"])
    df=df.sort_values('DATE')
    topics=df['TOPICS'].unique()
    
    #get analysis
    getrolling_headline_overall(df,name)   
    getrolling_headline_topics(df,topics,name)
    getrolling_content_overall(df,name)
    getrolling_content_topics(df,topics,name)
    avg_per_day_pre=int(getrolling_content_pre(df,name))
    avg_per_day_post=int(getrolling_content_post(df,name))
    getrolling_headline_pre(df,name)
    getrolling_headline_post(df,name)
    
    #plotting 
    plt.clf()
    data = {'Pre-Covid':avg_per_day_pre, 'Post-Covid':avg_per_day_post,}
    keys = list(data.keys())
    values = list(data.values())
   
    _, ax = plt.subplots()
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Articles per day')
    ax.set_title('Articles per day before \n and after covid')
    ax.bar(keys, values, color ='maroon',width = 0.4)
    
    
    plt.savefig(str(name)+"_average_articles_per_day_before_after_covid.png", bbox_inches='tight')
    
    
    
if __name__=="__main__":
    get_sentiments(["/Users/felixquinque/Documents/Programming/Work_Code/Sentiment Analysis/news-sentiments/GerVADER/BILD_SCRAPING_BEREINIGT_12.07.2022.xlsx"])
    
       
