import streamlit as st
import pandas as pd 
import numpy as np
import requests
import tweepy
import yfinance as yf
import plotly.graph_objects as go
import datetime 
import string

import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")


# tweet api auth
tweets_auth = tweepy.OAuthHandler(st.secrets["twitter_api_key"], st.secrets["twitter_api_key_secret"])
tweets_auth.set_access_token(st.secrets["twitter_access_token"], st.secrets["twitter_secret_token"])


tweets_api = tweepy.API(tweets_auth)


option = st.sidebar.selectbox("Selected Dashboard", ('Activist short sellers','Charts', 'Stocktwits'))


@st.cache
def g_mean(my_array):
    new_list = []
    for i in my_array:
        new_i = 1+i
        new_list.append(new_i)
    prod=np.prod(new_list)
    geo_mean = prod**(1/len(my_array))-1
    return geo_mean

TWITTER_USERNAMES = [

    "AlderLaneeggs",
    "Bleecker_St",
    "blueorcainvest",
    "CitronResearch",
    "CulperResearch",
    "GlassH_Research",
    "HindenburgRes",
    "IcebergResear",
    "JCap_Research",
    "KerrisdaleCap",
    "muddywatersre",
    "NMRtweet",
    "PresciencePoint",
    "QCMFunds",
    "ResearchGrizzly",
    "sharesleuth",
    "ScorpionFund",
    "viceroyresearch",
    "WhiteResearch",
    
]

if option == 'Activist short sellers':


    st.title('Activist shorts stocktwits')
    st.write(' ')
    st.write('Check the latest stocktwits from activist short sellers')
    st.write('(API v1.1)')
    st.write(' ')
    st.write(' ')
    st.write(' ')

    selected_users = st.sidebar.multiselect('short sellers selection', TWITTER_USERNAMES) 
    time_select = st.sidebar.select_slider(label='select time interval (tweets from the past n days)', options=range(1,181), value=7)
    today = datetime.datetime.today()

    symbol_list = []
    
    for tweets_username in selected_users:
        
        tweets_user = tweets_api.get_user(tweets_username)
        tweets = tweets_api.user_timeline(tweets_username, count=60, include_rts=False)
        
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        
        col1, col2, col3 = st.columns((1,3,5))
        col1.image(tweets_user.profile_image_url)
        col2.subheader(f"{tweets_user.name} stocktwits:")
        st.write(' ')
        

        for tweet in tweets: 
            if '$' in tweet.text and ((today - tweet.created_at) <= datetime.timedelta(days=time_select)):
                words = tweet.text.split(' ')
                for word in words:
                    if word.startswith('$') and word.isupper() and word[-1] in string.punctuation and word[-2] not in string.punctuation:
                        symbol = word[1:-1]
                    elif word.startswith('$') and word[1:].isalpha():
                        symbol = word[1:]
                    
                        symbol_list.append(symbol)
                        
                        st.write(' ')
                        col1, col2 = st.columns((3,2))
                        col1.write(f"### {symbol}")
                        col1.write(f"{tweet.created_at.strftime('%d %b %Y, %H:%M:%S')} UTC+0")
                        col1.write(f"{tweet.retweet_count} Retweets, {tweet.favorite_count} Likes")
                        col1.write(tweet.text)
                        col1.image(f"https://finviz.com/chart.ashx?t={symbol}")

    start_date = today - datetime.timedelta(days=time_select)
    symbols_array = np.asarray(symbol_list)
    (unique, counts) = np.unique(symbols_array, return_counts=True)
    symbols_frequency = np.asarray((unique, counts)).T
    freq_df = pd.DataFrame(symbols_frequency, columns=['ticker', 'counts'])
    st.sidebar.write(f"number of mentions ({start_date.strftime('%d %b %Y')} - {today.strftime('%d %b %Y')})")    
    st.sidebar.dataframe(freq_df)
    
    





if option == 'Charts':

    # sidebar inputs
    symbol = st.sidebar.text_input("Symbol", value='TWLO')
    select_period = st.sidebar.select_slider(label='Select period',options=['1d','5d','1mo','3mo','6mo','1y','2y','5y','10y','ytd','max'], value='1mo')
    select_interval = st.sidebar.select_slider(label='Select interval (intraday max period=60d)', options=['1m','2m','5m','15m','30m','60m','90m','1h','1d','5d','1wk','1mo','3mo'], value='1d')
    
    # DATA
    data = yf.Ticker(symbol)
    spy = yf.Ticker('SPY')

    history = pd.DataFrame(data.history(period=select_period, interval=select_interval))
    spy_data = spy.history(period=select_period, interval=select_interval)

    last_stock_price = history.Close[-1]
    first_stock_price = history.Close[0]
    max_close_price = history.Close.max()
    min_close_price = history.Close.min()
    mean_close_price = history.Close.mean()

    last_date=history.index[-1].strftime("%m/%d/%Y, %H:%M:%S")
    last_date_notime = history.index[-1].strftime("%m/%d/%Y")
    first_date=history.index[0].strftime("%m/%d/%Y")
    time_zone = data.info['exchangeTimezoneShortName']
    currency = data.info['currency']
    market_cap = data.info['marketCap']
   
    spy_price = spy_data.Close.rename('bench')
    stock_price = history.Close.rename('stock')

    # beta
    spy_price = spy_data.Close.rename('bench')
    stock_price = history.Close.rename('stock')
    df_ = pd.concat([spy_price, stock_price], axis=1)
    df_rets = df_.pct_change().dropna()
    y_beta = df_rets['stock'].values
    x_beta = df_rets['bench'].values.reshape(-1,1)
    beta_model = LinearRegression().fit(x_beta,y_beta)
    beta = beta_model.coef_[0]

    # stock rets
    returns = history.Close.pct_change()
    rets_std = returns.std()
    rets_avg = returns.mean()
    rets_geo_avg = g_mean(returns.dropna().values)
    perc_returns = returns*100
    change_52_week = data.info['52WeekChange']
    spy_change_52_week = spy.info['52WeekChange']

    # volume
    volume =  history.Volume[-1]
    volume_avg = history.Volume.mean()
    avg_volume_10d = data.info['averageVolume10days']

    # short info
    if data.info['quoteType'] == 'EQUITY':
        shares_out = data.info['sharesOutstanding']
        shares_numb_short = (data.info['sharesShort'])
        shares_short_prior_month = data.info['sharesShortPriorMonth']
        short_int_ratio = data.info['shortRatio']
        short_float_percent = data.info['sharesPercentSharesOut']
        
        # short indicators
        fig_c5 = go.Figure(go.Indicator(
            mode="number+delta",
            value=shares_numb_short,
            number={"font": {"size": 35}, "valueformat": ".3s"},
            delta={'position': "bottom", 'reference':shares_short_prior_month , 'relative': True},
            title = {"text": "total shares sold short<br><span style='font-size:0.8em;color:gray'>% change from previous month</span>", "font":{'size':18}}))
        fig_c5.update_layout(height=200, width=250)

        #float shorted
        fig_c11 = go.Figure(go.Indicator(
            mode="number",
            value= short_float_percent,
            number={"font": {"size": 35}, "valueformat": ".2%"},
            title = {"text": "percentage of float shorted", "font":{'size':18}}))
        fig_c11.update_layout(height=200, width=250)

        #short ratio
        fig_c12= go.Figure(go.Indicator(
            mode="number",
            value=short_int_ratio,
            number={"font": {"size": 35}, "valueformat": ".2f"},
            title = {"text": "short interest ratio", "font":{'size':18}}))
        fig_c12.update_layout(height=200, width=250)

         # sharesout ind.
        fig_c14 = go.Figure(go.Indicator(
            mode="number",
            value=shares_out,
            number={"font": {"size": 35}, "valueformat": ".3s"},
            title = {"text": "shares outstanding", "font":{'size':15}}))  
        fig_c14.update_layout(height=200, width=250)




    # CANDLESTICK PLOT
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Candlestick(x=history.index, 
        open=history.Open,
        high=history.High,
        low=history.Low,
        close=history.Close,
        name=symbol,
    ), secondary_y=True)

    fig.add_trace(go.Bar(x=history.index, y=history.Volume, name='volume', marker=dict(color='#9467bd')), secondary_y=False)

    fig.update_traces(showlegend=False)
    fig.update_traces(opacity=0.3, selector=dict(type='bar'))
    fig.layout.yaxis2.showgrid=False
    fig.update_xaxes(type='category')
    fig.update_layout(title=f"{symbol}", height=600)


    ## candle att
    fig_ca = go.Figure()

    # min
    fig_ca.add_trace(go.Indicator(
            mode="number",
            value=min_close_price,
            number={"font": {"size": 20}, "valueformat": ".2f"},
            title = {"text": f"min close price ({select_interval})<br><span style='font-size:0.8em;color:gray'>period: {first_date} - {last_date_notime}</span>", "font":{'size':15}},
            domain={'row': 0, 'column': 0}))

    # max
    fig_ca.add_trace(go.Indicator(
            mode="number",
            value=max_close_price,
            number={"font": {"size": 20}, "valueformat": ".2f"},
            title = {"text": f"max close price ({select_interval})<br><span style='font-size:0.8em;color:gray'>period: {first_date} - {last_date_notime}</span>", "font":{'size':15}},
            domain={'row': 1, 'column': 0}))

    # mean
    fig_ca.add_trace(go.Indicator(
            mode="number",
            value=mean_close_price,
            number={"font": {"size": 20}, "valueformat": ".2f"},
            title = {"text": f"mean close price ({select_interval})<br><span style='font-size:0.8em;color:gray'>period: {first_date} - {last_date_notime}</span>", "font":{'size':15}},
            domain={'row': 2, 'column': 0}))

    fig_ca.update_layout(grid={'rows':3, 'columns':1}, height=400, width=250)



    # last price + period ret. indicator
    fig_c1 = go.Figure(go.Indicator(
            mode="number+delta",
            value=last_stock_price,
            number={"font": {"size": 35}, "valueformat": ".2f"},
            delta={'position': "bottom", 'reference':first_stock_price, 'relative': True},
            title = {"text": f"last price ({currency}) <br><span style='font-size:0.8em;color:gray'>{last_date} {time_zone}</span><br><span style='font-size:0.8em;color:gray'> {first_date}-{last_date_notime} ret.%(unadj.)</span>", "font":{'size':18}}))
    fig_c1.update_layout(height=200, width=250)


    # beta indicator
    fig_c2 = go.Figure(go.Indicator(
            mode="number",
            value=beta,
            number={"font": {"size": 35}, "valueformat": ".2f"},
            title = {"text": f"beta {select_period} <br><span style='font-size:0.8em;color:gray'>{select_interval} rets (unadj.)</span><br><span style='font-size:0.8em;color:gray'>benchmark: S&P500</span>", "font":{'size':18}}))
    fig_c2.update_layout(height=200, width=250)


    # 52 week return indicator
    fig_c3 = go.Figure(go.Indicator(
            mode="number",
            value=change_52_week,
            number={"font": {"size": 35}, "valueformat": ".2%"},
            title = {"text":"52 week change (%)", "font":{'size':18}}))
    fig_c3.update_layout(height=200, width=250)


    # avg. volume indicator
    fig_c6 = go.Figure(go.Indicator(
            mode="number",
            value=volume_avg,
            number={"font": {"size": 35}, "valueformat": ".2s"},
            title = {"text": f"avg. {select_interval} volume<br><span style='font-size:0.8em;color:gray'>period: {first_date} - {last_date_notime}</span>", "font":{'size':18}}))       
    fig_c6.update_layout(height=200, width=250)

    # last 10days avg daily volume indicator
    fig_c8 = go.Figure(go.Indicator(
            mode="number",
            value=avg_volume_10d,
            number={"font": {"size": 35}, "valueformat": ".2s"},
            title = {"text": "avg. 10 days daily volume", "font":{'size':15}}))
    fig_c8.update_layout(height=200, width=250)

    # market cap indicator 
    fig_c9 = go.Figure(go.Indicator(
            mode="number",
            value=market_cap,
            number={"font": {"size": 35}, "valueformat": ".3s"},
            title = {"text": "current market cap", "font":{'size':18}}))      
    fig_c9.update_layout(height=200, width=250)

  


    ## rets mean std indicator
    fig_c7 = go.Figure()

    # std
    fig_c7.add_trace(go.Indicator(
            mode="number",
            value=rets_std,
            number={"font": {"size": 20}, "valueformat": ".2%"},
            title = {"text": f"{select_interval} rets std.<br><span style='font-size:0.8em;color:gray'>period: {first_date} - {last_date_notime}</span>", "font":{'size':15}},
            domain={'row': 0, 'column': 0}))

    # mean
    fig_c7.add_trace(go.Indicator(
            mode="number",
            value=rets_avg,
            number={"font": {"size": 20}, "valueformat": ".2%"},
            title = {"text": f"{select_interval} rets arithmetic mean<br><span style='font-size:0.8em;color:gray'>period: {first_date} - {last_date_notime}</span>", "font":{'size':15}},
            domain={'row': 1, 'column': 0}))

    # geo mean
    fig_c7.add_trace(go.Indicator(
            mode="number",
            value=rets_geo_avg,
            number={"font": {"size": 20}, "valueformat": ".2%"},
            title = {"text": f"{select_interval} rets geo mean<br><span style='font-size:0.8em;color:gray'>period: {first_date} - {last_date_notime}</span>", "font":{'size':15}},
            domain={'row': 2, 'column': 0}))
    
    fig_c7.update_layout(grid={'rows':3, 'columns':1}, height=400, width=250)
   


    # rets plot
    fig1 = px.line(returns, x=perc_returns.index, y=perc_returns)
    fig1.update_layout(title=f"{select_interval} price change % (not adj.)", height=400)
    fig1.layout.xaxis.showgrid=False
    fig1.update_xaxes(type='category')
    fig1.update_yaxes(title=symbol)


    # app    
    st.header(data.info['longName'])
    st.write(data.info['longBusinessSummary'])

    col1, col2, col3, col4, col5= st.columns(5)
    col1.plotly_chart(fig_c1)
    col2.plotly_chart(fig_c3)
    col3.plotly_chart(fig_c2)
    col4.plotly_chart(fig_c6)
    col5.plotly_chart(fig_c8)
   

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.plotly_chart(fig_c9)
    if data.info['quoteType'] == 'EQUITY':
        col2.plotly_chart(fig_c14)
        col3.plotly_chart(fig_c5)
        col4.plotly_chart(fig_c11)
        col5.plotly_chart(fig_c12)
    else:
        pass
        

    col1, col2 = st.columns((3,1))
    col1.plotly_chart(fig, use_container_width=True)
    col2.plotly_chart(fig_ca)
    

    col1, col2 = st.columns((3,1))
    col1.plotly_chart(fig1, use_container_width=True)
    col2.plotly_chart(fig_c7)
    

    st.write('OHLC data')
    st.write(history)

    @st.cache
    def convert_df(df):
        return df.to_csv().encode('utf-8')

    csv = convert_df(history)

    st.download_button(
        label='Download OHLC data as CSV',
        data=csv,
        file_name=f"{symbol}_OHLC.csv",
        mime='text/csv'
    )




if option == 'Stocktwits':

    st.image('images/stocktwits-LOGO.png')
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.write(' ')
    st.write(' ')

    symbol = st.sidebar.text_input(label="Symbol", value='TWLO', max_chars=5)
    
    r = requests.get(f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json")

    data = r.json()

    sentiment_list = []
    total_mess = 0

    for message in data['messages']:
        
    
        total_mess = total_mess + 1
        col1, col2, col3 = st.columns((2,4,5))
        col1.image(message['user']['avatar_url'])
        col1.write(message['created_at'])
        if message['entities']['sentiment'] is not None:
            sentiment = list(message['entities']['sentiment'].values())[0]
            sentiment_list.append(sentiment)
            col1.write(f"sentiment: {sentiment}")
        col2.write(f"#### {message['user']['username']}")
        col2.write(message['body'])
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write(' ')

        
    
    
    
    sent_array = np.asarray(sentiment_list)
    (unique, counts) = np.unique(sent_array, return_counts=True)
    counts_perc = counts / total_mess
    sent_frequency = np.asarray((unique, counts)).T
    sent_df = pd.DataFrame(sent_frequency, columns=['sentiment', 'counts'])
    sent_df['%'] = counts_perc
    sent_df['%'] = sent_df['%'].mul(100).astype(int).astype(str).add('%')
    st.sidebar.write(' ')
    st.sidebar.dataframe(sent_df)
        
    