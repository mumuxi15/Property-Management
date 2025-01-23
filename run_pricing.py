#!/usr/bin/env python3
import os, re
import json
import requests
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
from dateutil.relativedelta import relativedelta
from datetime import datetime
from bs4 import BeautifulSoup

from multiprocessing import Pool, freeze_support

from config import cabins





class PriceAlgo:
    def __init__(self, cabin, period):
        self.cabin = cabin
        self.day_of_week = {0: 'Sun', 1: 'Mon', 2: 'Tue', 3: 'Wed', 4: 'Thu', 5: 'Fri', 6: 'Sat'}
        self.month_in_text = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep',
                              10: 'Oct', 11: 'Nov', 12: 'Dec'}
        self.today = datetime.today()
        self.period = period
        return

    def read_excel(self, path):
        df = pd.read_excel(path, usecols=['Date','Rate','Min Nights'])
        df['Date'] = pd.to_datetime(df['Date'])
        date_range = pd.date_range(start=df['Date'].min(), end=datetime(self.today.year, 12, 31))
        df = df.set_index('Date').reindex(date_range)
        df['Date'] = df.index
        df['year'] = df['Date'].dt.year
        df['day'] = df['Date'].dt.day
        df['month'] = df['Date'].dt.month
        df['week'] = df['Date'].dt.isocalendar().week
        df['dow'] = df['Date'].dt.dayofweek
        return df

    def plot_weekly_heatmap(self, df):
        df['time'] = df['Date'] - df['Date'].dt.weekday * np.timedelta64(1, 'D')  # first day of the week
        tb = pd.pivot_table(df, index='dow', columns='time', values='Rate').T[1::]
        tb = tb.rename(columns=self.day_of_week)
        tb['year'] = tb.index.year
        years = tb['year'].unique()[1:-1]

        fig = make_subplots(rows=len(years), cols=1)
        for i, yr in enumerate(years):
            data = tb.loc[tb['year'] == yr][list(self.day_of_week.values())].T
            heatmap = px.imshow(data, color_continuous_scale="sunset")
            fig.add_trace(heatmap.data[0], row=i + 1, col=1)
        fig.update_layout(title="Price Per Night Distribution over the years ",
                          coloraxis=dict(colorscale='sunset'),
                          coloraxis_colorbar=dict(title="Price Per Night $", title_side="right")
                          )
        fig.show()

    def plot_monthly_rate(self, df):  # stacked bar graph
        gp = df.groupby(['year', 'month'])['Rate'].mean().reset_index()  #.unstack().T)
        gp = gp.loc[(gp['year'] > gp['year'].min()) & (gp['year'] < gp['year'].max())]
        gp['month'] = gp['month'].map(self.month_in_text)
        fig = px.bar(gp, x="month", y="Rate", color="year", title="Averaged Monthly Rate Per Night of 2022-2024")
        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Rate Per Night",
        )
        fig.show()
        return

    def yearly_dow_prices(self, df):
        df['year'] = df['Date'].dt.isocalendar().year
        df['date'] = 'w' + df['Date'].dt.isocalendar().week.astype(str) + '-d' + df['dow'].astype(str)
        df_prices = pd.pivot_table(df, index=['week', 'dow'], columns='year', values='Rate')
        return df_prices

    def get_neighbor_by_date(self, dates):
        """ search condition : 4 adults, location, entire home, hot tub, free parking"""
        checkin, checkout = dates
        response = requests.get(self.cabin['airbnb_search_url'].format(checkin=checkin,checkout=checkout))
        soup = BeautifulSoup(response.content, "html.parser")
        price = []
        script_tag = soup.find("script", attrs={"data-injector-instances": "true"})
        if script_tag:
            # Extract the content of the script tag
            data = json.loads(script_tag.string)['root > core-guest-spa'][1]
            for d in data:
                if isinstance(d, dict):
                    clientdata = d['niobeMinimalClientData']
                    for i in str(clientdata).split("'"):
                        if 'per night' in i:
                            price.append([i])
        else:
            raise Exception('Please check the website scraping format, airbnb might have changed the format')
        df = pd.DataFrame(price, columns=['txt'])
        df = df['txt'].str.split(',', expand=True)
        df.columns = ['neighbor_rate', 'neighbor_udp']  # neighbor un-discount price
        df['neighbor_rate'] = df['neighbor_rate'].str.extract(r'(\d+)')
        df['neighbor_udp'] = df['neighbor_udp'].str.extract(r'(\d+)')  #
        df['neighbor_udp'] = df['neighbor_udp'].fillna(df['neighbor_rate'])
        df[['neighbor_rate', 'neighbor_udp']] = df[['neighbor_rate', 'neighbor_udp']].apply(pd.to_numeric,
                                                                                            errors='coerce')
        df['discount'] = (df['neighbor_rate'] / df['neighbor_udp']).round(2)
        rs = df.mean(axis=0).round(2)
        rs['std'] = df['neighbor_rate'].std().round(2)
        rs['checkin'] = checkin
        return rs
    def airbnb_scrape(self):
        date_range = pd.date_range(start=self.today, end=datetime((self.today + relativedelta(months=3)).year, 12, 31))
        df = pd.DataFrame(date_range, columns=['date'])
        df['Date'] = df['date'].dt.date
        df['week'] = df['date'].dt.isocalendar().week
        df['dow'] = df['date'].dt.dayofweek
        df['checkin'] = df['Date'].astype(str)
        df['checkout'] = df['Date'].shift(-2).astype(str)
        df = df.head(self.period)
        args = df[['checkin','checkout']].values.tolist()
        with Pool(processes=5) as pool:
            rs= pool.map(self.get_neighbor_by_date, args)
        rs = pd.concat(rs, axis=1).T
        df = df.merge(rs, on="checkin")
        df = df.drop(columns=['date'])
        df.to_csv(f'data/scrapy/neighbor_prices_{self.today.date()}.csv')
        # return df

    def add_market_factor(self, df, pull_newdata=False):  # add airbnb data
        "df: self.yearly_dow_prices "
        #
        if pull_newdata:
            dp = self.airbnb_scrape()  # scrape
        dp = pd.read_csv('data/scrapy/' + sorted(os.listdir('data/scrapy/'))[-1], index_col=['week', 'dow'])
        df = pd.concat([dp, df], axis=1, join='inner')
        years = df.columns[len(dp.columns)::]
        df['suggested'] = (df['neighbor_rate'] + self.cabin['z-score'] * df['std'])  # 1std
        df.loc[:7,'suggested'] *= 0.95   # discount the first week
        df = df.reset_index()
        print (df.loc[(df['week']<=df['week'].min()+1)& (df['dow']<4)] )
        df.loc[(df['week']<=df['week'].min()+1)& (df['dow']<4),'suggested'] *= 0.9 # discount on weekdays of first week
        df['suggested'] = df['suggested'].round(0)
        ownerez = df[['checkin', 'suggested']].copy()
        ownerez['OwnerRez Property'] = self.cabin['id']
        ownerez.to_csv('export_rates.csv')

        df.to_csv('tmp.csv')

    def run(self):
        df = self.read_excel(path=self.cabin['spot_rates_sheet'])
        # print (df)
        # self.plot_weekly_heatmap(df)
        # self.plot_monthly_rate(df)
        df_prices = self.yearly_dow_prices(df)
        self.add_market_factor(df_prices, pull_newdata=False)


def data_analysis(property=None, loc='Nashville'):
    df = pd.read_csv('data/nashville_listing.csv', usecols=['id', 'listing_url', 'scrape_id', 'last_scraped',
                                                            'description', 'neighborhood_overview', 'host_id',
                                                            'host_since', 'host_location',
                                                            'host_response_time', 'host_acceptance_rate',
                                                            'host_is_superhost', 'host_neighbourhood',
                                                            'host_listings_count', 'neighbourhood_cleansed', 'latitude',
                                                            'longitude', 'room_type', 'accommodates', 'bathrooms',
                                                            'bathrooms_text', 'bedrooms', 'beds', 'amenities', 'price',
                                                            'minimum_nights', 'maximum_nights',
                                                            'minimum_nights_avg_ntm',
                                                            'maximum_nights_avg_ntm', 'has_availability',
                                                            'availability_30', 'availability_60', 'availability_90',
                                                            'availability_365', 'number_of_reviews',
                                                            'number_of_reviews_ltm', 'number_of_reviews_l30d',
                                                            'last_review', 'review_scores_rating', 'instant_bookable',
                                                            'reviews_per_month'])

    print('length: ', len(df))
    df = df.loc[(df['reviews_per_month'] > (1 / 12)) & (df['review_scores_rating'] > 4)]  #at least 1 reviews in a year
    df = df.loc[df['room_type'] == 'Entire home/apt']
    df = df.loc[df['last_review'] > '2023-01-01']
    df.to_csv('filtered_df.csv')
    print('length: ', len(df))



# bnb_data_analysis(property=hh)
if __name__ == '__main__':
    SK = PriceAlgo(cabin=cabins['sky'], period=120)
    SK.run()
