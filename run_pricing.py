#!/usr/bin/env python3
import os, re
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import requests
import json
from bs4 import BeautifulSoup
from config import cabins

class PriceAlgo:
	def __init__(self, cabin):
		self.cabin = cabin
		self.day_of_week = {0:'Sun', 1:'Mon', 2:'Tue',3:'Wed',4:'Thu',5:'Fri',6:'Sat'}
		self.month_in_text = { 1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec' }
		return
	def read_excel(self, path):
		today = datetime.today()
		df = pd.read_excel(path, usecols=['Date', 'Rate', 'Min Nights'])
		df['Date'] = pd.to_datetime(df['Date'])
		date_range = pd.date_range(start=df['Date'].min(), end=datetime(df['Date'].max().year, 12, 31))
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
		tb = pd.pivot_table(df, index='dow',columns='time',values='Rate').T[1::]
		tb = tb.rename(columns=self.day_of_week)
		tb['year'] = tb.index.year

		years = tb['year'].unique()[1:-1]

		fig = make_subplots(rows=len(years), cols=1)
		for i, yr in enumerate(years):
			data = tb.loc[tb['year']==yr][list(self.day_of_week.values())].T
			heatmap = px.imshow(data, color_continuous_scale="sunset")
			fig.add_trace(heatmap.data[0], row=i+1, col=1)
		fig.update_layout(title="Price Per Night Distribution over the years ",
						  coloraxis=dict(colorscale='sunset'),
						  coloraxis_colorbar=dict(title="Price Per Night $", title_side="right")
						  )
		# Show the figure
		fig.show()
	def plot_monthly_rate(self,df):
		gp = df.groupby(['year','month'])['Rate'].mean().reset_index()    #.unstack().T)
		gp = gp.loc[(gp['year']>2021)&(gp['year']<2025)]
		gp['month'] = gp['month'].map(self.month_in_text)

		fig = px.bar(gp, x="month", y="Rate", color="year", title="Averaged Monthly Rate Per Night of 2022-2024")
		fig.update_layout(
			xaxis_title="Month",
			yaxis_title="Rate Per Night",
		)
		fig.show()
		return

	def test(self, df):
		# df = df.dropna(how='any',axis=0)
		df['year'] = df['Date'].dt.isocalendar().year
		df['date'] = 'w' + df['Date'].dt.isocalendar().week.astype(str)+'-d' + df['dow'].astype(str)
		df.to_csv('dates.csv')
		tb = pd.pivot_table(df, index='date',columns='year',values='Rate')
		print (tb)
		tb.to_csv('tmp.csv')
		# tb.to_csv('tmp.csv')
		# gp = gp.groupby(['year', 'month','day'])['Rate'].mean().reset_index()  # .unstack().T)
		# print (gp)
		# return

	def run(self):
		df = self.read_excel(path=self.cabin['spot_rates_sheet'])
		# self.plot_weekly_heatmap(df)
		# self.plot_monthly_rate(df)
		self.test(df)
		
def data_analysis(property=None,loc='Nashville'):
	df = pd.read_csv('data/nashville_listing.csv', usecols= ['id', 'listing_url', 'scrape_id', 'last_scraped',
       'description', 'neighborhood_overview', 'host_id', 'host_since', 'host_location',
       'host_response_time', 'host_acceptance_rate',
       'host_is_superhost', 'host_neighbourhood', 'host_listings_count', 'neighbourhood_cleansed', 'latitude',
       'longitude', 'room_type', 'accommodates', 'bathrooms',
       'bathrooms_text', 'bedrooms', 'beds', 'amenities', 'price',
       'minimum_nights', 'maximum_nights', 'minimum_nights_avg_ntm',
       'maximum_nights_avg_ntm', 'has_availability',
       'availability_30', 'availability_60', 'availability_90',
       'availability_365', 'number_of_reviews',
       'number_of_reviews_ltm', 'number_of_reviews_l30d',
       'last_review', 'review_scores_rating', 'instant_bookable', 'reviews_per_month'])

	print ('length: ',len(df))
	df = df.loc[(df['reviews_per_month']>(1/12)) & (df['review_scores_rating']>4)]  #at least 1 reviews in a year
	df = df.loc[df['room_type']=='Entire home/apt']
	df = df.loc[df['last_review']>'2023-01-01']
	df.to_csv('filtered_df.csv')
	print ('length: ',len(df))
	# print (sorted(df['last_review'].unique()))

def airbnb_scrape():
	url = "https://www.airbnb.com/s/Gatlinburg--Tennessee--United-States/homes?refinement_paths%5B%5D=%2Fhomes&flexible_trip_lengths%5B%5D=one_week&monthly_start_date=2025-02-01&monthly_length=3&monthly_end_date=2025-05-01&price_filter_input_type=0&channel=EXPLORE&query=Gatlinburg%2C%20TN&place_id=ChIJiaUIy-pTWYgRqHm3fq7XsUo&location_bb=Qg8HaMKm2x9CDsoewqci9w%3D%3D&date_picker_type=calendar&checkin=2025-02-01&checkout=2025-02-04&adults=2&source=structured_search_input_header&search_type=autocomplete_click"

	response = requests.get(url)
	soup = BeautifulSoup(response.content, "html.parser")
	script_tag = soup.find("script", attrs={"data-injector-instances": "true"})
	if script_tag:
		# Extract the content of the script tag
		data = json.loads(script_tag.string)['root > core-guest-spa'][1]
		for d in data:
			if isinstance(d, dict):
				clientdata = d['niobeMinimalClientData']
				print (len(clientdata[1]))

				print (type(clientdata[1]))

# print (scripts)
	# for i, script in enumerate(soup.find_all('script')):
	#
	# 	print (script)
	# 	print (i, '---'*10)
	# script_tag = soup.find("script", {"id": "data-deferred-state-0"})
	# print (soup)
	# if script_tag:
	# 	script_content = script_tag.string
	# 	print(script_content)





# bnb_data_analysis(property=hh)

SK = PriceAlgo(cabin = cabins['sky'])
# SK.run()
# data_analysis()
airbnb_scrape()