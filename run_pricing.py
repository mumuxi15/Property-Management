#!/usr/bin/env python3
import os, re
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
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
		df['date'] = df['Date'].dt.strftime('%m-%d')
		# tb = pd.pivot_table(df, index='date',columns='year',values='Rate')
		# print (tb)
		# tb.to_csv('tmp.csv')
		# gp = gp.groupby(['year', 'month','day'])['Rate'].mean().reset_index()  # .unstack().T)
		# print (gp)
		# return

	def run(self):
		df = self.read_excel(path=self.cabin['spot_rates_sheet'])
		# self.plot_weekly_heatmap(df)
		# self.plot_monthly_rate(df)
		self.test(df)

def get_calendar_view(property):

	
	new_date = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq='D')
	new_date = pd.DataFrame({'Date': new_date})
	new_date['week'] = new_date['Date'].dt.isocalendar().week
	new_date['dow']=new_date['Date'].dt.dayofweek
#	df = pd.merge(df, new_date, on=['week', 'dow']) after prices are done 
	

	print (df)
		
def bnb_data_analysis(property=None,loc='Nashville'):
	directory = property['path'].replace('transactions/honey',f'pricing/insidebnb/{loc}/09-2022/listings.csv')
	df = pd.read_csv(directory, usecols=['name', 'host_id',
		'neighbourhood', 'latitude', 'longitude', 'room_type', 'price',
		'minimum_nights', 'number_of_reviews', 'last_review',
		'reviews_per_month', 'calculated_host_listings_count',
		'availability_365', 'number_of_reviews_ltm'])
	df = df.loc[df['number_of_reviews']>0]
	print (df['room_type'].unique())



		
# bnb_data_analysis(property=hh)

SK = PriceAlgo(cabin = cabins['sky'])
SK.run()
# holidays()