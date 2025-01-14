#!/usr/bin/env python3
import os, re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import datetime
from dateutil.relativedelta import relativedelta
from config import cabins

year = 2025

# def plot_calendar2(df):
# 	import plotly.express as px


def plot_calendar(x,y,z,labels):
	colorscale = [[False, '# eeeeee'], [True, '#76cf63']]
	data = [
		go.Heatmap(
			x=x,
			y=y,
			z=z,
			text=labels,
			hoverinfo="z",
			xgap=3,  # this
			ygap=3,  # and this is used to make the grid-like apperance
			showscale=False,
			colorscale=colorscale
		)
	]
	layout = go.Layout(
		title="booking",
		height=800,
		yaxis=dict(
			showline=False, showgrid=False, zeroline=False,
			tickmode="array",
			ticktext=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
			tickvals=[0, 1, 2, 3, 4, 5, 6],
		),
		xaxis=dict(
			showline=False, showgrid=False, zeroline=False,
		),
		plot_bgcolor=('#fff'),
		margin=dict(t=40))

	fig = go.Figure(data=data, layout=layout)

	fig.show()
	return fig

class PriceAlgo:
	def __init__(self, cabin):
		self.cabin = cabin
		return
	def read_excel(self):
		today = datetime.datetime.today()
		df = pd.read_excel(self.cabin['spot_rates_sheet'], usecols=['Date', 'Rate', 'Min Nights'])
		df['Date'] = pd.to_datetime(df['Date'])
		df['week'] = df['Date'].dt.isocalendar().week
		df['dow'] = df['Date'].dt.dayofweek
		df['year'] = df['Date'].dt.year
		df['day'] = df['Date'].dt.day
		df['labels'] =  df['Date'] - df['Date'].dt.weekday * np.timedelta64(1, 'D') # first day of the week
		tb = pd.pivot_table(df, index='dow',columns='labels',values='Rate')
		tb = tb[tb.columns[100:]]
		print (tb)
		fig = px.imshow(tb, color_continuous_scale='sunset')
		fig.show()
		return

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
SK.read_excel()
# holidays()