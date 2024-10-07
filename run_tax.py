#!/usr/bin/env python3
from tabula import read_pdf
import pandas as pd
import datetime
import calendar
import glob
import os, re
import numpy as np
from config import smoke

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 400)


def search_string(s, search):
	return search in str(s)

def assign_type(row):
	if 	row['Amount']>0:
		return 'income'
	else:
		if 'spectrum|verizon' in row['Description']:
			return 'utility'
		elif 'tjx rewards' in row['Description']:
			return 'supplies'
		elif 'sevier county el bank' in row['Description']:
			return 'electricity'
		elif 'ownerrez' in row['Description']:
			return 'subscription'

		elif 'airbnb' in row['Description']:
			return 'refund'
	# elif 'version' in row['Description']:
	# 	return


def calculate_mortgage(principal, annual_interest_rate, years, start_date,tax_year):
	num_payments = years * 12
	monthly_interest_rate = annual_interest_rate / 12 / 100
	
	# Calculate the monthly payment using the mortgage formula
	monthly_payment = (principal * monthly_interest_rate * (1 + monthly_interest_rate) ** num_payments) / \
						((1 + monthly_interest_rate) ** num_payments - 1)
	
	# Create a DataFrame to hold the amortization schedule
	payment_dates = pd.date_range(start=start_date, periods=num_payments, freq='ME')
	df = pd.DataFrame(index=payment_dates, columns=['Principal Payment', 'Interest Payment', 'Remaining Principal'])
	
	remaining_principal = principal
	for date in payment_dates:
		interest_payment = remaining_principal * monthly_interest_rate
		principal_payment = monthly_payment - interest_payment
		remaining_principal -= principal_payment
		df.loc[date] = [principal_payment, round(interest_payment,2), remaining_principal]
	df = df.loc[df.index.year == tax_year]
	return df




class Report():
	def __init__(self, property, taxyear):
		self.taxyear = taxyear
		for k,v in property.items():
			setattr(self,k,v)
		return

	def calculate_epm_fees(self):
		folder = f"{self.path}/eagle"
		area = [1.4, .35, 11, 8]  # specify the area in points (from top-left)
		print(np.array(area) * 72)

		tables = []
		for p in [f"{folder}/{f}" for f in os.listdir(folder) if f.endswith(('.pdf'))]:
			df = read_pdf(p,output_format="dataframe",pages='all', multiple_tables=True,area=[105, 25, 800, 576])[0]
			### select the table
			df.columns.values[0] = 'Res'

			u = df.loc[df['Res'].str.contains("Owner Charges/Expenses")].index[0]
			v = df.loc[df["Res"]=="TOTAL"].index[1]
			df = df.iloc[u+1:v]
			df = df.dropna(how="all", axis=1)
			if len(df.columns) == 1:
				df['Amount'] = df['Res'].str.replace(',', '').str.extract(r'\(\$([0-9]+(?:\.[0-9]{2})?)\)')
			else:
				df = df.rename(columns={'Unnamed: 6':'Amount'})
				df['Amount'] = df['Amount'].str.replace(',', '').str.extract(r'\(\$([0-9]+(?:\.[0-9]{2})?)\)')

			df['Date'] = pd.to_datetime(df['Res'].str[0:10], format='%m/%d/%Y', errors='coerce')
			df['Date'] = df['Date'].fillna(df['Res'].str.extract(r'(\d{2}/\d{2}/\d{4})')[0])

			tables.append(df)

		df = pd.concat(tables)
		df = df[df['Amount'].notna()]
		df['Amount'] = pd.to_numeric(df['Amount'])
		df.loc[df['Res'].str.contains('Cleaning Fee|Linen Charges'),'Type'] = 'C'
		df.loc[df['Res'].str.contains('Maintenance'),'Type'] = 'M'
		# print (df)
		return  df[['Date','Type','Amount']]

	def parse_checking(self):
		' description = chase credit crd autopay, or type=loan_pmt needs to be handled seperately'
		df = pd.read_csv(self.path+'/'+[p for p in os.listdir(self.path) if self.checking in p][0], index_col=False)
		df = df.rename(columns={'Posting Date':'Date'})
		date_column = [x for x in df.columns if 'date' in x.lower()]
		df[date_column] =df[date_column].apply(pd.to_datetime)
		df = df.map(lambda x: x.lower() if isinstance(x, str) else x) #all to lower case letters
		# discard account transfer

		df = df.loc[~df['Type'].isin(['acct_xfer','wire_outgoing'])]
		groups = {'spectrum': 'wifi', 'verizon':'wifi_2','tjx rewards': 'supplies', 'sevier county el bank': 'electricity', 'chase credit crd autopay':'credit card',
			'con ed of ny': 'electricity_2','jpmorgan chase':'mortgage','tn tap':'tax', ' govt pmts':'tax','airbnb':'refund','stripe':'refund','booking':'refund'}
		events = {'check 99':'M','tilebar':'improvements','swissmadiso':'improvements','1026494149218':'C','check 151':'tax', '1026465309282':'travel','1024784469148':'M'}
		df.loc[df['Amount']>0, 'Category'] = 'income'
		df.loc[df['Type']=='loan_pmt', 'Category'] = 'credit card'

		df.loc[df['Category'].isna(), 'Category'] = df.loc[df['Category'].isna(), 'Description'].str.extract('('+'|'.join(groups.keys())+')', expand=False).map(groups)
		df.loc[df['Category'].isna(), 'Category'] = df.loc[df['Category'].isna(), 'Description'].str.extract('('+'|'.join(events.keys())+')', expand=False).map(events)
		df['Month'] = df['Date'].dt.month
		df = df.loc[df['Category']!='mortgage']
		print (df.loc[df['Category']=='tax',['Date','Description','Amount']].sort_values(by='Date',ascending=True))
		df = df.groupby(by=['Month', 'Category']).agg({'Amount': 'sum'}).unstack()
		# print (df)
		# df.to_csv('checking account.csv')
		return

	def parse_credit(self):
		df = pd.read_csv(self.path+'/'+[p for p in os.listdir(self.path) if self.credit_card in p][0], index_col=False)
		df = df.rename(columns={'Transaction Date':'Date','Category':'-'})
		df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
		# print (df.head())
		# print (df[df['Type']=='payment'])
		df = df.loc[df['Type']!='payment']
		df['Category'] = None
		groups = {'ownerrez':'subscription', 'bklynchldrnmuseum': 'charity', 'eagle': 'eagle', 'tjx rewards': 'supplies',
				  'walgreens': 'supplies', 't j maxx': 'supplies', 'target':'supplies', 'zogics':'supplies', 'mta':'travel',
				  'spectrum': 'wifi','turbotax': 'legal', 'arrow exterminators': 'pest','pirate ship':'adv','mint':'phone','homeaway':'comission'
				  }

		df.loc[df['Category'].isna(), 'Category'] = df.loc[df['Category'].isna(), 'Description'].str.extract('('+'|'.join(groups.keys())+')', expand=False).map(groups)
		sv = df.loc[df['Category']=='travel',['Date','Description','Amount']].sort_values(by='Date',ascending=True)
		print (df.loc[df['Category'].isna()])
		# sv.to_csv('tmp.csv')
		
	def parse_amzn(self):
		df = pd.read_csv(self.path+'/'+[p for p in os.listdir(self.path) if self.amzn_card in p][0], index_col=False)
		
		
		

	def run(self):
		# df = self.calculate_epm_fees()
		# df['Month'] = df['Date'].dt.month
		# df = df.groupby(by=['Month','Type']).agg({'Amount':'sum'}).unstack()
		# print (df)
		# self.parse_checking()
		self.parse_credit()
		# d = calculate_mortgage(self.mortgage['loan_amt'],self.mortgage['rate'], self.mortgage['years'], self.mortgage['start_date'], self.taxyear )
		# print (d)
		# d.to_csv('mortgage_interest.csv')
	
	
ER=Report(property=smoke,taxyear=2023)
# ER.data_cleaning()
ER.run()

