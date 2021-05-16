import pandas

excel_data_df = pandas.read_excel('name.xlsx', sheet_name='SBER')

# print whole sheet data
print(excel_data_df.groupby('result').count())
