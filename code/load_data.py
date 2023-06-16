import pandas as pd

def get_economic_data(company_name):
    economic_dataframe = pd.read_csv('../data/economic_data.csv')
    description_dataframe = pd.read_csv('../data/train.csv')
    company_data = economic_dataframe[economic_dataframe['Company'] == company_name]
    company_description = description_dataframe[description_dataframe['label'] == company_name]['description'].values[0]

    return company_data, company_description


print(get_economic_data('ZF'))
