import sys
import pandas as pd

from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''Read the data from the csv files provided. One file contains messages and another file contains categories. 
    The data is merged and parse into a data frame. Return the merged dataframe.'''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    print('Data loaded \n Messages shape: {} \n Categories shape: {}'.format(messages.shape, categories.shape))
    return df

def clean_data(df):
    ''' The function parses the categories column into a 36 separate columns. 
    It also adds the column that later will be used for statistcs.
    Return the parsed and cleaned dataframe.'''
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    category_colnames = [x[0] for x in row.str.split('-')]
    categories.columns = category_colnames
    
    #Iterate through the category columns in df to keep only the last character of each string (the 1 or 0). 
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = [x[len(x)-1] for x in categories[column].astype(str).str.split('-')]    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    # Drop original category column to replace with the expanded version
    df.drop('categories', axis=1, inplace=True)
    # check for rows w/o any categories
    categories['cat_num'] = categories.sum(axis=1)
    # replace 2 with 0 and drop the category with only one instance
    categories['related'] = categories['related'].apply(lambda x: int(str(x).replace('2','0')))
    df = pd.merge(df, categories, left_on=df.index, right_on=categories.index)
    df.drop('key_0', axis=1, inplace=True)
    # drop duplicates
    df.drop_duplicates(inplace=True)
    print('Cleaned data ', df.shape)
    return df

def save_data(df, database_filename):
    '''The functions saves the dataframe provided into a sqlite database witht the specified name. The table name is Messages'''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('Messages', engine, if_exists='replace', index=False)  
    return

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()