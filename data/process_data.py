import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    This function loads the message and category dataset and merge them using inner join
    
    Input: File path of the message and category dataset
    Output: Loaded dataframe
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how="inner", on='id')
    return df

def clean_data(df):
    """
    This function clean the input dataset by applying text manipulation and dropping duplicates
    
    Input: Dataframe of interest
    Output: Cleaned dataframe
    """
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(";", expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0, :]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = row.map(lambda x: x[:-2])

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].map(lambda x: x[-1])

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df.drop(columns=["categories"], inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1, sort=False)

    # drop duplicates
    df = df[~(df.duplicated())]
    
    #Turn related value of 2 to 1 (I don't know what 2 is for?)
    df.loc[df['related'] == 2,'related'] = 1

    return df


def save_data(df, database_filename):
    """
    This function save the dataframe into designated SQLite database
    
    Input: Dataframe and database name
    Output: No output
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql("Disaster_Response_Table", engine, index=False, if_exists = 'replace')


def main():
    """
    This main function include the data processing pipeline
    
    Input: No input
    Output: No output
    """
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