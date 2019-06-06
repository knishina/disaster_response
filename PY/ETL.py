# import libraries
import pandas as pd
from sqlalchemy import create_engine

def merge_data():
    """
    Input: None
    Output: merged dataframe
    Tasks Performed:
        - read in csv for messages and categories.
        - merge the two dataframes on "id"
    """
    # load messages dataset
    messages = pd.read_csv("../Resources/messages.csv")
    categories = pd.read_csv("../Resources/categories.csv")

    # merge the datasets
    df = messages.merge(categories, on=["id"])

    return df


def category_columns(df):
    """
    Input: dataframe from merged_data
    Output: create dummies
    Tasks Performed:
        - split on ";" and assign to new columns.
        - take the text from first row, remove the non-alpha components.  Assign to columns.
        - convert the column values to 0 and 1s (remove alpha components)
        - assign column names
        - concat the new columns to the original dataframe.
        - drop duplicates
    """
    # split on ";", create 36 columns.
    categories = df["categories"].str.split(";", expand=True)

    # select the first row of the categories dataframe.
    # create new column names by removing the end number.
    row = list(categories.iloc[0])
    category_colnames = [row[x][:-2] for x in range(len(row))]

    # rename the columns of `categories`
    categories.columns = category_colnames

    # convert column values to 0 and 1's.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = [categories[column][x][-1:] for x in range(len(categories[column]))]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df = df.drop(columns=["categories"])

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df.drop_duplicates(subset=["message"], keep="first", inplace=True)

    return df


def ETL():
    df = merge_data()
    df = category_columns(df)
    engine = create_engine('sqlite:///DisasterResponse.db')
    df.to_sql('disaster_response', engine, index=False)
