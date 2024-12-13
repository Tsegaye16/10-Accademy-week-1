import pandas as pd


def LoadData(file_path):
    # Load the data from the file
    df = pd.read_csv(file_path)
    return df
def InspectData(dataFrame):
    return dataFrame.info()
   
def GetShape(df):
    return df.shape
def GetSummary(df):
    return df.describe()
def GetColumns(df):
    return df.columns
def GetColumnNames(df):
    return df.columns.tolist()
def GetColumnTypes(df):
    return df.dtypes
def CheckMissingValue(df):
    return df.isnull().sum()
def RemoveColumn(df):
    return df.drop(columns= ['Unnamed: 0'], inplace=True)
def AddHeadlineLength(df):
    df["headline_length"] = df["headline"].apply(len)
    return df
def GtHeadlineLengthStats(df):
    
    return df['headline_length'].describe()


# Convert a specified column to datetime and set it as the index
def ConvertDdate(df, column_name='Date'):        
    if column_name not in df.columns:

        raise ValueError(f"Column '{column_name}' not found in DataFrame.")
    
    # Convert the column to datetime
    df[column_name] = pd.to_datetime(df[column_name], errors='coerce').dt.date
    
    # Set this column as the index
    df.set_index(column_name, inplace=True)
    
    # Convert the index to DatetimeIndex
    df.index = pd.to_datetime(df.index)
    
    return df

########################################################################