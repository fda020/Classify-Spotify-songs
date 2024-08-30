import pandas as pd

#Display for output in terminal when running different dataframes
def dataframe_display():
    pd.set_option('display.max_colwidth', 20)  
    pd.set_option('display.width', 200)  
    pd.set_option('display.max_columns', 14)  