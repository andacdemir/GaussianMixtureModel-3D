import os
from pathlib import Path 
from glob import glob
import pandas as pd

'''
    Reads the log names and log paths from the current working directory
'''
def read__logname(cwd):
    try:
        filepath = os.path.join(Path(cwd), "B15_DATA")
        if not os.path.exists(filepath):
            raise Exception
        else:
            logfiles = glob(filepath + "\*")
    except Exception:
        print("Directory path %s does not exist.\n" %filepath)
        exit(1) # stop execution
    
    # parses the well id name from the path and appends to the lognames
    # logpaths stores the path for each log. 
    lognames, logpaths = [], []
    for id in logfiles:
        logpaths.append(id)
        id = id.split("B15_DATA\\",1)[1]
        lognames.append(id)
    
    return lognames, logpaths

'''
    Writes the lognames in each row of an excel file: LogCoordinates.xlsx
    Latitude and Longitude corresponding to logs were entered manually
    since there is no convention pursued in naming the dataset.
    Unfortunately latitude and longitude columns of this excel file had to
    be filled manually.
'''
def write_lognames(cwd, lognames):
    # check if it does not exist
    filepath = os.path.join(Path(cwd), "LogCoordinates.xlsx")
    if os.path.exists(filepath):
        pass
    else:
        columns = ["Latitude", "Longitude"]
        df = pd.DataFrame(index=lognames, columns=columns)
        writer = pd.ExcelWriter('LogCoordinates.xlsx')
        df.to_excel(writer)
        writer.save()
        
def main():
    cwd = os.getcwd()
    lognames, logpaths = read__logname(cwd)
    print(len(lognames))
    write_lognames(cwd, lognames)


if __name__ == '__main__':
    main()
