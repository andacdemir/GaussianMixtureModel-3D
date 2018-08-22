import os
from pathlib import Path 
from glob import glob
import pandas as pd
import lasio

'''
    Reads the log names and log paths from the current working directory
'''
def read_lognames(cwd):
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

'''
    Reads the features:
    Depth, Slowness, Bulk Density, Gamma Ray, Neutron Porosity and 
    Deep Resistivity
    from the data in las files
    If any of these features is equal to -999.2500, that means data is
    undefined for that depth, hence all the measurements in that depth
    are ignored.
'''
def read_features(path):
    # cd to LOGS folder from the logpath:
    if "LOGS" in os.listdir(path):
        path = os.path.join(path, "LOGS")

    # reads the las file from the path into a dataframe.
    # There is only one dir in the LOGS directory.
    # The others were deleted manually.
    # This needed to be done by a human to validate las file is well structured
    # and meets the criteria of the project in various aspects. 
    for dir in os.listdir(path):
        laspath = os.path.join(path, dir)
        las = lasio.read(laspath)
        df = las.df()
    
    # Removes the columns other than Depth, Slowness, Bulk Density, Gamma Ray, 
    # Neutron Porosity and Deep Resistivity:

    # Removes bad data (-999.2500, explained in the function definition):

    return df


def main():
    cwd = os.getcwd()
    lognames, logpaths = read_lognames(cwd)
    print(lognames[0], logpaths[0])
    read_features(logpaths[0])


if __name__ == '__main__':
    main()
