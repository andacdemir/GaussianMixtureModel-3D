import os
from pathlib import Path 
from glob import glob
import pandas as pd
import lasio
from shutil import rmtree
from tabulate import tabulate

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
    Each log directory has many sub directories with many data files in it.
    In this project only LOGS and WELL_PATH relevant
    This deletes the other directories
'''
def delete_subdirectories(logpaths):
    relevant_dirs = ["LOGS", "WELL_PATH"]
    for path in logpaths:
        for dir in os.listdir(path):
            if dir not in relevant_dirs:
                rmtree(os.path.join(path, dir))

'''
    For fitting the Gaussian Mixture Model, we only need the las files 
    that have the substring "Composite" in their name.
    This deletes the las files without the substring Composite
'''
def delete_las_files(logpaths):
    substr = "COMPOSITE"
    for path in logpaths:
        # cd to LOGS folder from the logpath:
        path = os.path.join(path, "LOGS")
        # deletes the las files that don't have the substring in their name
        for dir in os.listdir(path):
            if substr not in dir:
                os.remove(os.path.join(path, dir))

'''
    Reads the features:
    Depth, Slowness, Bulk Density, Gamma Ray, Neutron Porosity and 
    Deep Resistivity
    from the data in las files
    If any of these features is equal to -999.2500, that means data is
    undefined for that depth, hence all the measurements in that depth
    are ignored.
    DEPTH -> DEPT.M
    GAMMA RAY -> GR
    NEUTRON POROSITY -> NEUT, CN, CNC, NPHI, NEU
    BULK DENSITY -> DEN, ZDEN, RHOB
'''
def read_features(path, name, coords):
    # cd to LOGS folder from the logpath:
    if "LOGS" in os.listdir(path):
        path = os.path.join(path, "LOGS")

    # reads the las file from the path into a dataframe.
    # Las files that don't have the substring "COMPOSITE" in their names were 
    # deleted from the LOGS directory using the function delete_subdirectories
    # Results of the deletion were validated by a human to make sure that the 
    # las files are well structured and meet the criteria of the project in 
    # various aspects. 
    for dir in os.listdir(path):
        laspath = os.path.join(path, dir)
        las = lasio.read(laspath)
        df = las.df()
    
    # Removes the columns other than Depth, Gamma Ray, Neutron Porosity and
    # Bulk Density:
    
    features = ['DEPT.M', 'GR', 'NEUT', 'CN', 'CNC', 'NPHI', 'NEU', 'DEN', 
                'RHOB', 'ZDEN']
    df = df.filter(features)
    # Removes bad data (-999.2500, explained in the function definition):
    df = df.dropna()
    # Renames some of the 4 columns:
    df.columns = ["Gamma_Ray", "Neutron_Porosity", "Bulk_Density"]
    # For the logs neutron porosity is not a fraction (0-1), 
    # converts them to fraction
    if name not in ["15_6-12", "15_12-19", "15_12-23", "15_6-4", "15_12-24", 
                    "15_9-24", "15_9-5", "15_9-4"]:
        df["Neutron_Porosity"] /= 100
    # Removes bad measurements, keeps the good ones:
    df = df[(df.Gamma_Ray > 0) & (df.Gamma_Ray < 300) & 
            (df.Neutron_Porosity > 0) & (df.Neutron_Porosity < 0.5) &
            (df.Bulk_Density > 0)]
    # Gets latitude and longitude of the wellbores and adds them to
    # the dataframe
    df['Latitude'] = pd.Series(coords.at[name,"Latitude"], index=df.index)
    df['Longitude'] = pd.Series(coords.at[name,"Longitude"], index=df.index)
    # Adds a new column to the end, that is the log's name (id):
    df['Log_Name'] = pd.Series(name, index=df.index)
    return df

def main():
    cwd = os.getcwd()
    lognames, logpaths = read_lognames(cwd)
    print("Total number of logs:", len(lognames))
    delete_subdirectories(logpaths)
    delete_las_files(logpaths)

    coords = pd.read_excel('LogCoordinates.xlsx')
    frames = []
    for name, path in zip(lognames, logpaths):
        print(path)
        df = read_features(path, name, coords)
        frames.append(df)
        print(tabulate(df.head(), headers='keys', tablefmt='psql'))
    
    # Concatenates all the dataframes
    df = pd.concat(frames)
    print(tabulate(df, headers='keys', tablefmt='psql'))
    # Saves the dataframe as an excel file
    writer = pd.ExcelWriter('Processed_LogData.xlsx')
    df.to_excel(writer)
    writer.save()


if __name__ == '__main__':
    main()
