import glob
import pickle
import os
from cleaning.filter_files import SlsCleaner
import csv

def aggr(path="dataset/serverless_framework"):
    """
    Aggregates pickle files in the given path.
    """
    data = []
    for i in glob.glob(os.path.join(path,"*.pkl")):
        try:
            with open(i, "rb") as f:
                content = pickle.load(f)
            data = data + content
        except:
            print(i)
            # print(content)
            continue

    data = list(set(data))
    sls = SlsCleaner([data[i][0] for i in range(len(data))])

    return sls.processed_sls_list

def save_file(data, path="aws_aggregate.csv"):
    """
    Saves the data to the given path.
    """
    with open(path, "w") as f:
        f.write("Project_URLs\n")
        for i in data:
            f.write(i+"\n")



    
if __name__ == "__main__":
    # data = aggr(path="dataset/azure_functions")
    data = aggr(path="dataset/ibm_by_manifest_2")
    save_file(data, path="ibm_aggregate.csv")