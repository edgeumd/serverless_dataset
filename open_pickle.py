import pickle


def load_pickle(filename):
    """
    Load a pickle file..
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)


def filter_files(file, filter_list=["template", "example", "demo", "test", "/github.com/serverless/", "/github.com/serverless-components/", "boilerplate"]):
    """
    Filter a list of files.

    Returns: False if the file needs to be discarded otherwise returns True.
    """
    for word in filter_list:
        if word in file:
            return False
    return True


file = 'dataset/repos1000.pkl'

list_of_files = load_pickle(file)


newfile = 'dataset/repos1000_filtered.pkl'

nl = []
for file in list_of_files:
    if filter_files(file[3]):
        nl.append(file)

with open(newfile, 'wb') as f:
    pickle.dump(nl, f)

print(nl)
