def get_repositories(path = "./collected_resources/dataset1.csv"):
    """
    Returns list of repository in toy dataset

    :params:
        path : path of toy dataset file.

    :returns:
        list of repository in toy dataset
    """
    import pandas as pd

    res = pd.read_csv(path)
    return res['repo_name'].tolist()

def get_purpose_for_repo(repo, path = "./collected_resources/dataset1.csv"):
    """
    Returns purpose of repository
    
    :params:
        repo : name of repository
        path : path of toy dataset file.

    :returns:
        purpose of repository
    """
    import pandas as pd

    res = pd.read_csv("./collected_resources/dataset1.csv")
    try:
        return res[res['repo_name'] == repo]['purpose'].tolist()[0]
    except:
        return "unknown"

    