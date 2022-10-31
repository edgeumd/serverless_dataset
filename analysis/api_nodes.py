# Get API

# What is our goal? 
# We need a graph that connects api nodes to each other.
# Make keywords for searching applications. (Try to automate... but we can have manual list as well. (1 hr?))
# Using that list, find repositories from different sources, based on that find api's and connect them together. (maintain two lists, source dependent and source independent) 
# in source dependent, directly join api's to one another.
# in source independent, extend list for all -> then join the api's together.
# we will be using AWS repositories as base to find search query's since they are more representative of the applications.
# we can also use serverless instead of AWS. (Can train a NN model to do that, but its later step)

from functools import lru_cache
import remove_empty_sarvesh as rm
import code_handler as ch
import glob
import os
import pandas as pd
import csv
from collections import Counter, defaultdict

cs = rm.CodeSpecific()
ft = rm.FileTraversal()

def get_name(repo_path, from_end=1):
    """
    Get repo name given the repo path.
    :param repo_path:
    """
    return repo_path.split("/")[-from_end]

def get_important_tags(repo="aws"):
    """
    Returns a list of search queries for important tags based on repositories.
    NOTE: Needs to be updated.

    :param repo: Repository Source

    :return: List of search queries
    """
    storage_path = "storage/oneplace/" + repo + "/apis/"
    readme_files = glob.glob(storage_path + "*/readme.md")
    data = []
    for rf in readme_files:
        with open(rf,"r") as f:
            try:
                z = f.read()
                if(z==""):
                    continue

                z = z.split(" ")
                data.append(z)
            except:
                pass

    wvt = ch.Word2VecModelTrainer(data,path="readme.wordvectors")
    kv = wvt.load_trained()
    model = ch.Word2VecModelWorker(kv)

    res = []

    for i in range(len(data)):
        tmp = model.kv.most_similar(model.parse_code_adder(data[i]))
        tmp = [i[0] for i in tmp]
        tmp = " ".join(tmp)
        res.append(tmp)

    return res





def get_api_nodes_for_code(code_path=None):
    """
    Given a code path, returns a list of API that are used in the code.

    :params:
        code_path: Path to the code
    
    :return: List of API nodes
    """
    try:
        if(code_path==None):
            return None
        
        filePath = code_path
        if(os.path.isfile(filePath)):

            cp = ch.CodePreprocessor([""])
            with open(filePath, "r") as f:
                code = f.read()

            return cp.get_imports(cp.code_lang(filePath), code, remove_extra=True)

        else:
            return None
    except:
        return None


def connect_nodes_self(nodes, connections):
    """
    Connect the given nodes in the network.
    Inplace operation.
    :params:
        nodes: a list of nodes in the network (API's)
        connections: a list of connections between nodes
    :return:
        None
    """
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            v = tuple(sorted([nodes[i], nodes[j]]))
            connections[v] += 1
    return


def connect_nodes_external(nodes1, nodes2, connections):
    """
    Connect each node from nodes1 to each nodes in nodes2
    Inplace operation.

    :params:
        nodes1: List of nodes
        nodes2: List of nodes
        connections: List of connections
    :return:
        None
    """
    for i in range(len(nodes1)):
        for j in range(len(nodes2)):
            v = tuple(sorted([nodes1[i], nodes2[j]]))
            connections[v] += 1
    return

# def search_repositories_word2vec(query, repositories, location_mapping, with_score=False, num=5, all_repos=False, default_percent=0.2, thres=200):
#     """
#     Search repositories based on the given query that are in repositories.

#     :params:
#         query: Query to search
#         repositories: List of repositories
#         location_mapping: Location mapping (repo:[location,platform])
#         with_score: If True, returns a list of tuples (repo, score)
#         num: Number of results to return
#         all_repos: If True, returns all repositories
#         default_percent: Default percentage of repositories to return
#         thres: Threshold for the number of connections

#     :return:
#         List of repositories(names) that match the query

#     """

#     keywords = query
#     platform_mapping = {i: location_mapping[i][1] for i in repositories}

#     def pathBuilder(repo, type):
#         typeConfig = {
#             "api": "all_apis.txt",
#             "tok": "{}.tok".format(repo),
#             "config": "{}.cf".format(repo),
#             "readme": "readme.md"
#         }
#         return "./storage/oneplace/" + platform_mapping[repo] + "/apis/" + repo + "/" + typeConfig[type]

#     md_mappings = {i:pathBuilder(i,"readme") for i in repositories}
#     tok_mappings = {i:pathBuilder(i,"tok") for i in repositories}
#     config_mappings = {i:pathBuilder(i,"config") for i in repositories}
#     api_mappings = {i:pathBuilder(i,"api") for i in repositories}
#     toks = [tok_mappings[i] for i in repositories]
#     configs = [config_mappings[i] for i in repositories]
#     apis = [api_mappings[i] for i in repositories]
#     mds = [md_mappings[i] for i in repositories]

#     cp = ch.CodePreprocessor(["sss"])
#     res = []



    


def search_repositories_for(query, repositories, location_mapping, with_score=False, num=5, all_repos=False, default_percent=0.2, thres=200,word_vec=False):
    """
    Search repository based on the given query that are in repositories.

    :params:
        query: Search Query
        repositories: all the repositories that we are considering.
        location_mapping: location mapping for all the repositories.
        num: Number of results to return
        with_score: Return the score of the results
        all_repos: Return all the repositories and ignore num parameter
        default_percent: If lot's of repositories are returned, trims down to a certain percentage of repositories.
        thres: Threshold for the number of connections
        word_vec: If True, uses word vectors to search
    
    
    :returns:
        List of repositories (names) that match the query
        
    """
    keywords = query
    platform_mapping = {i:location_mapping[i][1] for i in repositories}
    def pathBuilder(repo, type):
        typeConfig = {
            "api": "all_apis.txt",
            "tok": "{}.tok".format(repo),
            "config": "{}.cf".format(repo),
            "readme": "readme.md"
        }
        return "./storage/oneplace/" + platform_mapping[repo] + "/apis/" + repo + "/" + typeConfig[type]

    md_mappings = {i:pathBuilder(i,"readme") for i in repositories}
    tok_mappings = {i:pathBuilder(i,"tok") for i in repositories}
    config_mappings = {i:pathBuilder(i,"config") for i in repositories}
    api_mappings = {i:pathBuilder(i,"api") for i in repositories}
    toks = [tok_mappings[i] for i in repositories]
    configs = [config_mappings[i] for i in repositories]
    apis = [api_mappings[i] for i in repositories]
    mds = [md_mappings[i] for i in repositories]
    cp = ch.CodePreprocessor(["sss"])
    res = []
    for i in range(len(mds)):
        api_score = cp.find_similar_score(keywords, apis[i], is_main_path=False, word_vec=word_vec)
        tok_score = cp.find_similar_score(keywords, toks[i], is_main_path=False, word_vec=word_vec)
        config_score = cp.find_similar_score(keywords, configs[i], is_main_path=False, word_vec=word_vec)
        md_score = cp.find_similar_score(keywords, mds[i], is_main_path=False, word_vec=word_vec)
        score = api_score + tok_score + config_score + md_score
        if word_vec and score < 1.5:
            res.append([score,api_score,config_score,tok_score,md_score,repositories[i]])
            continue
        if(api_score + config_score + tok_score + md_score > 0):
            res.append([api_score+config_score+tok_score+md_score,api_score,config_score,tok_score,md_score,repositories[i]])


    if(word_vec):
        res = sorted(res)
    else:
        res = sorted(res, reverse=True)
    nres = [i[-1] for i in res]
    if(with_score):
        if(all_repos):
            if(len(res) > thres):
                n = int(len(res)*default_percent)
                return res[:n]
            return res
        return res[:num]


    if(all_repos):
        if(len(res) > thres):
            n = int(len(nres)*default_percent)
            return nres[:n]
        return nres
    return nres[:num]


def search_repositories(query, platform="aws",num=5, with_score= False, all_repos=False, default_percent=0.2, thres=200, word_vec=False):
    """
    Search repositories based on the given query.
    :params:
        query: Search query
        platform: Platform to search
        num: Number of results to return
        with_score: Return the score of the results
        all_repos: Return all the repositories and ignore num parameter.
        default_percent: If lot's of repositories are returned, trims down to a certain percentage of repositories.
        thres: Threshold for the number of repositories
        word_vec: If True, uses word2vec to search
    :return:
        List of repositories (names)
    """
    keywords = query
    base_path = "./storage/oneplace/" + platform + "/"
    ap_path = "apis/"
    api_path = base_path + ap_path


    api_files = glob.glob(os.path.join(api_path, "*","*.txt"))
    tok_files = glob.glob(os.path.join(api_path, "*","*.tok"))
    cf_files = glob.glob(os.path.join(api_path, "*","*.cf"))
    md_files = glob.glob(os.path.join(api_path, "*","*.md"))


    cp = ch.CodePreprocessor(["sss"])

    res = []
    for i in range(len(api_files)):
        api_score = cp.find_similar_score(keywords,api_files[i],is_main_path=False,word_vec=word_vec)
        config_score = cp.find_similar_score(keywords,cf_files[i],is_main_path=False,word_vec=word_vec)
        token_score = cp.find_similar_score(keywords,tok_files[i],is_main_path=False,word_vec=word_vec)
        md_score = cp.find_similar_score(keywords,md_files[i],is_main_path=False,word_vec=word_vec)
        if(word_vec and api_score + config_score + token_score + md_score < 1.5):
            res.append([api_score+config_score+token_score+md_score,api_score,config_score,token_score,md_score,get_name(api_files[i],2)])
            continue
        if(api_score + config_score + token_score + md_score > 0):
            res.append([api_score+config_score+token_score+md_score,api_score,config_score,token_score,md_score,get_name(api_files[i],2)])

    if(word_vec):
        res = sorted(res)
    else:
        res = sorted(res,reverse=True)
    nres = [i[-1] for i in res]

    if(with_score):
        if(all_repos):
            if(len(res) > thres):
                n = int(len(res)*default_percent)
                if(n > thres):
                    n=200
                return res[:n]
            return res
        return res[:num]


    if(all_repos):
        if(len(res) > thres):
            n = int(len(nres)*default_percent)
            return nres[:n]
        return nres
    return nres[:num]



def get_apis_for_repos(repos, location_mapping):
    """
    Get the list of APIs for the given repositories.
    :params:
        repos: List of repositories
        location_mapping: Location mapping (repo:[location,platform])
    :return:
        Dictionary {repoName: [api1, api2, ...]}
    """

    # We need to think about platform as well.
    # e.g. IBM, AWS, etc.

    apis = dict()
    
    for repo in repos:
        # get all code paths.
        if(repo not in location_mapping):
            continue
        loc = location_mapping[repo][0]
        # print("193 - loc, repo", loc, repo)
        codefiles = cs.get_all_code_files(loc, depth=6)
        codefiles = [cf[1] for cf in codefiles]
        tmp = []
        # print("197 - codefiles", codefiles)
        for cf in codefiles:
            res = get_api_nodes_for_code(cf)
            if res == None:
                continue
            tmp.extend(res)

        tmp = set(tmp)
        apis[repo] = list(tmp)

    return apis

@lru_cache(maxsize=None)
def repo_location_mapping(original_file_path='original_paths.csv'):
    """
    Returns a dictionary that contains repo_name : [location,platform] mapping

    location_mapping = api_nodes.repo_location_mapping()
    location_mapping['anandray_lambda-app'] -> [location, platform]

    :params:
        original_file_path: Path to the original file
    :return:
        Dictionary {repo_name: [location,platform]}
    """
    with open(original_file_path,'r') as infile:
        reader = csv.reader(infile)
        mydict = {rows[0]:rows[1:] for rows in reader}
    return mydict
        


def get_platforms_for_repos(repos, platform='serverless'):
    if(platform != 'serverless'):
        return platform
    
    # We need to calculate which platform is this repository.
    # e.g. IBM, AWS, etc.

    platforms = {}
    for repo in repos:
        # Get serverless.yml.
        serverless_yml = cs.get_serverless_yml(repo)
        
    


if __name__ == "__main__":
    queries = get_important_tags(repo="aws")
    # Custom Queries
    fp = "collected_resources/bot_queries.txt"
    with open(fp, "r") as fp:
        queries = list(fp.readlines())
        queries = [q.replace("\n", "") for q in queries]
    

    original_paths = pd.read_csv("original_paths.csv")
    location_mapping = repo_location_mapping()
    repos = set()
    apis = dict()
    connections = Counter()
    for q in queries:
        print(q)
        platform = 'serverless'
        tmp = list(set(search_repositories(q,platform=platform,num=10)))
        # we have list of repositories.
        # Get apis for each repository.

        ## NOTE : How should we structure for base path? should we include directly? or should we explicitly mention?
        ## This is pretty tricky, we don't exactly know the base path for each repository...
        ## should we create a csv that leads to storage location? We can do that...
        apis.update(get_apis_for_repos(tmp, location_mapping))
        # we have apis for each repository.
        # now we actually need to find what platform each api belongs to...
        # platforms = get_platforms_for_repos(tmp)

        # tmp2 = set(search_repositories(q,platform="aws"))
        # tmp3 = set(search_repositories(q,platform="azure"))
        

        # Get combinations of tmp.
        if len(tmp) <= 1:
            continue

        for i in range(len(tmp)):
            for j in range(i+1, len(tmp)):
                connect_nodes_external(apis[tmp[i]], apis[tmp[j]], connections)
    



        # for repos in tmp, tmp2, tmp3:
        # get api nodes and collect for each of them.
        # connect them together.
        # for repos in tmp:
        # we need to get code files for each repo
        # for each code file, get api nodes, collect them in a list (extend?) (append?) (extend will simplify, make it set to remove repetitions.)
        # now search in azure, aws and serverless for what that repo stands for i.e. its readme?
        # what we need to do is basically connect API nodes from each source to one another. based on the connections, maybe we can find relationships?
        # we need to get api's for repos, connect them together.
        # to get code file we have made code handling class before.
        # print(tmp)
        # print(apis)
        repos = repos.union(tmp)
        print()
        if(len(repos)>=400):
            break
    print(repos)
    print(len(apis))
    print()
    print(connections.most_common(300))

