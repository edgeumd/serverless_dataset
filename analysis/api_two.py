from functools import lru_cache
import json
from re import L
import remove_empty_sarvesh as rm
import os
from itertools import chain
import api_nodes
import pandas as pd
from collections import Counter, defaultdict
import networkx as nx
import remove_empty_sarvesh as rm
import os
import glob
from itertools import chain
from networkx.algorithms.community import louvain_communities
import pickle
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import code_handler as ch


class PhaseUtils:
    @staticmethod
    @lru_cache(maxsize=None)
    def load_api_dict(path="api_dict.pkl"):
        """
        Load's the api_dict.pkl file.

        :params:
            path: path to the api_dict.pkl file
        
        :returns:
            global_api_dict: global_api_dict dictionary
        """
        with open(path, "rb") as f:
            return pickle.load(f)

    def get_name(repo_path, from_end=1):
        """
        Get repo name given the repo path.
        :param repo_path:
        """
        return repo_path.split("/")[-from_end]


    def get_license(self, repo, location_mapping, license_specific:rm.LicenseSpecific):
        """
        Given location mapping and repository name, find the license of the repository.

        :params:
            repo: repository name
            location_mapping: location mapping dictionary
            license_specific: license specific object from remove_empty_sarvesh

        :returns:
            license: license of the repository
        """
        # Get platform
        platform = location_mapping[repo][1]

        # Build query
        query = "storage/oneplace/{}/apis/{}/License".format(platform,repo)

        # Check if exists.
        if(not os.path.exists(query)):
            return "NONE"

        # Get license
        license = license_specific.which_license(query)

        return license


    @staticmethod
    def findLink(val):
        res = val.split("_")
        return f'www.github.com/{"".join(res[:-1])}/{res[-1]}'


class Phase1:
    def __init__(self) -> None:
        self.ps = PorterStemmer()

    # Preprocessor for readme files.
    def preprocessReadme(self, readme, isPath=True, saveInPath=False):
        """
        This function preprocesses the readme files by removing stopwords and applying stemming to it.

        :params:
            readme: path to the readme file by default, else string.
            isPath: determines if readme is a path or not. (default true)
            saveInPath: False. if we want to save in the path or return as string.(default string)

        :returns:
            readmeClean in path if isPath or readmeClean as string if isPath is false.
        """
        readmeFile = ""
        if(isPath):
            with open(readme, "r") as f:
                readmeFile = f.read()
        else:
            readmeFile = readme

        englishstop = set(stopwords.words('english'))

        readmeFile = readmeFile.lower()

        readmeClean = " ".join([self.ps.stem(i) for i in readmeFile.split(" ") if i not in englishstop])
        
        if(saveInPath):
            with open(readme, "wb") as f:
                f.write(readmeClean)

            return

        return readmeClean

        
    # For keyword based recommendation, we will generally need just common similar keywords
    # In programming perspective we need two functions, one responsible for calculating score for an individual file while other responsible for the rest.
    def keywordScore(self, myfile, processedQuery, isPath=True):
        """
        This function is responsible for returning the match score for a given file or string.

        :params:
            myfile: file path or raw string. (default filepath)
            processedQuery: query string that we need to search against. (string)
            isPath: lets us control myfile params behaviour. (default true)

        :returns:
            score: the higher the better.
        """
        finalFile = ""

        if(isPath):
            with open(myfile, "rb") as f:
                finalFile = f.read()
        else:
            finalFile = myfile

        finalFile = str(finalFile)
        finalFile = finalFile.split(" ")

        keywordCounts = Counter(finalFile)

        processedQueryList = processedQuery.split(" ")

        score = 0

        for q in processedQueryList:
            if q in keywordCounts:
                score += keywordCounts[q]
        
        return score

    def keywordProcessPath(self, filePath, query, isReadme=False):
        """
        Use for keyword score finding.

        :params:
            filePath: path of file for which we need to perform keyword matching.
            query: query that we want to search against.
            isReadme: determines if it is readme or not as readme requires special preprocessing.

        :returns:
            score.
        """
        fileRaw = ""
        with open(filePath,"rb") as f:
            fileRaw = f.read()

        if(isReadme):
            fileRaw = self.preprocessReadme(fileRaw, isPath=False)
            query = self.preprocessReadme(query, isPath=False)

        return self.keywordScore(fileRaw, query, isPath=False)

        
    # Basic algorithm to search for repositories. Phase 1.
    def search_repositories(self, query, platform="aws",num=5, with_score= False, all_repos=False, default_percent=0.2, thres=200, word_vec=False):
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
            # else:
            #     api_score = self.keywordProcessPath(api_files[i], keywords)
            #     config_score = self.keywordProcessPath(cf_files[i], keywords)
            #     token_score = self.keywordProcessPath(tok_files[i],keywords)
            #     md_score = self.keywordProcessPath(md_files[i], keywords)
            if(word_vec and api_score + config_score + token_score + md_score < 1.5):
                res.append([api_score+config_score+token_score+md_score,api_score,config_score,token_score,md_score,PhaseUtils.get_name(api_files[i],2)])
                continue
            if(api_score + config_score + token_score + md_score > 0):
                res.append([api_score+config_score+token_score+md_score,api_score,config_score,token_score,md_score,PhaseUtils.get_name(api_files[i],2)])

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



class Phase2:

    def get_api_dict_from_global(self, ntmp, global_api_dict):
        """
        Given a list of repositories, returns a dictionary of api_dict for the repositories.

        :params:
            ntmp: list of repositories
            global_api_dict: global api_dict dictionary

        :returns:
            api_dict: api_dict dictionary
        """
        api_dict = {}
        for repo in ntmp:
            if(repo in global_api_dict):
                api_dict[repo] = global_api_dict[repo]
        return api_dict


    def get_api_counts(self, api_dict):
        """
        Given an api_dict, returns a dictionary of api counts

        :params:
            api_dict: dictionary of repository: api_string_list ("api1 api2 api3")
        :returns:
            api_counts: dictionary of api:count
        
        """
        api_counts = Counter()
        for i in api_dict:
            res = api_dict[i].split(" ")
            for api in res:
                if(api == ""):
                    continue
                api_counts[api] += 1
        return api_counts

    def get_api_counts(self, api_dict):
        """
        Given an api_dict, returns a dictionary of api counts

        :params:
            api_dict: dictionary of repository: api_string_list ("api1 api2 api3")
        :returns:
            api_counts: dictionary of api:count
        
        """
        from collections import Counter
        api_counts = Counter()
        for i in api_dict:
            res = api_dict[i].split(" ")
            for api in res:
                if(api == ""):
                    continue
                api_counts[api] += 1
        return api_counts

    def get_repo_weight_dict(self, repo_list_with_weights):
        """
        Returns repository:weight dictionary from search with score = true.

        :params:
            repo_list_with_weights: result of api_nodes.search_repositories ([[0,0,0,0,0,reponame]])

        :returns:
            repo_weight_dict: dictionary of repo:weight

        """
        repo_weight_dict = {}

        weights = [repo[0] for repo in repo_list_with_weights]
        avgWeight = sum(weights)/len(weights)
        

        for i in range(len(repo_list_with_weights)):
            repo_weight_dict[repo_list_with_weights[i][-1]] = (repo_list_with_weights[i][0] + (avgWeight/(i+1)))
        return repo_weight_dict

    def get_weight(self, api, api_counts, adjust_by=10, clip=None):
        """
        Given an api and api_counts, returns the weight of the api

        :params:
            api: api to be weighted
            api_counts: api_counts dictionary
            adjust_by: adjust the weight by this value
            clip: clip the weight to this value
        :returns:
            weight: weight of the api

        """
        res = api_counts[api]
        if(clip != None and res < clip):
            return 0
        if(res != 0):
            return (1/res) * adjust_by
            # return 1
        else:
            return 0

    def calc_weight_between_repositories(self, repo1, repo2, counts, api_dict):
        """
        Given two repositories, returns the weight of the edge between them

        :params:
            repo1: repository name
            repo2: repository name
            counts: api_counts dictionary
        :returns:
            weight: weight of the edge between the two repositories

        """
        # Base case.
        if(repo1 == repo2):
            return 0

        average = int(sum(list(counts.values()))/len(list(counts.values())))
        
        corressponding = {
            "s3": "blob",
            "S3": "blob",
            "dynamodb": "cosmos",
            "DynamoDB": "cosmos",
            "sns":"eventgrid",
            "sns":"event-grid",
            "sqs":"queue",
            "ses":"sendgrid",
            "kinesis":"eventhub",
            "kinesis":"event-hubs",
            "kinesisanalytics":"eventhub",
            "kinesisanalytics":"event-hubs",
            "lex":"botbuilder",
            "polly":"speech",
        }
        corressponding_rev = {v:k for k,v in corressponding.items()}
        corressponding.update(corressponding_rev)

        # If both repositories is known then proceed.
        if(repo1 in api_dict and repo2 in api_dict):
            api_list1 = api_dict[repo1].split(" ")
            tmp = [corressponding[i] for i in api_list1 if i in corressponding]
            api_list1.extend(tmp)
            api_list2 = api_dict[repo2].split(" ")
            # First Set, add corressponding apis.
            common_apis = set(api_list1).intersection(api_list2)
            if(len(common_apis) == 0):
                return 0
            return sum([self.get_weight(i, counts, clip=average) for i in common_apis])
        else:
            return 0

    
    def get_community_energy(self, community, repo_weight, thresh=5):
        """
        Given a community and repo_weight dictionary, returns the energy of the community

        :params:
            community: community to be evaluated
            repo_weight: dictionary of repo:weight
        :returns:
            energy: energy of the community

        """
        if(len(community) < 3):
            return 0
        return sum([repo_weight[i]*thresh for i in community])/len(community)
    
    # Probably not needed
    def find_match(self, selected_apis,repositories, api_dict, threshold=1,skip=False):
        """
        Given a repository and api_dict, returns the number of matching apis.

        :params:
            selected_apis: set of selected apis
            repository: repository name
            api_dict: api_dict dictionary
            threshold: threshold for matching
        :returns:
            match: number of matching apis

        """
        res = []
        for repository in repositories:
            if(repository in api_dict):
                api_list = api_dict[repository].split(" ")
                if(skip):
                    if(len(api_list) > 15):
                        continue
                mlen = len([i for i in api_list if i in selected_apis])
                if(mlen > threshold):
                    res.append(repository)
        return res


    
    
    def processPhase2(self, phase1_result, thresh=5, debug=False):
        """
        Takes in phase1 result and returns selected API and repositories in a community.

        params:
            phase1_result
            thresh: how much important is repository by itself.
        """
        # Getting the results.
        tmp = phase1_result.copy()
        searched_repositories = tmp.copy()
        ntmp = [i[-1] for i in tmp][:100]
        if(debug):
            print(f'Phase 1 Result: {ntmp[:10]}')

        global_api_dict = PhaseUtils.load_api_dict()

        # Loading the api dict for the given repositories.
        api_dict = self.get_api_dict_from_global(ntmp, global_api_dict)

        # Getting counts for API from api_dict, location mapping for all repositories and repo_weight dictionary based on the score.
        counts = self.get_api_counts(api_dict)
        repo_weight = self.get_repo_weight_dict(tmp)
        if(debug):
            print(f'Repository Weight: {[repo_weight[i] for i in ntmp[:10]]}')

        # Creating a graph.
        G = nx.Graph()
        G.add_nodes_from(list(repo_weight.keys()))
        repositories = list(repo_weight.keys())

        for i in range(len(repositories)):
            for j in range(i+1, len(repositories)):
                # calculate weight between two repos needed.
                w = self.calc_weight_between_repositories(repositories[i], repositories[j], counts, api_dict)
                if(w != 0):
                    G.add_edge(repositories[i], repositories[j], weight=w)
        partition = louvain_communities(G, weight='weight')

        # Getting the best community.
        max_comm = set()
        max_energy = 0
        for i in partition:
            energy = self.get_community_energy(i, repo_weight)
            if(energy > max_energy):
                max_energy = energy
                max_comm = i

        if(debug):
            print(f'Repositories in Max Community: {max_comm}')
        fin = []
        for res in max_comm:
            fin.append((res, repo_weight[res]))

        new_comm = [i[0] for i in sorted(fin, key=lambda x: x[1], reverse=True)[:35]]
        if(debug):
            print(f'Selected Community : {new_comm}')


        mlis = []
        for i in new_comm:
            tmp = api_dict[i]
            tmp = tmp.split(" ")
            tmp = list(set(tmp))
            mlis.extend(tmp)

        mlis = Counter(mlis)
        selected_apis = {i[0] for i in mlis.most_common(15)}
        

        # print(f'Selected APIs : {selected_apis}')
        return selected_apis, new_comm




class Phase3:
    # Finds repositories that match with selected apis.
    def clearRepositories(self, selectedApis, repositories, api_dict=PhaseUtils.load_api_dict(), threshold=1, skip=False):
        """
        Removes repositories that does not use selected APIs.

        :params:
            selectedApis: set of selected apis
            repositories: repositories
            api_dict: api_dict dictionary repo:[api1,...], default: global api dict from PhaseUtils.
            threshold: threshold for matching i.e. only select repo if they have more than threshold intersection.
            skip: if total number of api's for a repository greater than 15, skip. (default false)
        :returns:
            repositories

        """
        res = []
        count = 0
        for repository in repositories:
            if(type(repositories) is list):
                if(count < 5):
                    res.append(repository)
                    count += 1
                    continue
            if(repository in api_dict):
                api_list = api_dict[repository].split(" ")
                if(skip):
                    if(len(api_list) > 15):
                        continue
                mlen = len([i for i in api_list if i in selectedApis])
                if(mlen > threshold):
                    res.append(repository)
        return res


    def findApiBasedRepositories(self, selectedApis, detectedCommunity, searchedRepositories):
        """
        Finds Repositories based on the selected APIs and selected repositories within the community.

        # TODO check if we can do it for all possible search results directly? should be possible, get count and based on that do it.
        
        :params:
            selectedApis: Api's selected in phase 2.
            detectedCommunity: community that we detected in phase 2.
            searchedRepositories: searched repositories in phase 1.

        :returns:
            a list with repo names without api intersection score with atleast 1 intersection.
        """
        globalApiDict = PhaseUtils.load_api_dict()
        ind = 10
        selectedRepositories = []

        while ind > 0 and len(selectedRepositories) == 0:
            ind -= 1
            selectedRepositories = self.clearRepositories(selectedApis, globalApiDict.keys(), threshold=ind)
            for repo in selectedRepositories:
                if repo in detectedCommunity or repo in searchedRepositories:
                    selectedRepositories.remove(repo)

        return selectedRepositories


    def get_license(self, repo, location_mapping, license_specific:rm.LicenseSpecific):
        """
        Given location mapping and repository name, find the license of the repository.

        :params:
            repo: repository name
            location_mapping: location mapping dictionary
            license_specific: license specific object from remove_empty_sarvesh

        :returns:
            license: license of the repository
        """
        # Get platform
        platform = location_mapping[repo][1]

        # Build query
        query = "storage/oneplace/{}/apis/{}/License".format(platform,repo)

        # Check if exists.
        if(not os.path.exists(query)):
            return "PARTIAL"

        # Get license
        license = license_specific.which_license(query)

        return license[0]


    def processPhase3(self, selectedAPIs, detectedCommunity, searchedRepositories, withScore, debug=False):
        """
        It returns reranked list of repositories after filtering, discovering new repositories and then reranking them.

        :params:
            selectedAPIs: Selected API's from phase 2
            detectedCommunity: Detected Community from phase 2
            searchedRepositories: result from phase 1.
            withScore: determines if the phase 1 result was with score or without score.

        :returns:
            (TODO least restrictive to most restrictive licenses.)
            list of repositories in reranked order.
        """

        # List of things to do in the following order.
        # First, discover more repositories that are not already in searched or detected community.
        # Add detected community to the list.
        # filter repositories.
        # Find license and rerank based on restrictiveness of license.

        if(withScore):
            searchedRepositories = [i[-1] for i in searchedRepositories][:100]
        
        searchedRepositoriesSet = set(searchedRepositories)
        
        # Discover
        apiBasedRepositories = self.findApiBasedRepositories(selectedAPIs, detectedCommunity, searchedRepositoriesSet)


        # Add
        searchedRepositories.extend(apiBasedRepositories)

        # Filter
        filteredRepositories = self.clearRepositories(selectedAPIs, searchedRepositories)


        if(len(filteredRepositories) < 4):
            filteredRepositories = searchedRepositories[:5] + [i for i in filteredRepositories if i not in searchedRepositories[:5]]
        else:
            filteredRepositories = filteredRepositories + [i for i in searchedRepositories if i not in filteredRepositories][:3]

        # Rerank based on license.
        location_mapping = api_nodes.repo_location_mapping()
        licenseMapping = defaultdict(list)

        ls = rm.LicenseSpecific()
        
        if debug:
            print(f'Original Repositories: {searchedRepositories[:10]}')
            print()
            print(f'Selected APIS: {selectedAPIs}')
            print()
            print(f'Filtered Repositories: {filteredRepositories}')
            print()
            print(f'Extra Repositories Based On API: {[repo for repo in filteredRepositories if repo not in searchedRepositories[:10]]}')
            print()
            print(f'Removed Repositories: {[repo for repo in searchedRepositories[:10] if repo not in filteredRepositories]}')
            print()
            print(f'Filtered Repo Licenses {[self.get_license(repo, location_mapping, ls) for repo in filteredRepositories][:5]}')


        

        for repo in filteredRepositories:
            repoLicense = self.get_license(repo, location_mapping, ls)
            licenseMapping[repoLicense].append(repo)

        finalRepositoryList = licenseMapping["unlicense"] + licenseMapping["mit"] + licenseMapping["bsl1"] + licenseMapping["apache2"] + licenseMapping["mpl2"] + licenseMapping["lgpl3"] + licenseMapping["gnu_gplv3"] + licenseMapping["gnu_agplv3"] + licenseMapping["sspl"] + licenseMapping["PARTIAL"]

        return finalRepositoryList





if __name__=="__main__":
    phase1 = Phase1()
    phase2 = Phase2()
    phase3 = Phase3()

    # TODO check if i am adding all the results appropriately.
    # TODO check if I am reranking properly. see if other considerations are to be made.

    query = "information retrieval and data access"
    isWordVec = False
    platform = "serverless"
    numberOfRepositories = 10
    ingoreNumberOfRepositories = True
    withScore = True

    # currently phase1 result is a list of repositories.
    phase1_result = phase1.search_repositories(query, word_vec=isWordVec, platform=platform, num=numberOfRepositories, all_repos=ingoreNumberOfRepositories, with_score=withScore)

    
    # As an output we have two things, 1. selected API and 2. Detected Community.
    selected_apis, detected_community = phase2.processPhase2(phase1_result=phase1_result[:100], debug=False, thresh=20)

    phase3_result = phase3.processPhase3(selectedAPIs=selected_apis, detectedCommunity=detected_community, searchedRepositories=phase1_result[:100], withScore=withScore, debug=True)

    print(phase3_result)