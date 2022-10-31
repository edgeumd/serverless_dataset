import json
import re
import remove_empty_base as rm

import os
import glob
import shutil
from typing import List, Set, Tuple
import yaml
from difflib import SequenceMatcher
import csv


class AzureJson:
    def __init__(self) -> None:
        pass


    def get_json_file(self,path):
        try:
            with open(path,"r") as f:
                data = json.load(f)
                return data
        except:
            return {}

    def isValid(self, jsonFile):
        try:
            if("bindings" in jsonFile):
                if(len(jsonFile["bindings"]) > 0):
                    return True
            else:
                return False
        except:
            return False


    def get_functions(self, jsonFile):
        functions = []
        if not self.isValid(jsonFile):
            return []
        for binding in jsonFile["bindings"]:
            try:
                functions.append(binding["name"])
            except:
                continue
        
        return functions


class AzureSpecific(AzureJson):
    def __init__(self) -> None:
        super().__init__()

    def remove_with_no_function(self, path='./storage/nfs/azure/'):
        """
        This function will remove the projects with no template.
        :param path:
        :return:
        """
        to_remove = []
        repositories = glob.glob(path + '*')
        res_templates = []
        for repo in repositories:
            templates = rm.find_files(repo, 'function.json', depth=5)
            # templates = glob.glob(repo + '/**/template.yml', recursive=True)
            if len(templates) == 0:
                to_remove.append(repo)
            else:
                res_templates.extend(templates)
        
        for repo in to_remove:
            try:
                shutil.rmtree(repo)
            except:
                continue

    def isEmptyRepo(self,repo_path):
        """
        This function will remove the projects with no functions.
        :param path:
        :return:
        True if repo is empty
        False if has functions.
        """
        
        # templates = glob.glob(repo_path + '/**/template.yml', recursive=True)
        templates = rm.find_files(repo_path, 'function.json', depth=4)
        required_templates = []
        tot_functions = 0
        for template in templates:
            try:
                yml_file = self.get_json_file(template)
            except:
                continue
            functions = self.get_functions(yml_file)
            tot_functions += len(functions)
            if len(functions) > 0:
                required_templates.append(template)
        if tot_functions == 0:
            return True
        else:
            return False


def remove_empty_azure(base_path="storage/nfs/azure/", repo_path="200-400/"):
    repositories = glob.glob(base_path+repo_path + '*')

    # Remove big projects
    for repo in repositories:
        try:
            sz = (rm.get_size(repo) / (10**7))

            if(sz > 45):
                # print("HERE", repo, sz)
                shutil.rmtree(repo)
                print("REPO BEING REMOVED: ", repo, " SIZE: ", sz)
        except Exception as e:
            continue

    # Remove unlicensed projects
    ls = rm.LicenseSpecific(root_path=base_path, repo_path=repo_path)
    ls.remove_unlicensed_folders(root_path=base_path+repo_path)


    azure = AzureSpecific()

    # Remove projects with no function.json files.
    azure.remove_with_no_function(base_path + repo_path)

    empty_repos = []
    # Removes projects with no functions.
    for repo in repositories:
        if azure.isEmptyRepo(repo):
            empty_repos.append(repo)
    print("Removing ", len(empty_repos), " empty repositories")
    for repo in empty_repos:
        try:
            shutil.rmtree(repo)
        except:
            continue

    ### MORE TO REMOVE: test, examples, .html, .css, .jpg, .png
    repositories = glob.glob(base_path+repo_path+"*")
    filter_keys = ["test", "examples","docs","Example","Test","Docs","example", "tests", "Examples","node_modules"]
    for key in filter_keys:
        for repo in repositories:
            files = rm.find_files(repo, key, depth=7)
            for f in files:
                try:
                    shutil.rmtree(f)
                    print("Removed : ", f)
                except Exception as e:
                    continue

    filter_files = ["*.ipynb", "*.html", "*.css", "*.jpg", "*.png", "*.xml", ".gif","package-lock.json"]
    for key in filter_files:
        for repo in repositories:
            files = rm.find_files(repo, key, depth=7)
            for f in files:
                try:
                    os.remove(f)
                    print("Removed : ", f)
                except Exception as e:
                    continue

if __name__ == "__main__":
    print("NA")