import re
import remove_empty_base as rm
from remove_empty_base import find_files

import os
import glob
import shutil
from typing import List, Set, Tuple
import yaml
from difflib import SequenceMatcher
import csv


# https://openwhisk.apache.org/documentation.html

class IBMYML(rm.YMLBase):
    """
    We already have a yml file loader from yml base.
    We first will check if actions: property is present.
    If it is present, we will check number of functions there.
    to identify if its useful to keep this file.
    """
    def __init__(self):
        super().__init__()

    
    def isValid(self, yml_file):
        """
        Returns true if file is valid.
        An IBM cloud functions file is valid if it has "actions" property.
        """
        try:
            if "packages" in yml_file:
                keys = yml_file["packages"].keys()
                count = 0
                for k in keys:
                    if("actions" in yml_file["packages"][k]):
                        count += 1

                if count > 0:
                    return True
        except:
            return False
        return False
    def get_functions(self,yml_file):
        """
        Returns functions associated with the file.
        packages -> package_name -> actions -> function_name -> functions:
        """
        if not self.isValid(yml_file):
            return []
        functions = []
        try:
            for package in yml_file["packages"].keys():
                for action in yml_file["packages"][package]["actions"].keys():
                    if "function" in yml_file["packages"][package]["actions"][action].keys():
                        functions.append(action)
        except:
            return []
        return functions
    

class IBMSpecific:
    def __init__(self) -> None:
        self.iyml = IBMYML()


    def remove_with_no_template(self, path='./storage/nfs/ibm/'):
        """
        This function will remove the projects with no template.
        :param path:
        :return:
        """
        to_remove = []
        repositories = glob.glob(path + '*')
        res_templates = []
        for repo in repositories:
            templates = find_files(repo, 'manifest.yml', depth=6)
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
        templates = find_files(repo_path, 'manifest.yml', depth=6)
        required_templates = []
        tot_functions = 0
        for template in templates:
            try:
                yml_file = self.iyml.get_yaml_file(template)
            except:
                continue
            functions = self.iyml.get_functions(yml_file)
            tot_functions += len(functions)
            if len(functions) > 0:
                required_templates.append(template)
        if tot_functions == 0:
            return True
        else:
            return False

     
def remove_empty_ibm(base_path="storage/nfs/aws/", repo_path="200-400/"):
    ibm = IBMSpecific()
    ibm.remove_with_no_template(base_path+repo_path)
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



    empty_repos = []
    for repo in repositories:
        if ibm.isEmptyRepo(repo):
            empty_repos.append(repo)
    print("Removing ", len(empty_repos), " empty repositories")
    for repo in empty_repos:
        try:
            shutil.rmtree(repo)
        except Exception as e:
            continue
