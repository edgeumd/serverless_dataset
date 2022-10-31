import re
import remove_empty_base as rm

import os
import glob
import shutil
from typing import List, Set, Tuple
import yaml
from difflib import SequenceMatcher
import csv




def find_files(root, filename, depth=1):
    """
    Finds file for depth d.
    """
    template_files = []
    s = "*/"
    for i in range(depth):
        sd = root + '/' + s*i + filename
        files = glob.glob(sd)
        template_files.extend(files)
    return template_files


class AWSYML(rm.YMLBase):
    def __init__(self) -> None:
        pass

    def isValid(self,ymlfile):
        """
        Returns true if contains two properties: 
            1. AWSTemplateFormatVersion
            2. Resources
        """
        try:
            if "AWSTemplateFormatVersion" in ymlfile and "Resources" in ymlfile:
                return True
            else:
                return False
        except:
            return False

    def get_functions(self,yml_file):
        """
        Returns functions associated with file.
        """
        if not self.isValid(yml_file):
            return []
        functions = []
        try:
            for key, value in yml_file["Resources"].items():
                if value["Type"] == "AWS::Serverless::Function":
                    functions.append(key)
        except:
            return []
        return functions

    


class AWSSpecific:
    def __init__(self) -> None:
        self.ayml = AWSYML()


    def remove_with_no_template(self, path='./storage/nfs/aws/'):
        """
        This function will remove the projects with no template.
        :param path:
        :return:
        """
        to_remove = []
        repositories = glob.glob(path + '*')
        res_templates = []
        for repo in repositories:
            templates = find_files(repo, 'template.yml', depth=3)
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
        templates = find_files(repo_path, 'template.yml', depth=3)
        required_templates = []
        tot_functions = 0
        for template in templates:
            try:
                yml_file = self.ayml.get_yaml_file(template)
            except:
                continue
            functions = self.ayml.get_functions(yml_file)
            tot_functions += len(functions)
            if len(functions) > 0:
                required_templates.append(template)
        if tot_functions == 0:
            return True
        else:
            return False

     
def remove_empty_aws(base_path="storage/nfs/aws/", repo_path="200-400/"):
    aws = AWSSpecific()
    aws.remove_with_no_template(base_path+repo_path)
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
        if aws.isEmptyRepo(repo):
            empty_repos.append(repo)
    print("Removing ", len(empty_repos), " empty repositories")
    for repo in empty_repos:
        try:
            shutil.rmtree(repo)
        except Exception as e:
            continue



if __name__ == "__main__":
    aws = AWSSpecific()
    aws.remove_with_no_template()
    repositories = glob.glob('./storage/nfs/aws/*')
    empty_repos = []
    for repo in repositories:
        if aws.isEmptyRepo(repo):
            empty_repos.append(repo)
    print("Removing ",len(empty_repos), " empty repositories")
    for repo in empty_repos:
        shutil.rmtree(repo)

    