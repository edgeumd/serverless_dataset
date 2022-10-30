from github import Github
from github import GithubException


from configparser import ConfigParser
import os
import re
import threading as th
import datetime
import time

import yaml


# Loading Token from config file
config = ConfigParser()
config.read('config.ini')
token = config.get('auth', 'token')
url = "https://api.github.com"


# Setting topic to serverless
filename = "serverless"
extension = "yml"

# Size Slices Params
interval = 20
size_start = 990
size_end = 3000

g = Github(token)


def get_size_slices(size_start, size_end, interval):
    """
    Get size slices
    """
    slices = []
    for i in range(size_start, size_end-interval+1, interval):
        slices.append(str(i)+".."+str(i+interval))
    return slices


# Test for get_size_slices
# print(get_size_slices(0, 100, 10))


def get_repo_with_file(filename, extension, size_start, size_end, interval):
    """
    Get repository with file
    """

    slices = get_size_slices(size_start, size_end, interval)
    repo_with_file = []
    c = 1
    slice_count = 0
    ecount = -1
    try:
        for size in slices:
            ecount += 1
            print("Size: ", size)
            repos = g.search_code(
                filename, filename=filename, extension=extension, size=size)

            for repo in repos:
                slice_count += 1
                c += 1
                repo_with_file.append(repo.repository.full_name)
                time.sleep(1)
                if(c % 500 == 0):
                    # Guarding API Limit
                    print("Guarding API Limit")
                    c = 0
                    time.sleep(60)
                # print("Size: ", len(repo_with_file))v
            print("Slice_Count: ", slice_count)
            if(slice_count >= 999):
                print("Error at Size: ", size)
                break
            print("Going to Rest...")
            print("Total Size till now: ", len(repo_with_file))
            slice_count = 0
            time.sleep(120)

    except Exception as e:
        # print("Error", (e))
        print("Recovery mode entered")
        time.sleep(120)
        repo_with_file.extend(get_repo_with_file(
            filename, extension, size_start+(ecount*interval), size_end, interval))

    return repo_with_file


repos = get_repo_with_file(filename, extension=extension,
                           size_start=size_start, size_end=size_end, interval=interval)


print("Writing Repos to {}_repos_with_file.txt ...".format(filename))
with open("{}_repos_with_file.txt".format(filename), "a") as f:
    for repo in repos:
        f.write(repo+"\n")

print("Done...")


class ScrapeCode:
    """
    Scrape Code
    """

    def __init__(self, repo):
        pass

    def get_code(self, contentFile, filename, path=""):
        """
        Get Code
        """
        try:
            content = contentFile.decoded_content
            # TODO check if os.join works...
            with open(os.join(path, filename), "wb") as f:
                f.write(content)
            return True
        except Exception as e:
            print("Error", e)
            return False

    def read_yaml(self, contentfile):
        """
        Read YAML
        """
        try:
            content = contentfile.decoded_content

            return yaml.load(content)
            # return True
        except Exception as e:
            print("Error", e)
            return False

    def read_file(self, repository, filename, filetype):
        """
        Read File

        Returns:
        ContentFile
        """
        return repository.get_contents(filename+"."+filetype)


class ParseYAML:
    """
    Parse YAML
    """

    def __init__(self, repo):
        pass

    def get_function_names(self):
        pass

    def get_file_path(self, function_name):
        pass

    def get_resource_used(self):
        pass

    def get_provider_name(self):
        pass

    def get_runtime(self):
        pass

    def get_plugins(self):
        pass
