from github import Github

from configparser import ConfigParser
import threading as th
from datetime import datetime
import time
import pickle


class GSearch:
    def __init__(self, config_path='config.ini', filename="serverless", extension="yml"):
        """
        ConfigFile should have : token within auth i.e. [auth] token=<token>
        """
        self.config = ConfigParser()
        self.config.read(config_path)
        self.token = self.config.get('auth', 'token')
        self.url = "https://api.github.com"
        self.filename = filename
        self.extension = extension

        self.g = Github(self.token)

    def get_size_slices(self, size_start=0, size_end=1000, interval=20):
        """
        Get size slices
        """
        slices = []
        for i in range(size_start, size_end-interval+1, interval):
            slices.append(str(i)+".."+str(i+interval))
        return slices

    def get_repo_with_file(self, filename, extension, size_start, size_end, interval, query="", create_file=False, create_code=False, save_file_path="dataset",page=0,fail=False, without_filename=False):
        """
        Get repository with file mentioned. e.g. searching for all repositories with serverless.yml

        filename: Filename to search for. e.g. serverless.yml
        extension: Extension of the file. e.g. yml
        size_start: Start of the size range. e.g. 0
        size_end: End of the size range. e.g. 1000
        interval: Interval of the size range. e.g. 20 for per execution



        --- Default Parameters ---
        query: Query to search for within the file. e.g. stepFunctions:
        without_filename: False 

        Returns:
        List of repositories with the file mentioned.

        """

        slices = self.get_size_slices(size_start, size_end, interval)
        repo_with_file = []

        c = 1
        slice_count = 0
        ecount = -1
        try:
            for size in slices:
                ecount += 1
                print("Size: ", size)
                if(without_filename):
                    repos = self.g.search_code(
                        query, extension=extension, size=size)
                else:
                    repos = self.g.search_code(
                        query, filename=filename, extension=extension, size=size)

                
                for i in range(page, 10):
                    r = repos.get_page(i)
                    for repo in r:
                        slice_count+=1
                        c +=1

                        # repo.repository.full_name, repo.repository.html_url, repo.path, repo.html_url)
                        print(repo.repository.full_name)
                        print(repo.path)
                        repo_with_file.append((repo.repository.full_name, repo.repository.html_url, repo.path, repo.html_url))
                        page = i
            


                # for repo in repos:
                #     slice_count += 1
                #     c += 1
                #     if(create_file):
                #         # self.save_file(repo.repository.full_name+"_" +
                #         #    "serverless", repo.decoded_content, save_file_path)
                #         pass

                #     if(create_code):
                #         # TODO Do something
                #         pass
                #     print(repo.repository.full_name)
                #     repo_with_file.append(
                #         (repo.repository.full_name, repo.repository.html_url, repo.path, repo.html_url))

                #     # time.sleep(1)
                #     if(c % 500 == 0):
                #         # Guarding API Limit
                #         print("Guarding API Limit")
                #         c = 0
                #         # time.sleep(60)
                #     # print("Size: ", len(repo_with_file))v
                print("Slice_Count: ", slice_count)
                if(slice_count >= 999):
                    print("Error at Size: ", size)
                    break
                print("Going to Rest...")
                print("Total Size till now: ", len(repo_with_file))
                slice_count = 0
                # time.sleep(60)

        except KeyboardInterrupt:
            print("Keyboard Interrupt")
            print("Size Start: ", size_start)
            print("Size End: ", size_end)
            print("Interval: ", interval)
            print("Total Size till now: ", len(repo_with_file))
            return repo_with_file

        except Exception as e:
            print("Error", (e))
            print("Recovery mode entered")
            print(size_start+(ecount*interval))
            if(fail):
                page = page+1
                fail = False

            else:
                fail = True
            if("secondary" not in e.args[1]["message"] and "rate limit" not in e.args[1]["message"]):
                print(e)
                return repo_with_file
            time.sleep(30)
            print(page)
            repo_with_file.extend(self.get_repo_with_file(
                filename, extension, size_start+(ecount*interval), size_end, interval, page=page,fail=fail))

        return repo_with_file

    def save_file(self, filename, content, path="dataset"):
        """
        Save the file to the path mentioned.
        """
        with open(path+"/"+filename+".pkl", "wb") as f:
            pickle.dump(content, f)

        # time.sleep(20)

    def load_file(self, filename, path="dataset"):
        """
        Load the file from the path mentioned.
        """
        with open(path+"/"+filename, "rb") as f:
            content = pickle.load(f)
        return content

    def get_code_using_sls(self, content):
        """
        TODO find all code files related to the content
        """
        pass


if __name__ == "__main__":
    g = GSearch()
    repos = []
    spaces = 100
    for i in range(0, 10000, spaces):
        
        # AZURE
        # by searching function.json

        # IBM
        repos = g.get_repo_with_file("manifest", "yml", i, i+spaces,
                                        spaces, query='actions:', create_file=True, create_code=False, save_file_path="dataset", without_filename=False)
        
        # repos = g.get_repo_with_file("serverless", "yml", i, i+1,
                                    #  1, query="", create_file=True, create_code=False, save_file_path="dataset")
        # repos = g.get_repo_with_file("template", "yml", i, i+1,
                                    #  1, query="AWSTemplateFormatVersion", create_file=True, create_code=False, save_file_path="dataset")

        g.save_file("repos-{}-{}".format(i, i+spaces) +
                    str(datetime.now()), repos)
        # time.sleep(120)

    # print(repos)
    #TILL 550
    # 0-100,100-200,200-300, 250-350, 350-450, 450-550,550-551,551-552...
