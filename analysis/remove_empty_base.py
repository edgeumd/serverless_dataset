# Contains 4 Classes:
# 1. FileTraversal
# 2. YMLSpecific
# 3. CodeSpecific
# 4. LicenseSpecific

import os
import glob
import shutil
from typing import List, Set, Tuple
import yaml
from difflib import SequenceMatcher
import csv

from pathlib import Path

def get_size(folder: str) -> int:
    """
    Returns size of the folder.
    """
    return sum(p.stat().st_size for p in Path(folder).rglob('*'))

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





class FileTraversal:
    def __init__(self, root_path="storage/nfs/",repo_path="serverless") -> None:
        self.root_path = root_path
        self.repo_path = repo_path
        self.storage_path = root_path+repo_path+"/"
        self.non_fragmented_features = ["root_folder","path","runtime","provider_name","plugins", "functions", "number_of_functions", "license"]
        self.fragmented_features= ["root_folder","path","runtime","plugins","number_of_functions","functions","single_function","function_path","function_event","license"]
        self.nff_eval = [4,5,6,7]
        self.ff_eval = [4,5,6,9,10]
        self.eval_lis = {"fragmented":self.ff_eval, "non_fragmented":self.nff_eval, "code":[]}


    def get_specific_files(self,path=None,filename="serverless.yml",recursive=True, exclude_examples=True, depth = 5) -> List[str]:
        """
        Returns a list of files that match the given filename

        :params:
            path: The path to search for files default is "**/" i.e. any file in any subdirectory.
            filename: default is serverless.yml
            recursive: default is True
            exclude_examples: default is True
            depth: how deep to search. Default=5
        
        :returns:
            list of files
        """
        files = find_files(self.storage_path, filename, depth=depth)
        path = path or self.storage_path + "**/"
        # files= sorted(glob.glob(path+filename,recursive=recursive))
        if exclude_examples:
            exclude = {"example","test","demo","template"}
            files = [i for i in files if not any(x in i for x in exclude)]
        return files


    def get_multiple_files(self, path=None, filename=["LICENSE","LICENSE.txt"],recursive=True, depth=3):
        """
        Returns a list of files that match the given filename
        :param path: The path to search for files default is "**/" i.e. any file in any subdirectory.
        :param filename: default is ["LICENSE","LICENSE.txt"]
        :param recursive: default is True (depreciated)
        :param depth: how deep to search. default 3.
        :return:
        list of files
        """
        path = path or self.storage_path + "**/"
        files = []
        for i in filename:
            files = files + self.get_specific_files(path=path, filename=i, recursive=recursive, depth=depth)
        return files


    def get_root_folder(self, path, storage_path=None,adjust_by=1) -> str:
        """
        Returns the root folder of serverless file.
        :param path:
        :param storage_path:
        :return:

        note: adjust_by 1 coz extra / at end.
        adjust accordingly.
        """
        storage_path = storage_path or self.storage_path
        return storage_path+path.split("/")[len(storage_path.split("/")) - adjust_by]

    def get_serverless_path(self, path)->str:
        """
        Returns path for serverless.yml location.
        """
        return path.replace("serverless.yml","")

    
    def save_result(self, res, path="result.csv", features=["root_folder","path","runtime","provider_name","plugins", "functions", "number_of_functions", "license"],sep=":"):
        """
        Save's the result as csv.
        
        params:
        res: list of tuples
        features: list of features to be saved. (default is for process_yaml ref. YMLSpecific)
        sep: Separator for csv. default is " +++x+++ "
        """
        with open(path,"w") as f:
            cw = csv.writer(f, delimiter=sep)
            cw.writerow(features)
            for val in res:
                cw.writerow(val)

    def load_result(self, path="result.csv", sep=":",type="non_fragmented") -> List[Tuple]:
        """
        Loads result from csv.
        :param path:
        :param sep:
        :param fragmented:
        :return: result

        type: fragmented, non_fragmented, code
        """
        eval_lis = self.eval_lis[type]
        with open(path, "r") as f:
            cr = csv.reader(f, delimiter=sep)
            res = []
            count =0
            for row in cr:
                if(count ==0):
                    count+=1
                    res.append(row)
                    continue
                # print(row)
                r = list(row)
                for i in eval_lis:
                    r[i] = eval(r[i])
                res.append(tuple(r))
        return res
        



    def get_project_id(self, path) -> str:
        """
        Returns Project ID
        """
        # TODO Return project id.
        return path



    
class LicenseSpecific:
    def __init__(self, root_path="storage/nfs/",repo_path="serverless", permissions_path="license_data/permissions/permissions.txt", licenses_path="license_data/licenses/"):
        self.permissions_path = permissions_path
        self.root_path = root_path
        self.repo_path = repo_path
        self.storage_path = root_path+repo_path+"/"
        self.licenses_path = licenses_path
        self.permissions = self.load_permissions()
        self.licenses = self.load_licenses()
        self.permissions["NONE"] = [["NOT ALLOWED"],["CHECK REPO FOR MORE INFO"],[]]

        self.ft = FileTraversal(root_path=self.root_path,repo_path=self.repo_path)
        # TODO LOAD LICENSE AND PERMISSIONS

    def get_license_files(self, depth=7):
        """
        Returns list of folders containing LICENCE files.
        :params:
        depth: how deep to search for license file. Defaults to 7.

        :returns:
        List of license's found for the given depth. (default depth is 7)
        """
        return self.ft.get_multiple_files(depth=depth)

    def get_license_files_in_root(self, root_folder=None):
        """
        Returns list of files in root folder.
        """
        root_folder = root_folder or self.root_path + self.repo_path + "/"
        return self.ft.get_multiple_files(path=root_folder+"*/")

    def get_to_remove_folders(self, root_path = None):
        """
        Returns list of folders that are to be removed.
        """
        root_path = root_path or self.root_path + self.repo_path + "/"
        license_files = self.get_license_files_in_root()
        lf_new = []
        for i in license_files:
            lf_new.append(self.ft.get_root_folder(i,storage_path=root_path))
        lf_new = set(lf_new)
        all_files = set(glob.glob(root_path+"*"))

    
        return list(all_files - lf_new)


    def remove_unlicensed_folders(self, root_path = None):
        """
        Removes folders that do not have license.
        """
        root_path = root_path or self.root_path + self.repo_path + "/"
        to_remove = set(self.get_to_remove_folders(root_path))
        for i in to_remove:
            try:
                shutil.rmtree(i)
            except:
                continue




    def load_permissions(self):
        """
        Load Permissions file.
        """
        with open(self.permissions_path, "r") as f:
            p = dict()
            for line in f.readlines():
                p[eval(line)[0]] = eval(line)[1:]
        return p

    def load_licenses(self):
        """
        Loads all license files
        
        """
        licenses = dict()
        for key in self.permissions.keys():
            licenses[key] = self.open_license_file(self.licenses_path+str(key)+".txt")
        return licenses

    def checkLicense(self, path , root=True)->bool:
        """
        Returns True and License name if license file exists.
        otherwise returns False and NONE
        """
        license_files = ["LICENSE","LICENSE.txt", "License.txt","license.txt","License","license"]
        
        for file in license_files:
            if root:
                if len(glob.glob(os.path.join(self.ft.get_root_folder(path),file))) > 0:
                    return True, self.which_license(glob.glob(os.path.join(self.ft.get_root_folder(path),file))[0])
            else:
                if len(glob.glob(os.path.join(self.ft.get_serverless_path(path),file))) > 0:
                    return True, self.which_license(glob.glob(os.path.join(self.ft.get_serverless_path(path),file))[0])
        return False,("NONE",1.0)

    def open_license_file(self, file_path):
        """
        Given a filepath to a license, returns the license text.
        params:
        file_path: path to a file
        returns:
        license text
        """
        with open(file_path, "r") as f:
            license_text = f.read()
        license_text = license_text.lower()
        license_text = license_text.strip()
        license_text = license_text.replace("\n"," ")
        return license_text

    def compare_files(self, f1, f2):
        """
        Basic sequence matching between two files.

        params:
        f1: str file
        f2: str file
        """
        m = SequenceMatcher(None, f1, f2)
        return m.ratio()


    def get_license_root_folder(self,path, storage_path=None,adjust_by=1):
        """
        Given a path to a folder, returns the license folder.
        params:
        path: path to a file
        storage_path: default is None
        adjust_by: default is 1 ref to FileTraversal.get_root_folder
        """
        storage_path = storage_path or self.storage_path
        return self.ft.get_root_folder(path,storage_path,adjust_by)

    def which_license(self,file_path):
        """
        Given a filepath to a license, returns which license it is.
        e.g. ./storage/oneplace/aws/apis/anandray_lambda-app/License
        params:
        file_path: path to a file

        returns:
        license name, ratio
        """
        # LOAD
        license_text = self.open_license_file(file_path)
        # Compare
        ratios = dict()
        for key in self.licenses.keys():
            ratios[key] = self.compare_files(license_text, self.licenses[key])
        res = max(ratios, key=ratios.get)
        return res, ratios[res]    

class YMLBase:
    def __init__(self):
        pass

    def get_yaml_file(self,path):
        """
        Returns yaml file from the given yaml path.
        """
        with open(path,"r") as f:
            return yaml.load(f, Loader=yaml.BaseLoader)

    



class YMLSpecific(YMLBase):
    def __init__(self):
        self.ls = LicenseSpecific()


    def process_fragmented_yaml(self, yaml_file, root_folder, path):
        """
        Returns fragmented functions.
        params:
        yaml_file: yaml file
        root_folder: root folder of the project
        path: path to the yaml file

        returns:
        fragmented function based features.
         : root_folder
         : path
         : runtime
         : plugins
         : number of functions
         : functions
         : single function
         : function path
         : function event
         : license (global)
        """
        runtime = self.get_runtime(yaml_file)
        provider_name = self.get_provider_name(yaml_file)
        plugins = self.get_plugins(yaml_file)
        functions = self.get_functions(yaml_file)
        number_of_functions = self.get_number_of_functions(yaml_file)
        # if(runtime == "Null"):
        #     print("Runtime is Null")
        res = []
        license = self.ls.checkLicense(path)[1]
        for function in functions:
            function_path = self.get_function_path(yaml_file, function)
            function_event = self.get_function_event(yaml_file, function)
            
            res.append((root_folder, path,runtime, provider_name, plugins, number_of_functions, functions, function, function_path, function_event, license))

        return res

    def process_yaml(self, yaml_file, root_folder, path) -> Tuple:
        """
        Returns tuple with all required parameters.
        params:
        yaml_file: yaml file
        root_folder: root folder of the project
        path: path to the yaml file

        returns:
        global features.
            : root_folder
            : path
            : runtime
            : provider_name
            : plugins
            : functions
            : number_of_functions
            : license
        """
        runtime = self.get_runtime(yaml_file)
        provider_name = self.get_provider_name(yaml_file)
        plugins = self.get_plugins(yaml_file)
        functions = self.get_functions(yaml_file)
        number_of_functions = self.get_number_of_functions(yaml_file)
        return (root_folder,path,runtime,provider_name,plugins,functions,number_of_functions, self.ls.checkLicense(path)[1])


    def get_functions(self, yaml_file) -> Set[str]:
        """
        Returns the functions mentioned in yaml file.
        """
        try:
            if("functions" in yaml_file):
                return set(yaml_file["functions"])
            else:
                return set()
        except Exception as e:
            return set()


    def get_number_of_functions(self,yaml_file) -> int:
        """
        Returns the number of functions mentioned in yaml file.
        """
        try:
            if("functions" in yaml_file):
                return len(yaml_file["functions"])
            else:
                return 0
        except Exception as e:
            return 0

    def get_function_path(self,yaml_file, function_name) -> str:
        """
        Returns function path for a specific serverless.yaml file

        Remember to handle null case when used.

        #NOTE this function can be refactored by guarding if statements.
        """
        try:
            if("functions" in yaml_file):
                if(function_name in yaml_file["functions"]):
                    path= yaml_file["functions"][function_name]["handler"]
                    if "." in path:
                        # TODO check right side of . is a function name and not a path.
                        if("\\" in path.split(".")[1]) or ("/" in path.split(".")[1] or (")" in path.split(".")[1])):
                            return "Null"
                        else:
                            return path
                    else:
                        return "Null"
                    # returns filepath.function_name
                else:
                    return "Null"
            else:
                return "Null"
        except Exception as e:
            return "Null"

    def get_function_event(self, yaml_file, function_name, verbose=False) -> List:
        """
        Returns event used by a function.

        Handles two types of event responses.

            a["functions"]["luckyNumber"]["events"]
                ['alexaSkill']

            a["functions"]["transcribe"]["events"]
                [{'s3': {'bucket': '${self:provider.environment.S3_AUDIO_BUCKET}', 'event': 's3:ObjectCreated:*'}}]
        
        """
        # TODO handle different kind of responses, e.g. 9 and 45
        # 
        try:
            if("functions" in yaml_file):
                if(function_name in yaml_file["functions"]):
                    temp = yaml_file["functions"][function_name]["events"]
                    # print(temp)
                    if(type(temp[0]) == str):
                        return list(temp.keys())
                    elif(type(temp[0]) == dict):
                        event_list = []
                        if not verbose:
                            for i in temp:
                                # print(i.keys())
                                event_list.extend(list(i.keys()))
                        else:
                            for i in temp:
                                for j in i.keys():
                                    event_list.append(i[j]['event'])
                        return event_list
                    else:
                        return []
                else:
                    return []
            else:
                return []
        except Exception as e:
            return []

    def get_provider_name(self,yaml_file) -> str:
        """
        Returns the provider name mentioned in yaml file.
        """
        try:
            if("provider" in yaml_file):
                if("name" in yaml_file["provider"]):
                    return yaml_file["provider"]["name"]
                else:
                    return "Null"
            else:
                return "Null"
        except Exception as e:
            return "Null"

    def get_runtime(self,yaml_file) -> str:
        """
        Returns the runtime mentioned in yaml file.
        """
        try:
            if("provider" in yaml_file):
                if "runtime" in yaml_file["provider"]:
                    return yaml_file["provider"]["runtime"]
                else:
                    return "Null"
            else:
                return "Null"
        except Exception as e:
            return "Null"

    def get_plugins(self, yaml_file) -> Set[str]:
        """
        Returns the plugins mentioned in yaml file.
        """
        try:
            if("plugins" in yaml_file):
                return set(yaml_file["plugins"])
            else:
                return set()
        except Exception as e:
            return set()



class CodeSpecific:
    def __init__(self, root_path="storage/nfs/", repo_path="serverless"):
        self.storage_path = root_path+repo_path+"/"
        self.code_extensions = ["py", "js", "ts", "java", "go", "cpp", "cs", "c", "swift"]
        self.repos= self.get_folders()

    def map_code_to_repo_all(self):
        """
        Returns all code files associated with a repo.
        """
        data = []
        for repo in self.repos:
            
            for ex in self.code_extensions:
                data = data + self.get_code_files(repo, language=ex)
            
        return data

    
    def get_file_name(self, path):
        """
        Returns the file name we have to find for specific fragment.
        params:
            path: path to the file i.e. handler.main something like that.
        """
        if(path == "Null"):
            return "Null"
        
        path = path.split(".")
        if(len(path)>1 and "::" in path[1]):
            # Handling cases like EvidenceApi::EvidenceApi.LambdaHandler::Function
            return path[1].split("::")[0]

        if(len(path) == 1):
            # src/get-by-hid/somefolder/somefile would work as in the end, this will be added as path...
            return path[0]

        if("/" in path[-2] or "\\" in path[-2]):
            # Redundant. Can be removed.
            return path[-2]
        
        return path[-2]
        
        # print("FAILED ",path)
        # return "Null"

    def get_language(self, code_file_path):
        """
        Returns the language of the code file.
        """
        try:
            return code_file_path.split(".")[-1]
        except Exception as e:
            return "Null"

    def get_specific_code_file(self, root, filename):
        """
        Returns file location for a file from root to filename.
        """
        if(filename == "Null"):
            return "Null"
        else:
            # TODO needs edge case handling.
            # Remove test , .html , .md etc paths and return a single path instead of list.

            res = find_files(root, filename+".*", depth=3)

            # res = glob.glob(root + "/**/" + filename +".*", recursive=True)
            if(len(res) == 0):
                return "Null"
            elif(len(res) == 1):
                return res[0]
            elif(len(res) > 1):
                nr = ""
                for i in res:
                    if(".html" in i or ".md" in i or ".txt" in i or ".json" in i or ".yaml" in i or ".yml" in i or ".xml" in i or "test" in i):
                        continue
                    else:
                        nr = i
                        break
                return nr


    def get_code_tuple(self, root, path, function_path):
        """
        Returns tuple of code file features.

        :params:
            root: root directory of the repo.
            path: path to the serverless file.
            function_path: path to the function i.e. handler.main something like that.

        :returns:
            (root,path, function_path, code_file_path ,language)
            : root: root directory of the repo.
            : path: path to the serverless file.
            : function_path: path to the function i.e. handler.main
            : code_file_path: root/src/handler/main.py
            : language: language of the code file.
        """

        filename = self.get_file_name(function_path)
        code_file_path = self.get_specific_code_file(path, filename)
        language = self.get_language(code_file_path)

        return (root, path, function_path, code_file_path, language)


    def get_all_code_files(self, repo, depth=3):
        """
        Returns code files for all languages (defined in self.code_extensions) for a repo.

        :params:
            repo: repo name.
            depth: depth of the search.
        
        :returns:
            List of code file paths.
        """
        data = []
        for ex in self.code_extensions:
            data.extend(self.get_code_files(repo, language=ex, depth=depth))
        return data



    def get_code_files(self, repo, language="js", depth=3):
        """
        Return code files for a specific language.

        :params:
            repo: repo path
            language: language extension
        
        :returns:
            list of code files path
        """
        data = find_files(repo, "*."+language, depth=depth)
        # data = glob.glob(os.path.join(repo,"**/*.{}".format(language)), recursive=True)
        nd = []
        for i in data:
            nd.append(tuple([repo,i]))

        return nd

    def get_folders(self, root_path=None):
        """
        Returns all folders/files in a particular path.
        """
        root_path = root_path or self.storage_path
        return glob.glob(root_path+"*")



# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


#TODO Proper documentation.
def remove_empty_serverless(base_path="storage/nfs/serverless/", repo_path="200-400/"):
    """
    Removes empty serverless files.
    base path is base path
    repo path is repo path
    """
    data = []
    
    repositories = glob.glob(base_path+repo_path+"*")
    for repo in repositories:
        try:
            sz = (get_size(repo) / (10**7))

            if(sz > 45):
                print("HERE", repo, sz)
                shutil.rmtree(repo)
                print("REPO BEING REMOVED: ", repo, " SIZE: ", sz)
        except Exception as e:
            continue


    # REMOVE UNLICENSED REPOSITORIES
    ls = LicenseSpecific(root_path=base_path, repo_path=repo_path)
    ls.remove_unlicensed_folders(root_path=base_path+repo_path)

    ft = FileTraversal(root_path=base_path, repo_path=repo_path)
    to_remove_root = set()
    to_keep_root = set()
    to_remove_directory = set()
    serverless_files = ft.get_specific_files()

    syml = YMLSpecific()

    for f in serverless_files:
        try:
            yf = syml.get_yaml_file(f)
            num_fun = syml.get_number_of_functions(yf)
            fun = syml.get_functions(yf)
            for i in fun:
                fp = syml.get_function_path(yf,i)
                if("$" in fp or fp == "Null"):
                    print(syml.get_function_path(yf,i), "Contains $ or is Null")
                    num_fun = 0
            
            if(num_fun != 0):
                # Keep Folders
                to_keep_root.add(ft.get_root_folder(f)) 
                # non_fragmented_tuples.append(syml.process_yaml(yf))
                # fragmented_tuples.extend(syml.process_fragmented_yaml(yf))
            else:
                # Remove the root folder (Maybe)
                to_remove_directory.add(ft.get_serverless_path(f))
                to_remove_root.add(ft.get_root_folder(f))
        except Exception as e:
            print("Error for file : ", f)
            print(e)
            to_remove_root.add(ft.get_root_folder(f))
            to_remove_directory.add(ft.get_serverless_path(f))

    print("Removing : ", len(to_remove_directory))

    # Remove useless files.
    for dir in to_remove_directory:
        print("Removing directory : ", dir)
        try:
            shutil.rmtree(dir)
        except Exception as e:
            continue

    to_remove = to_remove_root - to_remove_root.intersection(to_keep_root)
    for dir in to_remove:
        print("Removing directory : ", dir)
        try:
            shutil.rmtree(dir)
        except Exception as e:
            continue
    

    ### MORE TO REMOVE: test, examples, .html, .css, .jpg, .png
    repositories = glob.glob(base_path+repo_path+"*")
    filter_keys = ["test", "examples","docs","Example","Test","Docs","example", "tests", "Examples","node_modules"]
    for key in filter_keys:
        for repo in repositories:
            files = find_files(repo, key, depth=7)
            for f in files:
                try:
                    shutil.rmtree(f)
                    print("Removed : ", f)
                except Exception as e:
                    continue

    filter_files = ["*.ipynb", "*.html", "*.css", "*.jpg", "*.png", "*.xml", ".gif","package-lock.json"]
    for key in filter_files:
        for repo in repositories:
            files = find_files(repo, key, depth=7)
            for f in files:
                try:
                    os.remove(f)
                    print("Removed : ", f)
                except Exception as e:
                    continue

    # TODO remove directories with no config files.
    # Do this for every to remove...

if __name__ == '__main__':
    # REMOVE ALL FOLDERS THAT DO NOT HAVE LICENSE
    remove_empty_serverless(base_path="storage/nfs/serverless_t/", repo_path="")
    ls = LicenseSpecific(repo_path="serverless_t")
    # ls.remove_non_licensed_folders()
    ls.remove_unlicensed_folders()
    
    
    
    ft = FileTraversal(repo_path="serverless_t")
    to_remove_root = set()
    to_keep_root = set()
    to_remove_directory = set()
    print("HERE")
    serverless_files = ft.get_specific_files()
    syml = YMLSpecific()
    fragmented_tuples = []
    non_fragmented_tuples = []
    # TODO check if it contains functions.
    for f in serverless_files:
        try:
            yf = syml.get_yaml_file(f)
            num_fun = syml.get_number_of_functions(yf)
            fun = syml.get_functions(yf)
            for i in fun:
                fp = syml.get_function_path(yf,i)
                if("$" in fp or fp == "Null"):
                    print(syml.get_function_path(yf,i), "Contains $ or is Null")
                    num_fun = 0
            
            if(num_fun != 0):
                # Keep Folders
                to_keep_root.add(ft.get_root_folder(f)) 
                # non_fragmented_tuples.append(syml.process_yaml(yf))
                # fragmented_tuples.extend(syml.process_fragmented_yaml(yf))
            else:
                # Remove the root folder (Maybe)
                to_remove_directory.add(ft.get_serverless_path(f))
                to_remove_root.add(ft.get_root_folder(f))
        except Exception as e:
            print("Error for file : ", f)
            print(e)
            to_remove_root.add(ft.get_root_folder(f))
            to_remove_directory.add(ft.get_serverless_path(f))

    

    for dir in to_remove_directory:
        print("Removing directory : ", dir)
        try:
            shutil.rmtree(dir)
        except Exception as e:
            continue

    to_remove = to_remove_root - to_remove_root.intersection(to_keep_root)
    for dir in to_remove:
        print("Removing directory : ", dir)
        try:
            shutil.rmtree(dir)
        except Exception as e:
            continue

    # Get Serverless.yaml cleaned files.

    serverless_files = ft.get_specific_files()

    for f in serverless_files:
        yf = syml.get_yaml_file(f)
        non_fragmented_tuples.append(syml.process_yaml(yf,root_folder=ft.get_root_folder(f), path=ft.get_serverless_path(f)))
        res = syml.process_fragmented_yaml(yf, root_folder=ft.get_root_folder(f), path=ft.get_serverless_path(f))
        if(res!= None):

            fragmented_tuples.extend(res)


    cs = CodeSpecific()
    res= []
    for i in fragmented_tuples:
        # GET code features.
        res.append(cs.get_code_tuple(i[0], i[1], i[-3]))


    ft.save_result(res,"code_files.csv", features=["repo","file_path", "function_path", "code_file_path", "language"])


    # SAVE FRAGMENTED AND NON FRAGMENTED TUPLES
    # FOR RECOMMENDAION, WE will use only projects with active license.
    # REMOVE ALL NON LICENSED PROJECTS.

    ft.save_result(non_fragmented_tuples, "non_fragmented_tuples.csv")
    ft.save_result(fragmented_tuples, "fragmented_tuples.csv", features=["root_folder", "path", "runtime", "plugins", "number_of_functions", "functions", "single_function","function_path", "function_event","license"])



