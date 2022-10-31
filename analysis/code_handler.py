#TODO Load code files for recommendation engine.
# Process code and extract keyword features.
# Make Python AST Parser that returns a list.
# KNOWLEDGE GAP DEEP LEARNING - - This point.

import glob
import remove_empty_base as rm
import re
import itertools


from gensim.models import Word2Vec
from gensim.models import KeyedVectors

from collections import Counter
from math import sqrt


import pygments as p

from pygments.lexers.python import PythonLexer
from pygments.lexers.c_cpp import CLexer
from pygments.lexers.c_cpp import CppLexer
from pygments.lexers.javascript import JavascriptLexer
from pygments.lexers.javascript import TypeScriptLexer
from pygments.lexers.dotnet import CSharpLexer
from pygments.lexers.jvm import JavaLexer
from pygments.lexers.go import GoLexer

# Classes Defined:
# 1. Word2VecModelTrainer -> Trains a Word2Vec model.
# 2. Word2VecModelWorker -> Use Word2Vec
# 3. CodePreprocessor -> Load Code, 
#       Remove Structural Elements, Tokenize.






class Word2VecModelTrainer:
    """
    Word2Vec Features. Basic Process: call train, call load.
    """

    def __init__(self, sentences=[], path="word2vec.wordvectors"):
        """
        params: 
        sentences: list of sentences (default: [])
        NOTE sentences should be like this 
        [["i", "am", "sarvesh"],["some","other","sentence"]]
        """

        self.sentences = sentences
        self.path = path

    def train(self, size=100, window=9):
        """
        Trains the model on sentences
        params: 
        size: size of the vector
        window: window size
        """

        model = Word2Vec(sentences=self.sentences, vector_size=size,
                         window=window, min_count=1, workers=4)
        model.save("word2vec.model")

        model.train(self.sentences,
                    total_examples=model.corpus_count, epochs=30)
        word_vectors = model.wv
        word_vectors.save(self.path)
        return model, word_vectors

    def load_trained(self, path=""):
        """
        Loads the trained model
        """
        if(path != ""):
            load_path = path
        else:
            load_path = self.path
        wv = KeyedVectors.load(load_path, mmap='r')
        return wv


class Word2VecModelWorker:
    def __init__(self, kv) -> None:
        self.kv = kv

    def get_vector(self, word):
        """
        Returns the vector of the word
        """
        if(self.kv.has_index_for(word)):
            return self.kv.get_vector(word)
        else:
            return self.kv.get_vector("the")

    def parse_code_basic(self, tokenized_code):
        """
        Returns a list of vectors for each word in the code

        Although we are keeping information, its tough to make sense of it.
        due to different sizes of vectors produced.

        ["i","am","sarvesh"]

        """
        return [self.get_vector(token) for token in tokenized_code]

    def parse_code_adder(self, tokenized_code):
        """
        Returns a list of vectors after adding each of them together.

        We are losing lots of information here.
        """
        return sum(self.parse_code_basic(tokenized_code)) / len(tokenized_code)

    def direct_similarity(self, tok_sent1, tok_sent2):
        """
        Returns the similarity between two sentences.

        :params:
            tok_sent1: list of tokens
            tok_sent2: list of tokens

        :return:
            similarity: float 0.0 means same sentence, the lower the better.
        """
        return self.kv.wmdistance(tok_sent1, tok_sent2)


class WordHandler:
    """
    Class file to handle different kinds of words and files.
    """
    def camel_case_remove(self,word):
        """
        Splits camel case words
        """
        pat = r"[a-zA-Z](?:[a-z0-9.]+|[A-Z]*(?=[A-Z]|$))"
        res = " ".join(re.findall(pat, word))
        return res

    def dash_remove(self,word):
        """
        - with " "
        """
        res = re.sub("-", " ", word)
        return res

    def underscore_remove(self,word):
        """
        Splits words with underscores
        """
        res = re.sub("_", " ", word)
        return res

    def remove_chars(self,word):
        """
        Removes chars from the word
        """
        return word.replace(":", " ").replace(";", " ").replace("/", " ")

    def config_word_processor(self,word):
        """
        Processes the word
        # TODO refactor this... Can be done in a better way.
        """
        word = self.remove_chars(word)
        word = self.camel_case_remove(word)
        word = self.underscore_remove(word)
        word = self.dash_remove(word)
        return word.split(" ")

    def remove_structure(self, conf):
        """
        Removes structure from the conf
        """
        return re.sub("{|}|\(|\)", " ", conf)

    def remove_new_lines(self,conf):
        """
        Removes new lines from the config
        """
        return re.sub("\n", " ", conf)

    def remove_tabs(self, conf):
        """
        Removes tabs from the conf
        """
        return re.sub("\t", " ", conf)

    def replace_comma_by_space(self,conf):
        """
        Replaces commas by spaces
        """
        return re.sub(",", " ", conf)

    def remove_repeated_spaces(self,conf):
        """
        Removes repeated spaces from the config
        """
        return re.sub(" +", " ", conf)

    def remove_comments(self,conf):
        """
        Removes comments from the file.
        """
        pat = r'(#.*)(\n|$)|(\/\/.*?)(\n|$)'
        return re.sub(pat," ", conf)

    def remove_slashes(self, conf):
        """
        Removes slashes from the config
        """
        return conf.replace("/", " ").replace("\\", " ")

    def remove_comment_chars(self, conf):
        """
        Removes characters like #, //, etc...
        """
        return conf.replace("#", " ").replace("//", " ")




class CodePreprocessor(WordHandler):
    def __init__(self, code_tuples) -> None:
        # Can be refractored to do in time n instead of 4n time.
        code_tuples.pop(0)
        self.paths = [i[3] for i in code_tuples]
        self.language = [i[4] for i in code_tuples]
        self.repo_name = [i[0] for i in code_tuples]
        self.serverless_path = [i[1] for i in code_tuples]
        self.mapvals = {
            "py": PythonLexer(),
            "java": JavaLexer(),
            "c": CLexer(),
            "cpp": CppLexer(),
            "cs": CSharpLexer(),
            "js": JavascriptLexer(),
            "ts": TypeScriptLexer(),
            "go": GoLexer(),
        }



    def get_function_names(self, path):
        """
        Given a path, returns the function names in the file.
        """
        lang = self.code_lang(path)

        if lang not in self.mapvals:
            return []

        lexer = self.mapvals[lang]

        with open(path, "r") as f:
            code = f.read()

        # Get function names
        genobj = p.lex(code, lexer=lexer)
        function_names = [i[1] for i in genobj if "Token.Name" in str(i[0])]
        # Process these function names.
        function_names = set(function_names)
        new_function_names = []
        for name in function_names:
            new_function_names.extend(self.config_word_processor(name))
        return new_function_names

    

    def load_raw_code(self, path):
        """
        Loads the raw code from the path
        """
        if path == "Null":
            return ""
        with open(path, "r") as f:
            return f.read()



    def code_lang(self, path):
        """
        Returns language of code
        """
        return path.split(".")[-1]

    def load_code(self, path):
        """
        Loads code and performs preprocessing.
        Specifically we aim to remove the structure information from files.
        """
        with open(path,"r") as f:
            code = f.read()
        code = code.strip()
        code = self.remove_new_lines(code)
        code = self.remove_tabs(code)
        code = self.remove_structure(code)
        code = code.replace(":"," ")
        code = code.replace(";"," ")
        code = re.sub(" +"," ", code)
        code = code.strip()
        return code

    def process_readme(self, readme):
        """
        Processes the readme file.
        Simply ignores everything except of alpha chars.
        
        """
        return " ".join(re.findall("[a-zA-Z]+", readme))

    def process_config(self, conf):
        """
        Code to properly process config files.
        """
        # Note comments remove first, coz to detect, \n required.
        # conf = self.remove_comments(conf) #Comments might provide useful information Instead remove #,//,etc.
        conf = self.remove_comment_chars(conf)
        conf = self.remove_new_lines(conf)
        conf = self.replace_comma_by_space(conf)
        conf = self.remove_repeated_spaces(conf)
        conf = self.remove_slashes(conf)
        conf = conf.split(" ")
        nfcf = []
        for word in conf:
            nfcf = nfcf + self.config_word_processor(word)

        return nfcf

    def similarity(self,words1, words2):
        """
        Calculates the cosine similarity between two lists of words
        params:
        words1: list of words
        words2: list of words

        returns:
        cosine similarity
        """
        words1, words2 = Counter(words1), Counter(words2)
        dot = sum(words1[k] * words2[k] for k in words1.keys() & words2.keys())
        norm1 = sqrt(sum(v**2 for v in words1.values()))
        norm2 = sqrt(sum(v**2 for v in words2.values()))
        return dot / ((norm1 * norm2) + 1)


    def tokenize(self, code):
        """
        Tokenizes the given code.
        Current implementation:
        1. Lowers the code.
        2. Replaces all string literals with <STRTOKEN>
        3. returns split code.
        """
        # base try:
        # simply convert to lowercase and split by space.
        # TODO replace import statements with <IMPORTTOKEN_TOKENVALUE>
        # VERIFY WITH LOAD_CODE
        code = code.lower()
        str_pattern = '\'(.*?)\'|"(.*?)"'
        code = re.sub(str_pattern, '<STRTOKEN>', code)

        return code.split(" ")

    def get_code_by_language(self, languages={"py","js","ts","java","c","cpp","go","swift","php","cs"}, paths=None):
        if(not paths):
            paths = self.paths
        code_list = []
        # for i in range(len(self.paths)):
        for i in range(len(paths)):
            if self.language[i] in languages:
                # code = self.load_code(self.paths[i])
                code = self.load_code(paths[i])
                code_list.append(code)
            else:
                print("Language not supported", self.code_lang(paths[i]), i)
                code_list.append("this is not a supported language")
        return code_list

    def read_file(self,filepath):
        """
        Reads file
        """
        with open(filepath, "r") as f:
            code = f.read()
        return code

    def find_all_similar(self,path_file_main, all_files_path, is_main_path=True):
        """
        Finds all similar files.

        params:
            path_file_main: path to the original file OR string but set is_main_path to False
            all_files_path: list of all files to be compared with the main file
            is_main_path: boolean to indicate if the path is a path or a string.
        returns:
            list of configs with similarity score.
        """
        if(is_main_path):
            cf1 = self.process_config(self.read_file(path_file_main))
        else:
            cf1 = self.process_config(path_file_main)
        res = dict(map(lambda x: (x, self.similarity(cf1, self.process_config(self.read_file(x)))), all_files_path))
        return res

    def find_similar_score(self, path1, path2, is_main_path=True,word_vec=False, is_path2=True):
        """
        Used for similarity between two files.

        params:
        path1: path to the first file OR string but set is_main_path to False
        path2: path to the second file.
        is_main_path: boolean to indicate if the path1 i.e. query is a path or a string.
        word_vec: boolean to indicate if the similarity is to be calculated using word vectors.
        is_path2: True i.e. path2 is a path. if false it is assumed as string.

        returns:
            similarity score.

        """
        if(is_main_path):
            cf1 = self.process_config(self.read_file(path1))
        else:
            cf1 = self.process_config(path1)

        if(word_vec):
            wt = Word2VecModelTrainer()
            wk = wt.load_trained()
            wm = Word2VecModelWorker(wk)
            if(not is_path2):
                # In this case path2 is not a path but raw string.
                return wm.direct_similarity(cf1, path2.split(" "))
            return wm.direct_similarity(cf1, self.read_file(path2).split(" "))
            # TODO : GET WORD VECTOR.
        res = self.similarity(cf1, self.read_file(path2).split(" "))
        return res

    def get_top_n(self, res, n=20,word_vec=False):
        """
        Returns the top n similar files.

        params:
        res: dict of similarity scores
        """
        if(word_vec):
            return sorted(res.items(), key=lambda x: x[1])[1:n]
        return sorted(res.items(), key=lambda x: x[1], reverse=True)[1:n]

    def get_repo(self, name, all_configs):
        """
        Returns the repo name from the config file
        """
        return [i for i in all_configs if name in i]
    

    def get_imports(self, language, code,remove_extra=False, skipsystem=False):
        """
        Returns the imports from the code file

        :params:
        language: language of the code
        code: code to be parsed
        remove_extra: boolean to indicate if extra imports should be removed.
        """

        # python_import_patterns = "import ([a-zA-Z.]+)|from ([a-zA-Z.]+) import ([a-zA-Z]+)|from ([a-zA-Z.]+) import ([a-zA-Z*]+)"
        python_import_patterns = "import ([a-zA-Z][a-zA-Z.]+)|from ([a-zA-Z][a-zA-Z.]+) import ([a-zA-Z][a-zA-Z]+)|from ([a-zA-Z][a-zA-Z.]+) import ([a-zA-Z*]+)"
        # TODO separate by . in python.

        # import defaultExport from "module-name";
        # import * as name from "module-name";
        # import { export1 } from "module-name";
        # import { export1 as alias1 } from "module-name";
        # import { export1 , export2 } from "module-name";
        # import defaultExport, * as name from "module-name";
        # import "module-name" ;
        # var promise = import("module-name");
        # import zip = require("./ZipCodeValidator"); #Typescript specific


        # # Not covered
        # import { export1 , export2 as alias2 , [...] } from "module-name";
        # import defaultExport, { export1 [ , [...] ] } from "module-name";
        
        # js_import_patterns = 'import ([\'"a-zA-Z_0-9]+) *= *require(\([\'"a-zA-Z.\/]+\))|import [\* a-zA-Z,]+ as [a-zA-Z-]+ from (["\'a-zA-Z-.\\\/]+)|var [a-zA-Z ]+=[ ]*import([\(\'"a-zA-Z-".\\\/)]+)|const [a-zA-Z ]+=[ ]*import([\(\'"a-zA-Z-".\\\/)]+)|import ([\'"a-zA-Z-.\\\/]+) from ([\'"a-zA-Z-.\\\/]+)|import {([a-zA-Z0-9 ,]+)} from ([\'"a-zA-Z-.\\\/]+)|import ([\'"a-zA-Z-.\\\/]+)|require\(([a-zA-Z_\-\'".\/]+)\)'

        js_import_patterns = 'import ([\'"a-zA-Z_0-9]+) *= *require(\([\'"a-zA-Z.]+\))|import [\* a-zA-Z,]+ as [a-zA-Z-]+ from (["\'a-zA-Z-.]+)|var [a-zA-Z ]+=[ ]*import([\(\'"a-zA-Z-".)]+)|const [a-zA-Z ]+=[ ]*import([\(\'"a-zA-Z-".)]+)|import ([\'"a-zA-Z-.]+) from ([\'"a-zA-Z-]+)|import {([a-zA-Z0-9 ,]+)} from ([\'"a-zA-Z-]+)|import ([\'"a-zA-Z-.]+)|require\(([a-zA-Z_\-\'"]+)\)'

        # NOTE JS AND TS SAME.


        # import package.name.ClassName;   // To import a certain class only
        # import package.name.*   // To import the whole package


        java_patterns = 'import (["a-zA-Z-.]+)'


        # Using same for c, cpp.
        c_patterns = '#include "([a-zA-Z_.0-9]+)"|#include <([a-zA-Z_.0-9]+)>|#import "([a-zA-Z_.0-9]+)"|#import <([a-zA-Z_.0-9]+)>'


        # Go
        go_patterns = 'import \(([ \n\ta-zA-Z"]+)\)|import (["a-zA-Z]+)'
        # NOTE remove \t and \n to space, make it single space then separate... "abc" "def"
        # import (
        #     "fmt"
        #     "math"
        #     "something"
        # )

        # import "fmt"
        # import "math"


        # Swift
        swift_patterns = 'import [a-zA-Z]+ ([a-zA-Z.]+)+|import ([a-zA-Z.]+)'
        # NOTE remember to separate by .


        # php
        php_patterns = "include (['\"a-zA-Z]+.php)|require (['\"a-zA-Z]+.php)"
        # <?php include 'noFileExists.php';

        # <?php require 'noFileExists.php';

        #cs
        cs_patterns = 'using [a-zA-Z_]+ *= *([a-zA-Z.]+)|using [a-zA-Z.]+ ([a-zA-Z.]+)|using ([a-zA-Z.]+)'

        # Dont forget to spearate by .
        dic = {
            "py": python_import_patterns,
            "js": js_import_patterns,
            "java": java_patterns,
            "c": c_patterns,
            "go": go_patterns,
            "swift": swift_patterns,
            "php": php_patterns,
            "cs": cs_patterns,
            "ts": js_import_patterns
        }
        if(language not in dic):
            return []


        # No matter the language find default apis for configs.
        services_pattern = '(s3)|(S3)|(dynamodb)|(DynamoDB)|(sns)|(sqs)|(ses)|(kinesis)|(kinesisanalytics)|(lex)|(polly)|(iot)|(iotanalytics)|azure\.storage\.(blob)|azure\/storage-(blob)|(cosmos)|(eventgrid)|(event-grid)|storage\.(queue)|storage-(queue)|(sendgrid)|(eventhub)|(event-hubs)|(botbuilder)|cognitiveservices\.(speech)|cognitiveservices-(speech)'
        # z = re.findall(dic[language], code)
        # z = self.remove_new_lines(z)
        # z = self.remove_tabs(z)
        # z = self.remove_repeated_spaces(z)
        pat = dic[language]

        # TODO : check if its alright or not. May have to change while testing.
        pat = pat + '|' + services_pattern
        res = re.findall(pat,code)
        res = [list(i) for i in res]
        # combine multiple tuples in one.
        res = set(itertools.chain(*res))
        if "" in res:
            res.remove("")
        # Remove " and ' from api names.
        res = [re.sub('"','',i) for i in res]
        res = [re.sub("'",'',i) for i in res]
        nres = []
        for i in res:
            if("," in i):
                nres.extend(i.split(","))
            else:
                nres.append(i)
        nnres = []
        for i in nres:
            if(" " in i):
                nnres.extend(i.split(" "))
            else:
                nnres.append(i)

        res = set(nnres)

        global_system_apis = {"re","random","time","flask","requests","urllib","List","S3","hashlib","uuid","datetime","boto","utils","urllib.parse","parse","setuptools","aws-sdk", "AWS","os","json","serverless-webpack","axios","success","failure","util","logging","aws-lambda","dotenv","sys","App","Context","..","get","io","typing","System","Flask","print","load","from","find","NotFound","in","OnInit","into","make","by","src","it","Any","first","pick","now","toArray",""}
        to_remove_names = {"abc","ABC","botocore.exceptions","nacl.signing","nacl.exceptions","ClientError","toLower","sendEvent","toPairs","","handler", "use", "string","the","a","is","fs","of","path","and","./index",".utils","./errors","app","to","as"}
        extra_to_remove = {"aws-sdk","AWS","boto","aws-lambda","AWSError","json","os","requests","logging","re","utils","config","dotenv","datetime","time","load","sys","typing","collections","argparse","functools","util","Dict",
        "request","importlib","shutil","itertools","defaultdict","List","symbol","math","setuptools","event","random","base","tuple","urllib","urllib.request","numpy","Tuple","Callable","Sequence","queue","Response","flask","Flask","axios","urllib.parse","ctypes","Callback","Optional","EmnistLinesDataset",
        "read","print","src.utils","pathlib","Dataset","urlparse","copy","Path","Axios","io",}
        res.difference_update(to_remove_names)
        if(skipsystem):
            global_system_apis = {""}
        res.difference_update(global_system_apis)
        if(remove_extra):
            res.difference_update(extra_to_remove)
        res = list(res)
        res = [i for i in res if len(i) > 1]

        return res


        
    




    

if __name__ == "__main__":
    fs = rm.FileTraversal(repo_path="serverless_t")
    res = fs.load_result(path="code_files.csv",type="code")
    serverless_configs=  glob.glob("storage/nfs/serverless_t/**/serverless.yml")
    # TODO Other repo configs
    # aws_configs
    # azure_configs
    # ibm_configs

    # process serverless configs
    selected_repo = "paulovfmarques"
    