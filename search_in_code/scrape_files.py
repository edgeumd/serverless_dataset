from github import Github
from github import GithubException

from configparser import ConfigParser
import os
import re
import threading as th
import datetime
import time


# Loading Token from config file
config = ConfigParser()
config.read('../config.ini')
token = config.get('auth', 'token')
url = "https://api.github.com"


search_in_code = "Type: AWS::Serverless::StateMachine"
extension = "yml"


