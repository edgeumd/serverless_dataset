from github import Github

from configparser import ConfigParser
import os
import re
import threading as th
import datetime
import time


# Loading Token from config file
config = ConfigParser()
config.read('config.ini')
token = config.get('auth', 'token')
url = "https://api.github.com"


# Setting topic to serverless
topic = "serverless-framework"

# Defining Start Date and End Date
start_date = "2019-10-28"
end_date = "2020-12-31"

g = Github(token)


def get_repo(day, topic="serverless"):
    """
    Returns a list of full name repositories for a given day and a given topic.

    :param day: string in format YYYY-mm-dd (Note: for range: set day as start_date..end_date)
    :param topic: string

    returns: list of strings (full name of repositories)
    """
    repositories = g.search_repositories(
        query="topic:{} created:{}".format(topic, day))
    repos = []
    for rep in repositories:
        repos.append(rep.full_name)
    return repos

# Test for get_repo
# print(get_repo("2020-10-10",topic="serverless-framework"))


def get_date_list(start_date, end_date):
    """
    Makes a list of dates from start_date to end_date
    includes start_date and end_date

    :param start_date: string in format YYYY-mm-dd
    :param end_date: string in format YYYY-mm-dd

    returns: list of strings in format YYYY-mm-dd
    """
    date_list = []
    start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    while start_date <= end_date:
        date_list.append(start_date.strftime("%Y-%m-%d"))
        start_date += datetime.timedelta(days=1)
    return date_list


# Test for get_date_list
# print(get_date_list("2020-01-01", "2020-03-03"))


def get_repos_per_day(start_date, end_date, topic="serverless"):
    """
    Returns a list of repositories for a given day and a given topic.

    :param start_date: string in format YYYY-mm-dd
    :param end_date: string in format YYYY-mm-dd
    :param topic: string

    returns: list of strings (full name of repositories)
    """
    date_list = get_date_list(start_date, end_date)
    repos_per_day = []
    try:
        for day in date_list:
            print("Getting repositories for {}".format(day))
            repos_per_day.extend(get_repo(day, topic))
            print("Repo Size: ", len(repos_per_day))
            time.sleep(5)
    except Exception as e:
        print("Error: ", e)
    return repos_per_day


# Test for get_repos_per_day
repos = get_repos_per_day(start_date, end_date, topic)
repos = set(repos)
print("Final Total: ", len(repos))
print("Writing to file...")
with open("topic_{}.txt".format(topic), "a") as f:
    for repo in repos:
        f.write(repo + "\n")

print("Done!")
