#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on Mon Mar 20 00:10:29 2017
@author: Dinesh

About:
Python wrapper for the New York Times Archive API
https://developer.nytimes.com/article_search_v2.json
"""

import csv
import json
from newsapi import NewsAPI
import requests
import sys

class ArchiveAPI(object):

    def __init__(self, key=None):
        """
        Initializes the ArchiveAPI class. Raises an exception if no API key is given.
        :param key: New York Times API Key
        """
        self.key = key
        self.root = 'http://api.nytimes.com/svc/archive/v1/{}/{}.json?api-key={}'
        if not self.key:
            nyt_dev_page = 'http://developer.nytimes.com/docs/reference/keys'
            exception_str = 'Warning: API Key required. Please visit {}'
            raise NoAPIKeyException(exception_str.format(nyt_dev_page))

    def query(self, year=None, month=None, key=None,):
        """
        Calls the archive API and returns the results as a dictionary.

        :param key: Defaults to the API key used to initialize the ArchiveAPI class.
        """
        if not key: key = self.key
        if (year < 1882) or not (0 < month < 13):
            # currently the Archive API only supports year >= 1882
            exception_str = 'Invalid query: See http://developer.nytimes.com/archive_api.json'
            raise InvalidQueryException(exception_str)
        url = self.root.format(year, month, key)
        r = requests.get(url)
        return r.json()

class APIKeyException(Exception):

    def __init__(self, message):
        self.message = message

class InvalidQueryException(Exception):

    def __init__(self, message):
        self.message = message

key = '96af62a035db45bda517a9ca62a25ac3'
params = {}
api = NewsAPI(key)
sources = api.sources(params)
articles = api.articles(sources[0]['id'], params)
reload(sys)
sys.setdefaultencoding('utf8')
api = ArchiveAPI('0ba6dc04a8cb44e0a890c00df88c393a')
months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
years = [2016, 2015, 2014, 2013, 2012, 2011, 2010, 2009, 2008, 2007]
for year in years:
    for month in months:
        mydict = api.query(year, month)
        file_str = 'data/nytimes/' + str(year) + '-' + '{:02}'.format(month) + '.json'
        with open(file_str, 'w') as fout:
            json.dump(mydict, fout)
        fout.close()
