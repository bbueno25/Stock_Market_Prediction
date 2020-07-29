"""
DOCSTRING
"""
import datetime
import csv
import json
import matplotlib.pyplot as pyplot
import newsapi
import nltk
import numpy
import pandas
import requests
import sys
import treeinterpreter
import unicodedata

class APIKeyException(Exception):
    """
    @author: Dinesh
    """
    def __init__(self, message):
        self.message = message

class ArchiveAPI:
    """
    :param key: New York Times API Key
   
    @author: Dinesh
    """
    def __init__(self, key=None):
        self.key = key
        self.root = 'http://api.nytimes.com/svc/archive/v1/{}/{}.json?api-key={}'
        if not self.key:
            nyt_dev_page = 'http://developer.nytimes.com/docs/reference/keys'
            exception_str = 'Warning: API Key required. Please visit {}'
            raise NoAPIKeyException(exception_str.format(nyt_dev_page))

    def query(self, year=None, month=None, key=None):
        """
        Calls the archive API and returns the results as a dictionary.

        :param key: Defaults to the API key used to initialize the ArchiveAPI class.

        NOTE: Currently the Archive API only supports year >= 1882
        """
        if not key: key = self.key
        if (year < 1882) or not (0 < month < 13):
            exception_str = 'Invalid query: See http://developer.nytimes.com/archive_api.json'
            raise InvalidQueryException(exception_str)
        url = self.root.format(year, month, key)
        r = requests.get(url)
        return r.json()

class CollectData:
    """
    Python wrapper for the New York Times Archive API.
    https://developer.nytimes.com/article_search_v2.json

    @author: Dinesh
    """
    def __init__(self):
        key = '96af62a035db45bda517a9ca62a25ac3'
        params = {}
        api = newsapi.NewsAPI(key)
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

class InvalidQueryException(Exception):
    """
    @author: Dinesh
    """
    def __init__(self, message):
        self.message = message

class PrepData:
    """
    Analysing & Filteration

    @author: Dinesh
    """
    def __init__(self):
        with open('data/djia_data.csv', 'rb') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            data_list = list(spamreader)
        header = data_list[0]
        data_list = data_list[1:]
        data_list = numpy.asarray(data_list)
        selected_data = data_list[:, [0, 4, 6]]
        df = pandas.DataFrame(
            data=selected_data[0:,1:], index=selected_data[0:,0],
            columns=['close', 'adj close'], dtype='float64')
        df1 = df
        idx = pandas.date_range('12-29-2006', '12-31-2016')
        df1.index = pandas.DatetimeIndex(df1.index)
        df1 = df1.reindex(idx, fill_value=numpy.NaN)
        #df1.count()
        interpolated_df = df1.interpolate()
        interpolated_df.count()
        interpolated_df = interpolated_df[3:]

    def __call__(self):
        print(count_articles_filtered)
        print(count_total_articles)
        print(count_main_not_exist)
        print(count_unicode_error)
        ## putting all articles if no section_name or news_desk not found
        for date, row in interpolated_df.T.iteritems():
            if len(interpolated_df.loc[date, 'articles']) <= 400:
                #print(interpolated_df.loc[date, 'articles'])
                #print(date)
                month = date.month
                year = date.year
                file_str = 'data/nytimes/' + str(year) + '-' + '{:02}'.format(month) + '.json'
                with open(file_str) as data_file:
                    NYTimes_data = json.load(data_file)
                count_total_articles = count_total_articles + len(NYTimes_data["response"]["docs"][:])
                interpolated_df.set_value(date.strftime('%Y-%m-%d'), 'articles', '')
                for i in range(len(NYTimes_data["response"]["docs"][:])):
                    try:
                        articles_dict = {
                            your_key: NYTimes_data["response"]["docs"][:][i][your_key]
                            for your_key in dict_keys}
                        articles_dict['headline'] = articles_dict['headline']['main']
                        #articles_dict['headline'] = articles_dict['lead_paragraph']
                        pub_date = try_parsing_date(articles_dict['pub_date'])
                        #print('article_dict: ' + articles_dict['headline'])
                        if date.strftime('%Y-%m-%d') == pub_date:
                            interpolated_df.set_value(
                                pub_date, 'articles',
                                interpolated_df.loc[pub_date, 'articles'] + 
                                '. ' + articles_dict['headline'])
                    except KeyError:
                        print('key error')
                        #print(NYTimes_data["response"]["docs"][:][i])
                        #count_main_not_exist += 1
                        pass
                    except TypeError:
                        print("type error")
                        #print(NYTimes_data["response"]["docs"][:][i])
                        #count_main_not_exist += 1
                        pass

        #print(count_articles_filtered) # 44077
        #print(count_total_articles) # 1073132
        # filtering the whole data for a year
        #filtered_data = interpolated_df.ix['2016-01-01':'2016-12-31']
        #filtered_data.to_pickle('data/pickled_ten_year_all.pkl')
        interpolated_df.to_pickle('data/pickled_ten_year_filtered_lead_para.pkl')
        interpolated_df.to_csv(
            'data/sample_interpolated_df_10_years_filtered_lead_para.csv',
            sep='\t', encoding='utf-8')
        dataframe_read = pandas.read_pickle('data/pickled_ten_year_filtered_lead_para.pkl')
        # filtered_data = interpolated_df.ix['2016-01-01':'2016-12-31']
        # NYTimes_data["response"]["docs"][1:2][:]['headline']['main']
        # NYTimes_data["response"]["docs"][1:2][0]['pub_date']
        #    articles_dict = {
        #    your_key: NYTimes_data["response"]["docs"][:][i][your_key]
        #    for your_key in dict_keys}
        #    try:
        #        articles_dict['headline'] = articles_dict['headline']['main']
        #    except KeyError:
        #        count_main_not_exist += 1
        #        pass
        #    except TypeError:
        #        count_main_not_exist += 1
        #        pass
        # find articles with less number of articles
        # for date, row in interpolated_df.T.iteritems():
        #     if len(interpolated_df.loc[date, 'articles']) < 300:
        #         print(interpolated_df.loc[date, 'articles'])
        #         print(date)

    def prepare_data(self):
        date_format = ["%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S+%f"]
        def try_parsing_date(text):
            for fmt in date_format:
                #return datetime.datetime.strptime(text, fmt)
                try:
                    return datetime.datetime.strptime(text, fmt).strftime('%Y-%m-%d')
                except ValueError:
                    pass
            raise ValueError('no valid date format found')

        years = [2016, 2015, 2014, 2013, 2012, 2011, 2010, 2009, 2008, 2007]
        months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        dict_keys = ['pub_date', 'headline']
        articles_dict = dict.fromkeys(dict_keys)
        type_of_material_list = [
            'blog', 'brief', 'news', 'editorial', 'op-ed', 'list', 'analysis']
        section_name_list = [
            'business', 'national', 'world', 'u.s.' , 'politics',
            'opinion', 'tech', 'science',  'health']
        news_desk_list = [
            'business', 'national', 'world', 'u.s.' , 'politics',
            'opinion', 'tech', 'science',  'health', 'foreign']
        current_date = '2016-01-01'

        #years = [2015]
        #months = [3]
        current_article_str = ''
        interpolated_df["articles"] = ''
        count_articles_filtered = 0
        count_total_articles = 0
        count_main_not_exist = 0
        count_unicode_error = 0
        count_attribute_error = 0
        for year in years:
            for month in months:
                file_str = 'data/nytimes/' + str(year) + '-' + '{:02}'.format(month) + '.json'
                with open(file_str) as data_file:
                    NYTimes_data = json.load(data_file)
                count_total_articles = count_total_articles + len(NYTimes_data["response"]["docs"][:])
                for i in range(len(NYTimes_data["response"]["docs"][:])):
                    try:
                        if any(
                            substring in NYTimes_data["response"]["docs"][:][i]['type_of_material'].lower()
                            for substring in type_of_material_list):
                            if any(
                                substring in NYTimes_data["response"]["docs"][:][i]['section_name'].lower()
                                for substring in section_name_list):
                                #count += 1
                                count_articles_filtered += 1
                                #print('i: ' + str(i))
                                articles_dict = {
                                    your_key: NYTimes_data["response"]["docs"][:][i][your_key]
                                    for your_key in dict_keys}
                                articles_dict['headline'] = articles_dict['headline']['main']
                                #articles_dict['headline'] = articles_dict['lead_paragraph']
                                date = try_parsing_date(articles_dict['pub_date'])
                                #print('article_dict: ' + articles_dict['headline'])
                                if date == current_date:
                                    current_article_str = \
                                        current_article_str + '. ' + articles_dict['headline']
                                else:
                                    interpolated_df.set_value(
                                        current_date, 'articles',
                                        interpolated_df.loc[current_date, 'articles']
                                        + '. ' + current_article_str)
                                    current_date = date
                                    #interpolated_df.set_value(date, 'articles', current_article_str)
                                    #print(str(date) + current_article_str)
                                    current_article_str = articles_dict['headline']
                                # for last condition in a year
                                if (date == current_date) and (
                                    i == len(NYTimes_data["response"]["docs"][:]) - 1):
                                    interpolated_df.set_value(date, 'articles', current_article_str)
                    # exception for section_name or type_of_material absent
                    except AttributeError:
                        #print('attribute error')
                        #print(NYTimes_data["response"]["docs"][:][i])
                        count_attribute_error += 1
                        # if article matches news_desk_list if none section_name found
                        try:
                            if any(
                                substring in NYTimes_data["response"]["docs"][:][i]['news_desk'].lower()
                                for substring in news_desk_list):
                                    #count += 1
                                    count_articles_filtered += 1
                                    #print('i: ' + str(i))
                                    articles_dict = {
                                        your_key: NYTimes_data["response"]["docs"][:][i][your_key]
                                        for your_key in dict_keys}
                                    articles_dict['headline'] = articles_dict['headline']['main']
                                    #articles_dict['headline'] = articles_dict['lead_paragraph']
                                    date = try_parsing_date(articles_dict['pub_date'])
                                    #print('article_dict: ' + articles_dict['headline'])
                                    if date == current_date:
                                        current_article_str = current_article_str \
                                            + '. ' + articles_dict['headline']
                                    else:
                                        interpolated_df.set_value(
                                            current_date, 'articles',
                                            interpolated_df.loc[current_date,'articles']
                                            + '. ' + current_article_str)
                                        current_date = date
                                        #interpolated_df.set_value(
                                        #    date, 'articles', current_article_str)
                                        #print(str(date) + current_article_str)
                                        current_article_str = articles_dict['headline']
                                    # for last condition in a year
                                    if (date == current_date) and \
                                        (i == len(NYTimes_data["response"]["docs"][:]) - 1):
                                        interpolated_df.set_value(
                                            date, 'articles', current_article_str)
                        except AttributeError:
                            pass
                        pass
                    except KeyError:
                        print('key error')
                        #print(NYTimes_data["response"]["docs"][:][i])
                        count_main_not_exist += 1
                        pass
                    except TypeError:
                        print("type error")
                        #print(NYTimes_data["response"]["docs"][:][i])
                        count_main_not_exist += 1
                        pass

class GenerateModels:
    """
    DOCSTRING
    """
    def __init__(self):
        df_stocks = pandas.read_pickle('data/pickled_ten_year_filtered_data.pkl')
        df_stocks['prices'] = df_stocks['adj close'].apply(numpy.int64)
        df_stocks = df_stocks[['prices', 'articles']]
        df_stocks['articles'] = df_stocks['articles'].map(lambda x: x.lstrip('.-'))
        df = df_stocks[['prices']].copy()
        df["compound"] = ''
        df["neg"] = ''
        df["neu"] = ''
        df["pos"] = ''
        sid = nltk.sentiment.vader.SentimentIntensityAnalyzer()

        for date, row in df_stocks.T.iteritems():
            try:
                sentence = unicodedata.normalize(
                    'NFKD', df_stocks.loc[date, 'articles']).encode('ascii', 'ignore')
                ss = sid.polarity_scores(str(sentence))
                df.set_value(date, 'compound', ss['compound'])
                df.set_value(date, 'neg', ss['neg'])
                df.set_value(date, 'neu', ss['neu'])
                df.set_value(date, 'pos', ss['pos'])
            except TypeError:
                print(df_stocks.loc[date, 'articles'])
                print(date)

        train_start_date = '2007-01-01'
        train_end_date = '2014-12-31'
        test_start_date = '2015-01-01'
        test_end_date = '2016-12-31'
        train = df.ix[train_start_date : train_end_date]
        test = df.ix[test_start_date:test_end_date]

        sentiment_score_list = list()
        for date, row in train.T.iteritems():
            #sentiment_score = numpy.asarray(
            #    [df.loc[date, 'compound'],
            #     df.loc[date, 'neg'],
            #     df.loc[date, 'neu'],
            #     df.loc[date, 'pos']])
            sentiment_score = numpy.asarray([df.loc[date, 'neg'],df.loc[date, 'pos']])
            sentiment_score_list.append(sentiment_score)
        numpy_df_train = numpy.asarray(sentiment_score_list)

        sentiment_score_list = list()
        for date, row in test.T.iteritems():
            #sentiment_score = numpy.asarray(
            #    [df.loc[date, 'compound'],
            #     df.loc[date, 'neg'],
            #     df.loc[date, 'neu'],
            #     df.loc[date, 'pos']])
            sentiment_score = numpy.asarray([df.loc[date, 'neg'],df.loc[date, 'pos']])
            sentiment_score_list.append(sentiment_score)
        numpy_df_test = numpy.asarray(sentiment_score_list)
        y_train = pandas.DataFrame(train['prices'])
        y_test = pandas.DataFrame(test['prices'])
        rf = sklearn.ensemble.RandomForestRegressor()
        rf.fit(numpy_df_train, y_train)
        print(rf.feature_importances_)
        prediction, bias, contributions = treeinterpreter.treeinterpreter.predict(rf, numpy_df_test)
        print(prediction)
        print(contributions)
        idx = pandas.date_range(test_start_date, test_end_date)
        predictions_df = pandas.DataFrame(
            data=prediction[0:], index = idx, columns=['prices'])
        print(predictions_df)
        #predictions_df.plot()
        #test['prices'].plot()
        predictions_plot = predictions_df.plot()
        fig = y_test.plot(ax = predictions_plot).get_figure()
        fig.savefig("graphs/random forest without smoothing.png")
        ax = predictions_df.rename(
            columns={"prices": "predicted_price"}).plot(
                title='Random Forest predicted prices 8-2 years')
        ax.set_xlabel("Dates")
        ax.set_ylabel("Stock Prices")
        fig = y_test.rename(columns={"prices": "actual_price"}).plot(ax = ax).get_figure()
        fig.savefig("graphs/random forest without smoothing.png")
        #colors = [
        #'332288', '88CCEE', '44AA99', '117733', '999933',
        #'DDCC77', 'CC6677', '882255', 'AA4499']
        print(test)
        # increasing the prices by a constant value so that
        # it represents closing price during the testing
        temp_date = test_start_date
        average_last_5_days_test = 0
        total_days = 10
        for i in range(total_days):
            average_last_5_days_test += test.loc[temp_date, 'prices']
            # converting string to date time
            temp_date = datetime.datetime.strptime(temp_date, "%Y-%m-%d").date()
            # reducing one day from date time
            difference = temp_date + datetime.timedelta(days=1)
            # converting again date time to string
            temp_date = difference.strftime('%Y-%m-%d')
            #print(temp_date)
        average_last_5_days_test = average_last_5_days_test / total_days
        print(average_last_5_days_test)
        temp_date = test_start_date
        average_upcoming_5_days_predicted = 0
        for i in range(total_days):
            average_upcoming_5_days_predicted += predictions_df.loc[temp_date, 'prices']
            # converting string to date time
            temp_date = datetime.datetime.strptime(temp_date, "%Y-%m-%d").date()
            # adding one day from date time
            difference = temp_date + datetime.timedelta(days=1)
            # converting again date time to string
            temp_date = difference.strftime('%Y-%m-%d')
            print(temp_date)
        average_upcoming_5_days_predicted = average_upcoming_5_days_predicted / total_days
        print(average_upcoming_5_days_predicted)
        # average train.loc['2013-12-31', 'prices'] - advpredictions_df.loc['2014-01-01', 'prices']
        difference_test_predicted_prices = average_last_5_days_test - average_upcoming_5_days_predicted
        print(difference_test_predicted_prices)
        print(predictions_df)
        # Adding 6177 to all the advpredictions_df price values
        predictions_df['prices'] = predictions_df['prices'] + difference_test_predicted_prices
        print(predictions_df)
        ax = predictions_df.rename(columns={"prices": "predicted_price"}).plot(title='Random Forest predicted prices 8-2 years after aligning')
        ax.set_xlabel("Dates")
        ax.set_ylabel("Stock Prices")
        fig = y_test.rename(columns={"prices": "actual_price"}).plot(ax = ax).get_figure()
        fig.savefig("graphs/random forest with aligning.png")

    def data_smoothing():
        """
        Applying EWMA pandas to smooth the stock prices.
        """
        print(predictions_df)
        predictions_df['ewma'] = pandas.ewma(
            predictions_df["prices"], span=60, freq="D")
        print(predictions_df)
        predictions_df['actual_value'] = test['prices']
        predictions_df['actual_value_ewma'] = pandas.ewma(
            predictions_df["actual_value"], span=60, freq="D")
        print(predictions_df)
        # changing column names
        predictions_df.columns = [
            'predicted_price',
            'average_predicted_price',
            'actual_price',
            'average_actual_price']
        # now plotting test predictions after smoothing
        predictions_plot = predictions_df.plot(
            title='Random Forest predicted prices 8-2 years after aligning & smoothing')
        predictions_plot.set_xlabel("Dates")
        predictions_plot.set_ylabel("Stock Prices")
        fig = predictions_plot.get_figure()
        fig.savefig("graphs/random forest after smoothing.png")
        # plotting just predict and actual average curves
        predictions_df_average = predictions_df[
            ['average_predicted_price', 'average_actual_price']]
        predictions_plot = predictions_df_average.plot(
            title='Random Forest 8-2 years after aligning & smoothing')
        predictions_plot.set_xlabel("Dates")
        predictions_plot.set_ylabel("Stock Prices")
        fig = predictions_plot.get_figure()
        fig.savefig("./graphs/random forest after smoothing 2.png")

    def logistic_regression():
        """
        DOCSTRING
        """
        #average_upcoming_5_days_predicted += predictions_df.loc[temp_date, 'prices']
        # Converting string to date time
        #temp_date = datetime.datetime.strptime(temp_date, "%Y-%m-%d").date()
        # Adding one day from date time
        #difference = temp_date + datetime.timedelta(days=1)
        # Converting again date time to string
        #temp_date = difference.strftime('%Y-%m-%d')
        #start_year = datetime.datetime.strptime(train_start_date, "%Y-%m-%d").date().month
        prediction_list = list()
        years = [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]
        for year in years:
            # splitting the training and testing data
            train_start_date = str(year) + '-01-01'
            train_end_date = str(year) + '-10-31'
            test_start_date = str(year) + '-11-01'
            test_end_date = str(year) + '-12-31'
            train = df.ix[train_start_date : train_end_date]
            test = df.ix[test_start_date:test_end_date]
            # calculating the sentiment score
            sentiment_score_list = list()
            for date, row in train.T.iteritems():
                sentiment_score = numpy.asarray(
                    [df.loc[date, 'compound'],
                     df.loc[date, 'neg'],
                     df.loc[date, 'neu'],
                     df.loc[date, 'pos']])
                #sentiment_score = numpy.asarray(
                #    [df.loc[date, 'neg'], df.loc[date, 'pos']])
                sentiment_score_list.append(sentiment_score)
            numpy_df_train = numpy.asarray(sentiment_score_list)
            sentiment_score_list = list()
            for date, row in test.T.iteritems():
                sentiment_score = numpy.asarray(
                    [df.loc[date, 'compound'],
                     df.loc[date, 'neg'],
                     df.loc[date, 'neu'],
                     df.loc[date, 'pos']])
                #sentiment_score = numpy.asarray(
                #    [df.loc[date, 'neg'], df.loc[date, 'pos']])
                sentiment_score_list.append(sentiment_score)
            numpy_df_test = numpy.asarray(sentiment_score_list)
            # generating models
            lr = sklearn.linear_model.LogisticRegression()
            lr.fit(numpy_df_train, train['prices'])
            prediction = lr.predict(numpy_df_test)
            prediction_list.append(prediction)
            #print(
            #    train_start_date + ' ' + train_end_date + ' ' +
            #    test_start_date + ' ' + test_end_date)
            idx = pandas.date_range(test_start_date, test_end_date)
            #print(year)
            predictions_df_list = pandas.DataFrame(
                data=prediction[0:], index = idx, columns=['prices'])
            difference_test_predicted_prices = offset_value(
                test_start_date, test, predictions_df_list)
            # adding offset to all the advpredictions_df price values
            predictions_df_list['prices'] = \
                predictions_df_list['prices'] + difference_test_predicted_prices
            predictions_df_list
            # smoothing the plot
            predictions_df_list['ewma'] = pandas.ewma(
                predictions_df_list["prices"], span=10, freq="D")
            predictions_df_list['actual_value'] = test['prices']
            predictions_df_list['actual_value_ewma'] = pandas.ewma(
                predictions_df_list["actual_value"], span=10, freq="D")
            # Changing column names
            predictions_df_list.columns = [
                'predicted_price',
                'average_predicted_price',
                'actual_price',
                'average_actual_price']
            predictions_df_list.plot()
            predictions_df_list_average = predictions_df_list[
                ['average_predicted_price', 'average_actual_price']]
            predictions_df_list_average.plot()
            #predictions_df_list.show()
            print(lr.classes_)
            print(lr.coef_[0])

    def mlp_classifier():
        """
        DOCSTRING
        """
        #average_upcoming_5_days_predicted += predictions_df.loc[temp_date, 'prices']
        # converting string to date time
        #temp_date = datetime.datetime.strptime(temp_date, "%Y-%m-%d").date()
        # adding one day from date time
        #difference = temp_date + datetime.timedelta(days=1)
        # converting again date time to string
        #temp_date = difference.strftime('%Y-%m-%d')
        #start_year = datetime.datetime.strptime(train_start_date, "%Y-%m-%d").date().month
        prediction_list = []
        years = [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]
        for year in years:
            # splitting the training and testing data
            train_start_date = str(year) + '-01-01'
            train_end_date = str(year) + '-10-31'
            test_start_date = str(year) + '-11-01'
            test_end_date = str(year) + '-12-31'
            train = df.ix[train_start_date:train_end_date]
            test = df.ix[test_start_date:test_end_date]
            # calculating the sentiment score
            sentiment_score_list = []
            for date, row in train.T.iteritems():
                sentiment_score = numpy.asarray(
                    [df.loc[date, 'compound'],
                     df.loc[date, 'neg'],
                     df.loc[date, 'neu'],
                     df.loc[date, 'pos']])
                #sentiment_score = numpy.asarray(
                #[df.loc[date, 'neg'],df.loc[date, 'pos']])
                sentiment_score_list.append(sentiment_score)
            numpy_df_train = numpy.asarray(sentiment_score_list)
            sentiment_score_list = []
            for date, row in test.T.iteritems():
                sentiment_score = numpy.asarray(
                    [df.loc[date, 'compound'],
                     df.loc[date, 'neg'],
                     df.loc[date, 'neu'],
                     df.loc[date, 'pos']])
                #sentiment_score = numpy.asarray(
                #[df.loc[date, 'neg'],df.loc[date, 'pos']])
                sentiment_score_list.append(sentiment_score)
            numpy_df_test = numpy.asarray(sentiment_score_list)
            # generating models
            mlpc = sklearn.neural_network.MLPClassifier(
                hidden_layer_sizes=(100, 200, 100),
                activation='relu',
                solver='lbfgs',
                alpha=0.005,
                learning_rate_init = 0.001,
                shuffle=False) # span = 20 # best 1
            mlpc.fit(numpy_df_train, train['prices'])
            prediction = mlpc.predict(numpy_df_test)
            prediction_list.append(prediction)
            #print(train_start_date + ' ' + train_end_date + ' ' + 
            #test_start_date + ' ' + test_end_date)
            idx = pandas.date_range(test_start_date, test_end_date)
            #print(year)
            predictions_df_list = pandas.DataFrame(
                data=prediction[0:], index = idx, columns=['prices'])
            difference_test_predicted_prices = offset_value(
                test_start_date, test, predictions_df_list)
            # adding offset to all the advpredictions_df price values
            predictions_df_list['prices'] = \
                predictions_df_list['prices'] + difference_test_predicted_prices
            predictions_df_list
            # smoothing the plot
            predictions_df_list['ewma'] = pandas.ewma(
                predictions_df_list["prices"], span=20, freq="D")
            predictions_df_list['actual_value'] = test['prices']
            predictions_df_list['actual_value_ewma'] = pandas.ewma(
                predictions_df_list["actual_value"], span=20, freq="D")
            # changing column names
            predictions_df_list.columns = [
                'predicted_price',
                'average_predicted_price',
                'actual_price',
                'average_actual_price']
            predictions_df_list.plot()
            predictions_df_list_average = predictions_df_list[
                ['average_predicted_price', 'average_actual_price']]
            predictions_df_list_average.plot()
            #predictions_df_list.show()
        mlpc = sklearn.neural_network.MLPClassifier(
            hidden_layer_sizes=(100, 200, 100),
            activation='tanh',
            solver='lbfgs',
            alpha=0.010,
            learning_rate_init = 0.001,
            shuffle=False)
        mlpc = sklearn.neural_network.MLPClassifier(
            hidden_layer_sizes=(100, 200, 100),
            activation='relu',
            solver='lbfgs',
            alpha=0.010,
            learning_rate_init = 0.001,
            shuffle=False) # span = 20
        mlpc = sklearn.neural_network.MLPClassifier(
            hidden_layer_sizes=(100, 200, 100),
            activation='relu',
            solver='lbfgs',
            alpha=0.005,
            learning_rate_init=0.001,
            shuffle=False) # span = 20 ### best 1 ###
        mlpc = sklearn.neural_network.MLPClassifier(
            hidden_layer_sizes=(100, 200, 50),
            activation='relu',
            solver='lbfgs',
            alpha=0.005,
            learning_rate_init=0.001,
            shuffle=False)
        # checking the performance of training data itself
        prediction, bias, contributions = treeinterpreter.treeinterpreter.predict(rf, numpy_df_train)
        idx = pandas.date_range(train_start_date, train_end_date)
        predictions_df1 = pandas.DataFrame(
            data=prediction[0:], index = idx, columns=['prices'])
        predictions_df1.plot()
        train['prices'].plot()

    def offset_value(test_start_date, test, predictions_df):
        """
        Increasing the prices by a constant value so that
        it represents closing price during the testing.
        """
        temp_date = test_start_date
        average_last_5_days_test = 0
        average_upcoming_5_days_predicted = 0
        total_days = 10
        for i in range(total_days):
            average_last_5_days_test += test.loc[temp_date, 'prices']
            temp_date = datetime.datetime.strptime(temp_date, "%Y-%m-%d").date()
            difference = temp_date + datetime.timedelta(days=1)
            temp_date = difference.strftime('%Y-%m-%d')
        average_last_5_days_test = average_last_5_days_test / total_days
        temp_date = test_start_date
        for i in range(total_days):
            average_upcoming_5_days_predicted += predictions_df.loc[temp_date, 'prices']
            temp_date = datetime.datetime.strptime(temp_date, "%Y-%m-%d").date()
            difference = temp_date + datetime.timedelta(days=1)
            temp_date = difference.strftime('%Y-%m-%d')
        average_upcoming_5_days_predicted = average_upcoming_5_days_predicted / total_days
        difference_test_predicted_prices = \
            average_last_5_days_test - average_upcoming_5_days_predicted
        return difference_test_predicted_prices

    def random_forest_regressor():
        """
        DOCSTRING
        """
        #average_upcoming_5_days_predicted += predictions_df.loc[temp_date, 'prices']
        # converting string to date time
        #temp_date = datetime.strptime(temp_date, "%Y-%m-%d").date()
        # adding one day from date time
        #difference = temp_date + datetime.timedelta(days=1)
        # converting again date time to string
        #temp_date = difference.strftime('%Y-%m-%d')
        #start_year = datetime.strptime(train_start_date, "%Y-%m-%d").date().month
        prediction_list = list()
        years = [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]
        for year in years:
            # splitting the training and testing data
            train_start_date = str(year) + '-01-01'
            train_end_date = str(year) + '-10-31'
            test_start_date = str(year) + '-11-01'
            test_end_date = str(year) + '-12-31'
            train = df.ix[train_start_date : train_end_date]
            test = df.ix[test_start_date:test_end_date]
            # calculating the sentiment score
            sentiment_score_list = list()
            for date, row in train.T.iteritems():
                sentiment_score = numpy.asarray(
                    [df.loc[date, 'compound'],
                     df.loc[date, 'neg'],
                     df.loc[date, 'neu'],
                     df.loc[date, 'pos']])
                #sentiment_score = numpy.asarray(
                #    [df.loc[date, 'neg'],df.loc[date, 'pos']])
                sentiment_score_list.append(sentiment_score)
            numpy_df_train = numpy.asarray(sentiment_score_list)
            sentiment_score_list = list()
            for date, row in test.T.iteritems():
                sentiment_score = numpy.asarray(
                    [df.loc[date, 'compound'],
                     df.loc[date, 'neg'],
                     df.loc[date, 'neu'],
                     df.loc[date, 'pos']])
                #sentiment_score = numpy.asarray(
                #    [df.loc[date, 'neg'], df.loc[date, 'pos']])
                sentiment_score_list.append(sentiment_score)
            numpy_df_test = numpy.asarray(sentiment_score_list)
            # generating models
            rf = sklearn.ensemble.RandomForestRegressor(random_state=1)
            rf.fit(numpy_df_train, train['prices'])
            # print(rf)
            prediction, bias, contributions = treeinterpreter.treeinterpreter.predict(rf, numpy_df_test)
            prediction_list.append(prediction)
            #print(
            #    train_start_date + ' ' + train_end_date + ' ' +
            #    test_start_date + ' ' + test_end_date)
            idx = pandas.date_range(test_start_date, test_end_date)
            # print(year)
            predictions_df_list = pandas.DataFrame(
                data=prediction[0:], index = idx, columns=['prices'])
            difference_test_predicted_prices = offset_value(
                test_start_date, test, predictions_df_list)
            # adding offset to all the advpredictions_df price values
            predictions_df_list['prices'] = \
                predictions_df_list['prices'] + difference_test_predicted_prices
            print(predictions_df_list)
            # smoothing the plot
            predictions_df_list['ewma'] = pandas.ewma(
                predictions_df_list["prices"], span=10, freq="D")
            predictions_df_list['actual_value'] = test['prices']
            predictions_df_list['actual_value_ewma'] = pandas.ewma(
                predictions_df_list["actual_value"], span=10, freq="D")
            # changing column names
            predictions_df_list.columns = [
                'predicted_price',
                'average_predicted_price',
                'actual_price',
                'average_actual_price']
            predictions_df_list.plot()
            predictions_df_list_average = predictions_df_list[
                ['average_predicted_price', 'average_actual_price']]
            predictions_df_list_average.plot()
            #predictions_df_list.show()
        #from IPython.display import Image
        #dot_data = tree.export_graphviz(
        #    rf, out_file=None,
        #    feature_names=['comp', 'neg', 'neu', 'pos'],
        #    class_names=iris.target_names,
        #    filled=True, rounded=True,
        #    special_characters=True)
        #graph = pydotplus.graph_from_dot_data(dot_data)
        #Image(graph.create_png())

if __name__ == '__main__':
    collect_data = CollectData()
