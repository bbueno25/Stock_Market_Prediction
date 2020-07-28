from datetime import datetime
from datetime import timedelta
from matplotlib import pyplot
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy
import pandas
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeRegressor
from treeinterpreter import treeinterpreter as ti
import unicodedata

df_stocks = pandas.read_pickle('data/pickled_ten_year_filtered_data.pkl')
df_stocks['prices'] = df_stocks['adj close'].apply(numpy.int64)
df_stocks = df_stocks[['prices', 'articles']]
df_stocks['articles'] = df_stocks['articles'].map(lambda x: x.lstrip('.-'))
df = df_stocks[['prices']].copy()
df["compound"] = ''
df["neg"] = ''
df["neu"] = ''
df["pos"] = ''
sid = SentimentIntensityAnalyzer()

for date, row in df_stocks.T.iteritems():
    try:
        sentence = unicodedata.normalize('NFKD', df_stocks.loc[date, 'articles']).encode('ascii', 'ignore')
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

sentiment_score_list = []
for date, row in train.T.iteritems():
    # sentiment_score = numpy.asarray([df.loc[date, 'compound'],df.loc[date, 'neg'],df.loc[date, 'neu'],df.loc[date, 'pos']])
    sentiment_score = numpy.asarray([df.loc[date, 'neg'],df.loc[date, 'pos']])
    sentiment_score_list.append(sentiment_score)
numpy_df_train = numpy.asarray(sentiment_score_list)

sentiment_score_list = []
for date, row in test.T.iteritems():
    # sentiment_score = numpy.asarray([df.loc[date, 'compound'],df.loc[date, 'neg'],df.loc[date, 'neu'],df.loc[date, 'pos']])
    sentiment_score = numpy.asarray([df.loc[date, 'neg'],df.loc[date, 'pos']])
    sentiment_score_list.append(sentiment_score)
numpy_df_test = numpy.asarray(sentiment_score_list)
y_train = pandas.DataFrame(train['prices'])
y_test = pandas.DataFrame(test['prices'])
rf = RandomForestRegressor()
rf.fit(numpy_df_train, y_train)
print(rf.feature_importances_)
prediction, bias, contributions = ti.predict(rf, numpy_df_test)
print(prediction)
print(contributions)
idx = pandas.date_range(test_start_date, test_end_date)
predictions_df = pandas.DataFrame(data=prediction[0:], index = idx, columns=['prices'])
print(predictions_df)
# predictions_df.plot()
# test['prices'].plot()
predictions_plot = predictions_df.plot()
fig = y_test.plot(ax = predictions_plot).get_figure()
fig.savefig("graphs/random forest without smoothing.png")
ax = predictions_df.rename(columns={"prices": "predicted_price"}).plot(title='Random Forest predicted prices 8-2 years')
ax.set_xlabel("Dates")
ax.set_ylabel("Stock Prices")
fig = y_test.rename(columns={"prices": "actual_price"}).plot(ax = ax).get_figure()
fig.savefig("graphs/random forest without smoothing.png")
# colors = ['332288', '88CCEE', '44AA99', '117733', '999933', 'DDCC77', 'CC6677', '882255', 'AA4499']
print(test)

" Increasing the prices by a constant value so that it represents closing price during the testing "
temp_date = test_start_date
average_last_5_days_test = 0
total_days = 10
for i in range(total_days):
    average_last_5_days_test += test.loc[temp_date, 'prices']
    # Converting string to date time
    temp_date = datetime.strptime(temp_date, "%Y-%m-%d").date()
    # Reducing one day from date time
    difference = temp_date + timedelta(days=1)
    # Converting again date time to string
    temp_date = difference.strftime('%Y-%m-%d')
    # print temp_date
average_last_5_days_test = average_last_5_days_test / total_days
print(average_last_5_days_test)
temp_date = test_start_date
average_upcoming_5_days_predicted = 0
for i in range(total_days):
    average_upcoming_5_days_predicted += predictions_df.loc[temp_date, 'prices']
    # Converting string to date time
    temp_date = datetime.strptime(temp_date, "%Y-%m-%d").date()
    # Adding one day from date time
    difference = temp_date + timedelta(days=1)
    # Converting again date time to string
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
    " Applying EWMA pandas to smooth the stock prices "
    print(predictions_df)
    predictions_df['ewma'] = pandas.ewma(predictions_df["prices"], span=60, freq="D")
    print(predictions_df)
    predictions_df['actual_value'] = test['prices']
    predictions_df['actual_value_ewma'] = pandas.ewma(predictions_df["actual_value"], span=60, freq="D")
    print(predictions_df)
    # Changing column names
    predictions_df.columns = ['predicted_price', 'average_predicted_price', 'actual_price', 'average_actual_price']
    # Now plotting test predictions after smoothing
    predictions_plot = predictions_df.plot(title='Random Forest predicted prices 8-2 years after aligning & smoothing')
    predictions_plot.set_xlabel("Dates")
    predictions_plot.set_ylabel("Stock Prices")
    fig = predictions_plot.get_figure()
    fig.savefig("graphs/random forest after smoothing.png")
    # Plotting just predict and actual average curves
    predictions_df_average = predictions_df[['average_predicted_price', 'average_actual_price']]
    predictions_plot = predictions_df_average.plot(title='Random Forest 8-2 years after aligning & smoothing')
    predictions_plot.set_xlabel("Dates")
    predictions_plot.set_ylabel("Stock Prices")
    fig = predictions_plot.get_figure()
    fig.savefig("./graphs/random forest after smoothing 2.png")

def offset_value(test_start_date, test, predictions_df):
    " Increasing the prices by a constant value so that it represents closing price during the testing "
    temp_date = test_start_date
    average_last_5_days_test = 0
    average_upcoming_5_days_predicted = 0
    total_days = 10
    for i in range(total_days):
        average_last_5_days_test += test.loc[temp_date, 'prices']
        temp_date = datetime.strptime(temp_date, "%Y-%m-%d").date()
        difference = temp_date + timedelta(days=1)
        temp_date = difference.strftime('%Y-%m-%d')
    average_last_5_days_test = average_last_5_days_test / total_days
    temp_date = test_start_date
    for i in range(total_days):
        average_upcoming_5_days_predicted += predictions_df.loc[temp_date, 'prices']
        temp_date = datetime.strptime(temp_date, "%Y-%m-%d").date()
        difference = temp_date + timedelta(days=1)
        temp_date = difference.strftime('%Y-%m-%d')
    average_upcoming_5_days_predicted = average_upcoming_5_days_predicted / total_days
    difference_test_predicted_prices = average_last_5_days_test - average_upcoming_5_days_predicted
    return difference_test_predicted_prices

def logistic_regression():
    # average_upcoming_5_days_predicted += predictions_df.loc[temp_date, 'prices']
    # # Converting string to date time
    # temp_date = datetime.strptime(temp_date, "%Y-%m-%d").date()
    # # Adding one day from date time
    # difference = temp_date + timedelta(days=1)
    # # Converting again date time to string
    # temp_date = difference.strftime('%Y-%m-%d')
    # start_year = datetime.strptime(train_start_date, "%Y-%m-%d").date().month
    prediction_list = []
    years = [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]
    for year in years:
        # Splitting the training and testing data
        train_start_date = str(year) + '-01-01'
        train_end_date = str(year) + '-10-31'
        test_start_date = str(year) + '-11-01'
        test_end_date = str(year) + '-12-31'
        train = df.ix[train_start_date : train_end_date]
        test = df.ix[test_start_date:test_end_date]
        # Calculating the sentiment score
        sentiment_score_list = []
        for date, row in train.T.iteritems():
            sentiment_score = numpy.asarray([df.loc[date, 'compound'],df.loc[date, 'neg'],df.loc[date, 'neu'],df.loc[date, 'pos']])
            # sentiment_score = numpy.asarray([df.loc[date, 'neg'],df.loc[date, 'pos']])
            sentiment_score_list.append(sentiment_score)
        numpy_df_train = numpy.asarray(sentiment_score_list)
        sentiment_score_list = []
        for date, row in test.T.iteritems():
            sentiment_score = numpy.asarray([df.loc[date, 'compound'],df.loc[date, 'neg'],df.loc[date, 'neu'],df.loc[date, 'pos']])
            # sentiment_score = numpy.asarray([df.loc[date, 'neg'],df.loc[date, 'pos']])
            sentiment_score_list.append(sentiment_score)
        numpy_df_test = numpy.asarray(sentiment_score_list)
        # Generating models
        lr = LogisticRegression()
        lr.fit(numpy_df_train, train['prices'])
        prediction = lr.predict(numpy_df_test)
        prediction_list.append(prediction)
        # print train_start_date + ' ' + train_end_date + ' ' + test_start_date + ' ' + test_end_date
        idx = pandas.date_range(test_start_date, test_end_date)
        # print year
        predictions_df_list = pandas.DataFrame(data=prediction[0:], index = idx, columns=['prices'])
        difference_test_predicted_prices = offset_value(test_start_date, test, predictions_df_list)
        # Adding offset to all the advpredictions_df price values
        predictions_df_list['prices'] = predictions_df_list['prices'] + difference_test_predicted_prices
        predictions_df_list
        # Smoothing the plot
        predictions_df_list['ewma'] = pandas.ewma(predictions_df_list["prices"], span=10, freq="D")
        predictions_df_list['actual_value'] = test['prices']
        predictions_df_list['actual_value_ewma'] = pandas.ewma(predictions_df_list["actual_value"], span=10, freq="D")
        # Changing column names
        predictions_df_list.columns = ['predicted_price', 'average_predicted_price', 'actual_price', 'average_actual_price']
        predictions_df_list.plot()
        predictions_df_list_average = predictions_df_list[['average_predicted_price', 'average_actual_price']]
        predictions_df_list_average.plot()
        # predictions_df_list.show()
        print(lr.classes_)
        print(lr.coef_[0])

def random_forest_regressor():
    # average_upcoming_5_days_predicted += predictions_df.loc[temp_date, 'prices']
    # # Converting string to date time
    # temp_date = datetime.strptime(temp_date, "%Y-%m-%d").date()
    # # Adding one day from date time
    # difference = temp_date + timedelta(days=1)
    # # Converting again date time to string
    # temp_date = difference.strftime('%Y-%m-%d')
    # start_year = datetime.strptime(train_start_date, "%Y-%m-%d").date().month
    prediction_list = []
    years = [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]
    for year in years:
        # Splitting the training and testing data
        train_start_date = str(year) + '-01-01'
        train_end_date = str(year) + '-10-31'
        test_start_date = str(year) + '-11-01'
        test_end_date = str(year) + '-12-31'
        train = df.ix[train_start_date : train_end_date]
        test = df.ix[test_start_date:test_end_date]
        # Calculating the sentiment score
        sentiment_score_list = []
        for date, row in train.T.iteritems():
            sentiment_score = numpy.asarray([df.loc[date, 'compound'],df.loc[date, 'neg'],df.loc[date, 'neu'],df.loc[date, 'pos']])
            # sentiment_score = numpy.asarray([df.loc[date, 'neg'],df.loc[date, 'pos']])
            sentiment_score_list.append(sentiment_score)
        numpy_df_train = numpy.asarray(sentiment_score_list)
        sentiment_score_list = []
        for date, row in test.T.iteritems():
            sentiment_score = numpy.asarray([df.loc[date, 'compound'],df.loc[date, 'neg'],df.loc[date, 'neu'],df.loc[date, 'pos']])
            # sentiment_score = numpy.asarray([df.loc[date, 'neg'],df.loc[date, 'pos']])
            sentiment_score_list.append(sentiment_score)
        numpy_df_test = numpy.asarray(sentiment_score_list)
        # Generating models
        rf = RandomForestRegressor(random_state=1)
        rf.fit(numpy_df_train, train['prices'])
        # print rf
        prediction, bias, contributions = ti.predict(rf, numpy_df_test)
        prediction_list.append(prediction)
        # print train_start_date + ' ' + train_end_date + ' ' + test_start_date + ' ' + test_end_date
        idx = pandas.date_range(test_start_date, test_end_date)
        # print year
        predictions_df_list = pandas.DataFrame(data=prediction[0:], index = idx, columns=['prices'])
        difference_test_predicted_prices = offset_value(test_start_date, test, predictions_df_list)
        # Adding offset to all the advpredictions_df price values
        predictions_df_list['prices'] = predictions_df_list['prices'] + difference_test_predicted_prices
        print(predictions_df_list)
        # Smoothing the plot
        predictions_df_list['ewma'] = pandas.ewma(predictions_df_list["prices"], span=10, freq="D")
        predictions_df_list['actual_value'] = test['prices']
        predictions_df_list['actual_value_ewma'] = pandas.ewma(predictions_df_list["actual_value"], span=10, freq="D")
        # Changing column names
        predictions_df_list.columns = ['predicted_price', 'average_predicted_price', 'actual_price', 'average_actual_price']
        predictions_df_list.plot()
        predictions_df_list_average = predictions_df_list[['average_predicted_price', 'average_actual_price']]
        predictions_df_list_average.plot()
        # predictions_df_list.show()
    # from IPython.display import Image
    # dot_data = tree.export_graphviz(rf, out_file=None,
    #                      feature_names=['comp', 'neg', 'neu', 'pos'],
    #                      class_names=iris.target_names,
    #                      filled=True, rounded=True,
    #                      special_characters=True)
    # graph = pydotplus.graph_from_dot_data(dot_data)
    # Image(graph.create_png())

def mlp_classifier():
    # average_upcoming_5_days_predicted += predictions_df.loc[temp_date, 'prices']
    # # Converting string to date time
    # temp_date = datetime.strptime(temp_date, "%Y-%m-%d").date()
    # # Adding one day from date time
    # difference = temp_date + timedelta(days=1)
    # # Converting again date time to string
    # temp_date = difference.strftime('%Y-%m-%d')
    # start_year = datetime.strptime(train_start_date, "%Y-%m-%d").date().month
    prediction_list = []
    years = [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]
    for year in years:
        # Splitting the training and testing data
        train_start_date = str(year) + '-01-01'
        train_end_date = str(year) + '-10-31'
        test_start_date = str(year) + '-11-01'
        test_end_date = str(year) + '-12-31'
        train = df.ix[train_start_date : train_end_date]
        test = df.ix[test_start_date:test_end_date]
        # Calculating the sentiment score
        sentiment_score_list = []
        for date, row in train.T.iteritems():
            sentiment_score = numpy.asarray([df.loc[date, 'compound'],df.loc[date, 'neg'],df.loc[date, 'neu'],df.loc[date, 'pos']])
            # sentiment_score = numpy.asarray([df.loc[date, 'neg'],df.loc[date, 'pos']])
            sentiment_score_list.append(sentiment_score)
        numpy_df_train = numpy.asarray(sentiment_score_list)
        sentiment_score_list = []
        for date, row in test.T.iteritems():
            sentiment_score = numpy.asarray([df.loc[date, 'compound'],df.loc[date, 'neg'],df.loc[date, 'neu'],df.loc[date, 'pos']])
            # sentiment_score = numpy.asarray([df.loc[date, 'neg'],df.loc[date, 'pos']])
            sentiment_score_list.append(sentiment_score)
        numpy_df_test = numpy.asarray(sentiment_score_list)
        # Generating models
        mlpc = MLPClassifier(hidden_layer_sizes=(100, 200, 100),
                            activation='relu',
                            solver='lbfgs',
                            alpha=0.005,
                            learning_rate_init = 0.001,
                            shuffle=False)    # span = 20 # best 1
        mlpc.fit(numpy_df_train, train['prices'])
        prediction = mlpc.predict(numpy_df_test)
        prediction_list.append(prediction)
        # print train_start_date + ' ' + train_end_date + ' ' + test_start_date + ' ' + test_end_date
        idx = pandas.date_range(test_start_date, test_end_date)
        # print year
        predictions_df_list = pandas.DataFrame(data=prediction[0:], index = idx, columns=['prices'])
        difference_test_predicted_prices = offset_value(test_start_date, test, predictions_df_list)
        # Adding offset to all the advpredictions_df price values
        predictions_df_list['prices'] = predictions_df_list['prices'] + difference_test_predicted_prices
        predictions_df_list
        # Smoothing the plot
        predictions_df_list['ewma'] = pandas.ewma(predictions_df_list["prices"], span=20, freq="D")
        predictions_df_list['actual_value'] = test['prices']
        predictions_df_list['actual_value_ewma'] = pandas.ewma(predictions_df_list["actual_value"], span=20, freq="D")
        # Changing column names
        predictions_df_list.columns = ['predicted_price', 'average_predicted_price', 'actual_price', 'average_actual_price']
        predictions_df_list.plot()
        predictions_df_list_average = predictions_df_list[['average_predicted_price', 'average_actual_price']]
        predictions_df_list_average.plot()
        # predictions_df_list.show()
    mlpc = MLPClassifier(hidden_layer_sizes=(100, 200, 100),
                        activation='tanh',
                        solver='lbfgs',
                        alpha=0.010,
                        learning_rate_init = 0.001,
                        shuffle=False)
    mlpc = MLPClassifier(hidden_layer_sizes=(100, 200, 100),
                        activation='relu',
                        solver='lbfgs',
                        alpha=0.010,
                        learning_rate_init = 0.001,
                        shuffle=False)    # span = 20
    mlpc = MLPClassifier(hidden_layer_sizes=(100, 200, 100),
                        activation='relu',
                        solver='lbfgs',
                        alpha=0.005,
                        learning_rate_init=0.001,
                        shuffle=False)    # span = 20 ### best 1 ###
    mlpc = MLPClassifier(hidden_layer_sizes=(100, 200, 50),
                        activation='relu',
                        solver='lbfgs',
                        alpha=0.005,
                        learning_rate_init=0.001,
                        shuffle=False)
    # checking the performance of training data itself
    prediction, bias, contributions = ti.predict(rf, numpy_df_train)
    idx = pandas.date_range(train_start_date, train_end_date)
    predictions_df1 = pandas.DataFrame(data=prediction[0:], index = idx, columns=['prices'])
    predictions_df1.plot()
    train['prices'].plot()
