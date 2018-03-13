from sklearn import tree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model

import statistics
std_boundary = 0
mean_boundary = 0

# boundary
b1 = 0.01
b2 = 0.0025

# You can write code above the if-main block.
def simple_moving_avg(period, data):
    moving_avg_data = []
    for i in range(len(data)-period+1):
        l = data[i:i+period]
        l = map(float, l)
        moving_avg_data.append(sum(l) / float(period))
    return moving_avg_data

def create_table(open_price_moving, open_change, trend):
    feature = []
    label = []
    volumn = len(open_price_moving)-3
    print(volumn)
    for i in range(volumn):
        data = open_price_moving[i:i+3]+open_change[i:i+2]
        feature.append(data)
        label.append(trend[i+2])
    # print(feature)
   
    return (feature, label)


class Trader(object):
    def train(self, training_data):
        # moving average and classifier
        print("start training")
        open_price = training_data[0]
        high_price = training_data[1]
        low_price = training_data[2]
        close_price = training_data[3]


        # convert to moving average
        MA_days = 10
        open_price_moving = simple_moving_avg(MA_days, open_price)
        high_price_moving = simple_moving_avg(MA_days, high_price)
        low_price_moving = simple_moving_avg(MA_days, low_price)
        close_price_moving = simple_moving_avg(MA_days, close_price)

        # feature and label:
        """
        feature:
        open(t-2), change(t-2, t-1), open(t-1), change(t-1, t), open(t)
        """
        open_change = [(open_price_moving[x+1]-open_price_moving[x])/float(open_price_moving[x]) for x in range(len(open_price_moving)-1)]
        # print(open_change)
        std_boundary = statistics.stdev(open_change)
        mean_boundary = sum(open_change)/float(len(open_change))


        print(std_boundary)
        print(mean_boundary)
        
        trend = []
        for i in open_change:
            if i < -b1:
                trend.append(-2)
            elif i >= -b1 and i < -b2:
                trend.append(-1)
            elif i >= -b2 and i <= b2:
                trend.append(0)
            elif i > b2 and i <= b1:
                trend.append(1)
            elif i > b1:
                trend.append(2)

        # combine feature and label
        feature, label = create_table(open_price_moving, open_change, trend)

        # model = tree.DecisionTreeClassifier(random_state=0)
        # model = SVC()
        # model = KNeighborsClassifier()
        model = RandomForestClassifier(random_state=0)
        # model = linear_model.LinearRegression()
        
        model = model.fit(feature, label)

        return model

    def predict_action(self, predict_trend):
        # return a list of action for everyday
        action = [0]*11
        
        current_state = 0       # stable
        
        # print(predict_trend)
        for pt in predict_trend:

            if current_state == 0:
                if pt == 0:
                    action.append(0)
                elif pt == 1:
                    action.append(1)
                    current_state = 1
                elif pt == -1:
                    action.append(-1)
                    current_state = -1
                elif pt == 2:
                    action.append(1)
                    current_state = 1
                elif pt == -2:
                    action.append(-1)
                    current_state = -1


            elif current_state == 1:
                if pt == 0 or pt == 1 or pt == 2:
                    action.append(0)
                elif pt == -1 or pt == -2:
                    action.append(-1)
                    current_state = 0

            elif current_state == -1:
                if pt == 0 or pt == -1 or pt == -2:
                    action.append(0)
                elif pt == 1 or pt == 2:
                    action.append(1)
                    current_state = 0

        return action

    def compute_profit(self, actions, testing_data_opp, testing_data_clp):
        profit = 0
        current_state = 0
        print("compute_profit")
        # action length is 1 less
        testing_data_opp = list(map(float, testing_data_opp))
        testing_data_clp = list(map(float, testing_data_clp))
        print(len(actions), len(testing_data_opp), len(testing_data_clp))
        for i in range(len(actions)):
            if current_state == 0:
                if actions[i] == 1:
                    current_state = 1
                    stock_in = testing_data_opp[i+1]
                    # print(stock_in)
                elif actions[i] == -1:
                    current_state = -1
                    stock_in = -testing_data_opp[i+1]
                    # print(stock_in)
            elif current_state == 1:
                if actions[i] == -1:
                    current_state = 0
                    stock_out = -testing_data_opp[i+1]
                    # print(stock_out)
                    profit += (stock_in+stock_out)*-1
                    # print(profit)
            elif current_state == -1:
                if actions[i] == 1:
                    current_state = 0
                    stock_out = testing_data_opp[i+1]
                    # print(stock_out)
                    profit += (stock_in+stock_out)*-1
                    # print(profit)
        # if hold +1 or -1
        if current_state == 1:
            print("check last close")
            profit += testing_data_clp[-1]-stock_in
        elif current_state == -1:
            print("check last close")
            profit += (testing_data_clp[-1]+stock_in)*-1
        return profit


    def buy_and_hold(testing_data):
        # baseline method, need not to train
        result = 0

        return result


def load_data(file_name):
    import csv

    open_price = []
    high_price = []
    low_price = []
    close_price = []
    with open(file_name, 'r') as file:
        for row in csv.reader(file):
            open_price.append(row[0])
            high_price.append(row[1])
            low_price.append(row[2])
            close_price.append(row[3])

    return (open_price, high_price, low_price, close_price)





if __name__ == '__main__':
    # You should not modify this part.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')
    parser.add_argument('--testing',
                        default='testing_data.csv',
                        help='input testing data file name')
    parser.add_argument('--output',
                        default='output.csv',
                        help='output file name')
    args = parser.parse_args()
    
    # The following part is an example.
    # You can modify it at will.
    training_data = load_data(args.training)
    trader = Trader()
    model = trader.train(training_data)

    print(model)


    ### testing data section:
    testing_data = load_data(args.testing)
    open_price = testing_data[0]
    high_price = testing_data[1]
    low_price = testing_data[2]
    close_price = testing_data[3]


    # convert to moving average
    MA_days = 10
    open_price_moving = simple_moving_avg(MA_days, open_price)
    high_price_moving = simple_moving_avg(MA_days, high_price)
    low_price_moving = simple_moving_avg(MA_days, low_price)
    close_price_moving = simple_moving_avg(MA_days, close_price)

    print(open_price_moving)


    open_change = [(open_price_moving[x+1]-open_price_moving[x])/float(open_price_moving[x]) for x in range(len(open_price_moving)-1)]
    print(open_change)
    trend = []
    for i in open_change:
        if i < -b1:
            trend.append(-2)
        elif i >= -b1 and i < -b2:
            trend.append(-1)
        elif i >= -b2 and i <= b2:
            trend.append(0)
        elif i > b2 and i <= b1:
            trend.append(1)
        elif i > b1:
            trend.append(2)
    print(trend)
    print(len(open_change), len(trend))

    feature, label = create_table(open_price_moving, open_change, trend)
    print(feature, label)

    # predict
    predict_trend = model.predict(feature)
    actions = trader.predict_action(predict_trend)
    print("actions")
    print(actions, len(actions))
    for a in actions:
        print(a)

    profit = trader.compute_profit(actions, open_price, close_price)
    print("profit")
    print(profit)
    
    # with open(args.output, 'w') as output_file:
    #     for row in testing_data:
    #         # We will perform your action as the open price in the next day.
    #         action = trader.predict_action(datum)
    #         output_file.write(action)

    #         # this is your option, you can leave it empty.
    #         trader.re_training(i)