import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Load CSV file into DataFrame
file_path = './data/survey_lung_cancer_data.csv'
df = pd.read_csv(file_path)
# del df['AGE']
# print(df)
# new_col_value = 1
# df.insert(loc=0,
#           column='new-col',
#           value=new_col_value)
df.head()
df_target_value = df.iloc[0:, df.shape[1] - 1].values
# target_value = np.array(df_target_value)

misclassified_value = df.isnull().sum()
print("Missing values per column:\n", misclassified_value)

df = pd.DataFrame(df)
# print the original DataFrame
print("Original DataFrame :")
print(df)

# shuffle the DataFrame rows
df = df.sample(frac=1)
df_train = df[:248]
print('df_train: ', df_train)

df_test = df[249:]
print('df_test: ', df_test)

train_examples = np.array(df_train.iloc[:, :])
test_example = np.array(df_test.iloc[:, :])

# weights:
# Set the random seed for reproducibility
np.random.seed(1)
# Initialize weights
n_features = df.shape[1] - 1  # Number of features in your dataset
weights = np.random.uniform(low=-0.001, high=0.001, size=n_features)
# add w0
weights = np.append(-0.2, weights)
delta_w = np.array([0.0 for i in range(len(weights))])
rates = [0.01, 0.02, 0.05, 0.07, 0.8]
misclassified = 1
misclassified_ = []
correct_classified_ = []
weights_Ranes = []
performances_Rates_Weights_Train = []
Train_Rates = []
Train_Performance = []
Train_Weights = []
all_Train_weights_cuccess = []
epochs = 5

# the file path to save data
csv_file_path = './data/weights_data.csv'


def perceptron_algorithm(misclassified, rate):
    # while misclassified != 0:
    # weights_Train_cuccess = []
    for epoch in range(epochs):
        misclassified = 0
        correct_classified = 0
        weights_Train_cuccess = []
        for i in range(len(train_examples)):
            sum = 0
            isinstance = train_examples[i]
            for j in range(len(isinstance) - 1):
                sum = weights[j+1] * isinstance[j]
            sum += weights[0]
            if sum > 0:
                perceptron_output = 1
            else:
                perceptron_output = -1
            target_value = isinstance[len(isinstance) - 1]
            if perceptron_output != target_value:
                misclassified += 1
                # delta_W0
                delta_w[0] += rate * \
                    (target_value - perceptron_output)
                weights[0] += delta_w[0]
                # delta_w = rate * (target_value - perceptron_output) * input_value
                for k in range(len(isinstance)-1):
                    delta_w[k+1] += rate * (target_value -
                                            perceptron_output) * isinstance[k]
                    # w_i = w_i + delta_w_i
                    weights[k+1] += delta_w[k+1]
            else:
                if len(weights_Train_cuccess) == 0:
                    correct_classified = 1
                    weights_Train_cuccess.append(
                        {'rate': rate, 'correct_classified': correct_classified, 'weights': tuple(weights)})
                else:
                    correct_classified += 1
                    find_item = ()
                    for n in range(len(weights_Train_cuccess)):
                        item = weights_Train_cuccess[n]['weights']
                        if np.array_equal(item, weights):
                            find_item = item
                            break
                    if find_item:
                        weights_Train_cuccess[n]['correct_classified'] = correct_classified
                    else:
                        weights_Train_cuccess.append(
                            {'rate': rate, 'correct_classified': correct_classified, 'weights': tuple(weights)})
    # all_Train_weights_cuccess.append(weights_Train_cuccess)
    all_Train_weights_cuccess.append({rate: weights_Train_cuccess})

    # Select the max performance classified
    if len(weights_Train_cuccess) != 0:
        max_Classified_Item = max(
            weights_Train_cuccess, key=lambda x: x['correct_classified'])
        max_performance = max_Classified_Item['correct_classified'] / \
            len(train_examples) * 100.0
        rate_max_Classified_Item = max_Classified_Item['rate']
        weights_max_Classified_Item = max_Classified_Item['weights']
        Train_Rates.append(rate_max_Classified_Item)
        Train_Performance.append(max_performance)
        Train_Weights.append(weights_max_Classified_Item)
        performances_Rates_Weights_Train.append(
            {'rate': rate_max_Classified_Item, 'performance': max_performance, 'weights': weights_max_Classified_Item})
        # csv file for weights_Train_cuccess
        weights_data = pd.DataFrame(performances_Rates_Weights_Train)
        weights_data.to_csv(csv_file_path, index=False)

# for x in performances_Rates_Weights_Train:
#     Train_Rates.append(x['rate'])
#     Train_Performance.append(x['performance'])
#     Train_Weights.append(x['performance'])


# Test
weights_test = performances_Rates_Weights_Train
performances_Test = []
rates_Test = []


def perceptron_test(weights_test):

    for k in range(len(weights_test)):
        misclassified_test = 0
        classified_test = 0
        performance = 0
        weight_t = weights_test[k]['weights']
        rate_test = weights_test[k]['rate']
        rates_Test.append(rate_test)

        for i in range(len(test_example)):
            sum_test = 0
            isinstance_Test = test_example[i]
            for j in range(len(isinstance_Test) - 1):
                sum_test += weight_t[j+1] * isinstance_Test[j]
            # w0_test
            sum_test += weight_t[0]

            if sum_test > 0:
                perceptron_output_test = 1
            else:
                perceptron_output_test = -1

            target_value_Test = isinstance_Test[len(isinstance_Test) - 1]
            if perceptron_output_test != target_value_Test:
                misclassified_test += 1
            else:
                classified_test += 1

        performance = classified_test / len(test_example) * 100.0
        performances_Test.append(performance)


# perceptron_algorithm
for i in range(len(rates)):
    perceptron_algorithm(misclassified, rates[i])
perceptron_test(weights_test)

# Create a line plot
plt.plot(Train_Rates, Train_Performance, label='Train', marker='o')
plt.plot(rates_Test, performances_Test, label='Test', marker='o')
# Add labels and title
plt.xlabel('rates')
plt.ylabel('performance')
plt.title('Rate vs Performance Train')
plt.legend()
# Show the plot
plt.show()
