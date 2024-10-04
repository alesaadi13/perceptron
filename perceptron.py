import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Load CSV file into DataFrame
file_path = './data/survey_lung_cancer_data.csv'
df = pd.read_csv(file_path)
# new_col_value = 1
# df.insert(loc=0,
#           column='new-col',
#           value=new_col_value)
df.head()
df_target_value = df.iloc[0:, df.shape[1] - 1].values
# target_value = np.array(df_target_value)

misclassified_value = df.isnull().sum()
print("Missing values per column:\n", misclassified_value)

df_train = df[:248]
print('df_train: ', df_train)

df_test = df[249:]
print('df_test: ', df_test)

train_examples = np.array(df_train.iloc[:, :])
test_example = np.array(df_test.iloc[:, :])

# weights
w0 = -0.8
random_state = 1
rgen = np.random.RandomState(random_state)
weights = rgen.normal(loc=0.0, scale=0.01, size=df.shape[1] - 1)
weights = np.append(w0, weights)
delta_w = np.array([0.0 for i in range(len(weights))])
rate = 0.1

misclassified = 1
misclassified_ = []


def perceptron_algorithm(misclassified):

    while misclassified != 0:
        misclassified = 0

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
                misclassified_.append(misclassified)
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
            break
    if misclassified == 0:
        print('success: ')
        # print('perceptron delta_w: ', delta_w)
        print('perceptron weights: ', weights)
    else:
        print('fail')


# Test
weights_test = [-0.6,         0.21624345, 13.79388244,  0.19471828,  0.38927031,  0.40865408,
                0.17698461,  0.21744812,  0.39238793,  0.20319039,  0.3975063,   0.41462108,
                0.37939859,  0.39677583,  0.39615946,  0.41133769]


def perceptron_test(weights_test):
    misclassified_test = 0
    classified_test = 0

    for i in range(len(test_example)):
        sum_test = 0
        isinstance = test_example[i]
        for j in range(len(test_example[i]) - 1):
            sum_test += weights_test[j+1] * isinstance[j]
        # w0_test
        sum_test += weights_test[0]
        if sum_test > 0:
            perceptron_output_test = 1
        else:
            perceptron_output_test = -1
        target_value = isinstance[len(isinstance) - 1]
        if perceptron_output_test != target_value:
            misclassified_test += 1
        else:
            classified_test += 1
    print('success test: ', classified_test)
    print('fail test: ', misclassified_test)


perceptron_algorithm(misclassified)
perceptron_test(weights_test)

# print('train_examples', len(train_examples.shape))
# misclassified_ = np.array(misclassified_)
# print('misclassified_', misclassified_.shape)
# epochs = np.arange(1, len(train_examples))
# plt.plot(epochs, misclassified_)
# plt.xlabel('iterations')
# plt.ylabel('misclassified')
# plt.show()
