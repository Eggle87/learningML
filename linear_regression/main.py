#data
import numpy as np
import pandas as pd

#ML
import keras
import ml_edu.experiment
import ml_edu.results

#data visualization
import plotly.express as px


chicago_taxi_dataset = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv")

# Updata dataframe to use specific columns
training_df = pd.DataFrame(chicago_taxi_dataset.loc[:, ('TRIP_MILES', 'TRIP_SECONDS', 'FARE', 'COMPANY', 'PAYMENT_TYPE', 'TIP_RATE')])


print("Read dataset completed successfully.")
print('Total number of rows: {0}\n\n'.format(len(training_df.index)))

# a='all'
# training_df.describe(include=a)
# training_df.head(200)

# answer = '''
# What is the maximum fare? 				              Answer: $159.25
# What is the mean distance across all trips? 		Answer: 8.2895 miles
# How many cab companies are in the dataset? 		  Answer: 31
# What is the most frequent payment type? 		    Answer: Credit Card
# Are any features missing data? 				          Answer: No
# '''

# # You should be able to find the answers to the questions about the dataset
# # by inspecting the table output after running the DataFrame describe method.
# #
# # Run this code cell to verify your answers.

# # What is the maximum fare?
# max_fare = training_df['FARE'].max()
# print("What is the maximum fare? 				Answer: ${fare:.2f}".format(fare = max_fare))

# # What is the mean distance across all trips?
# mean_distance = training_df['TRIP_MILES'].mean()
# print("What is the mean distance across all trips? 		Answer: {mean:.4f} miles".format(mean = mean_distance))

# # How many cab companies are in the dataset?
# num_unique_companies =  training_df['COMPANY'].nunique()
# print("How many cab companies are in the dataset? 		Answer: {number}".format(number = num_unique_companies))

# # What is the most frequent payment type?
# most_freq_payment_type = training_df['PAYMENT_TYPE'].value_counts().idxmax()
# print("What is the most frequent payment type? 		Answer: {type}".format(type = most_freq_payment_type))

# # Are any features missing data?
# missing_values = training_df.isnull().sum().sum()
# print("Are any features missing data? 				Answer:", "No" if missing_values == 0 else "Yes")


# training_df.corr(numeric_only = True)

# matrix = px.scatter_matrix(training_df, dimensions=["FARE", "TRIP_MILES", "TRIP_SECONDS"])
# print(ml_edu.experiment.ExperimentSettings.input_features)

def create_model(
        settings: ml_edu.experiment.ExperimentSettings,
        metrics: list[keras.metrics.Metric]
) -> keras.Model:
    """Create and compile a single linear regression model"""
    #Describe topopgraphy of the model
    #The topography of a simple linear regression model
    # is a single node in a single layer
    inputs = {name: keras.Input(shape=(1,), name=name) for name in settings.input_features}
    concatenated_inputs = keras.layers.Concatenate()(list(inputs.values()))
    outputs = keras.layers.Dense(units=1)(concatenated_inputs)
    model = keras.Model(inputs=inputs,outputs=outputs)

    # Compile the model topography into code that keras can efficiently
    # execute. Configure training to minimize the model's mean squared error

    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=settings.learning_rate),
                  loss="mean_squared_error",
                  metrics=metrics)
    return model 

def train_model(
        experiment_name:str,
        model: keras.Model,
        dataset: pd.DataFrame,
        label_name: str,
        settings: ml_edu.experiment.ExperimentSettings,
) -> ml_edu.experiment.Experiment:
    """Train the model by feeding it data"""

    # Feed the model the feature and add the label
    # The model will train for the specified number of epochs
    features={name:dataset[name].values for name in settings.input_features}
    label = dataset[label_name].values
    history = model.fit(x=features,
                        y=label,
                        batch_size=settings.batch_size,
                        epochs=settings.number_epochs)
    return ml_edu.experiment.Experiment(
        name=experiment_name,
        settings=settings,
        model=model,
        epochs=history.epoch,
        metrics_history=pd.DataFrame(history.history)
    )

print("SUCCESS: defining linear regression functions complete.")

#@title Code - Experiment 1
# # The following variables are the hyperparameters.
# settings_1 = ml_edu.experiment.ExperimentSettings(
#     learning_rate = 0.001,
#     number_epochs = 20,
#     batch_size = 50,
#     input_features = ['TRIP_MILES']
# )

# metrics = [keras.metrics.RootMeanSquaredError(name='rmse')]

# model_1 = create_model(settings_1, metrics)

# experiment_1 = train_model('one_feature', model_1, training_df, 'FARE', settings_1)

# ml_edu.results.plot_experiment_metrics(experiment_1, ['rmse'])
# ml_edu.results.plot_model_predictions(experiment_1, training_df, 'FARE')


# #@title Code - Experiment 2

# # The following variables are the hyperparameters.
# # TODO - Adjust these hyperparameters to see how they impact a training run.
# settings_2 = ml_edu.experiment.ExperimentSettings(
#     learning_rate = 0.001,
#     number_epochs = 20,
#     batch_size = 50,
#     input_features = ['TRIP_MILES']
# )

# metrics = [keras.metrics.RootMeanSquaredError(name='rmse')]

# model_2 = create_model(settings_2, metrics)

# experiment_2 = train_model('one_feature_hyper', model_2, training_df, 'FARE', settings_2)

# ml_edu.results.plot_experiment_metrics(experiment_2, ['rmse'])
# ml_edu.results.plot_model_predictions(experiment_2, training_df, 'FARE')


#@title Code - Experiment 3

# The following variables are the hyperparameters.
settings_3 = ml_edu.experiment.ExperimentSettings(
    learning_rate = 0.0005,
    number_epochs = 50,
    batch_size = 50,
    input_features = ['TRIP_MILES', 'TRIP_MINUTES']
)

training_df['TRIP_MINUTES'] = training_df['TRIP_SECONDS']/60

metrics = [keras.metrics.RootMeanSquaredError(name='rmse')]

model_3 = create_model(settings_3, metrics)

experiment_3 = train_model('two_features', model_3, training_df, 'FARE', settings_3)

ml_edu.results.plot_model_predictions(experiment_3, training_df, 'FARE')



#@title Code - Define functions to make predictions
def format_currency(x):
  return "${:.2f}".format(x)

def build_batch(df, batch_size):
  batch = df.sample(n=batch_size).copy()
  batch.set_index(np.arange(batch_size), inplace=True)
  return batch

def predict_fare(model, df, features, label, batch_size=50):
  batch = build_batch(df, batch_size)
  predicted_values = model.predict_on_batch(x={name: batch[name].values for name in features})

  data = {"PREDICTED_FARE": [], "OBSERVED_FARE": [], "L1_LOSS": [],
          features[0]: [], features[1]: []}
  for i in range(batch_size):
    predicted = predicted_values[i][0]
    observed = batch.at[i, label]
    data["PREDICTED_FARE"].append(format_currency(predicted))
    data["OBSERVED_FARE"].append(format_currency(observed))
    data["L1_LOSS"].append(format_currency(abs(observed - predicted)))
    data[features[0]].append(batch.at[i, features[0]])
    data[features[1]].append("{:.2f}".format(batch.at[i, features[1]]))

  output_df = pd.DataFrame(data)
  return output_df

def show_predictions(output):
  header = "-" * 80
  banner = header + "\n" + "|" + "PREDICTIONS".center(78) + "|" + "\n" + header
  print(banner)
  print(output)
  return

output=predict_fare(experiment_3.model, training_df, experiment_3.settings.input_features, 'FARE')
show_predictions(output)