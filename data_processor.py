import pandas as pd

DEBUG = False

path_train_data = 'train_data/alpine-1.csv'

race_df = pd.read_csv(path_train_data)
race_df = race_df.fillna(0.0)

print(race_df)

output_key = ['ACCELERATION', 'BRAKE', 'STEERING']
input_key = [x for x in race_df if x not in output_key]

input_data = race_df[input_key].values
output_data = race_df[output_key].values

print(output_data)

if DEBUG:
    print(input_key)
    print(output_key)
    print(input_data)
    print(output_data)
