import pandas as pd

DEBUG = False

path_train_data_list = ['train_data/aalborg.csv', 'train_data/alpine-1.csv', 'train_data/f-speedway.csv']
#path_train_data_list = ['train_data/alpine-1.csv']
#path_train_data = 'train_data/alpine-1.csv'
race_df = None
for data_path in path_train_data_list:
    if race_df is None:
        race_df = pd.read_csv(data_path)    
    else:
        race_df.append(pd.read_csv(data_path), ignore_index=True)

race_df = race_df.fillna(0.0)

output_key = ['ACCELERATION', 'BRAKE', 'STEERING']
#output_key = ['ACCELERATION','BRAKE', 'TRACK_POSITION']
remove_key = []
input_key = [x for x in race_df if x not in output_key]
input_key = [x for x in input_key if x not in remove_key]

input_data = race_df[input_key].values
output_data = race_df[output_key].values

print(output_data)

if DEBUG:
    print(input_key)
    print(output_key)
    print(input_data)
    print(output_data)
