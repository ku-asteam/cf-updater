import csv
import sys
from collections import defaultdict
from typing import TextIO, Tuple

import pandas as pd
import surprise


def get_data_from_csv(data_file: TextIO) -> Tuple[list, list, list]:
    reader = csv.reader(data_file)

    result_data = list()
    for line in reader:
        result_data.append(line)

    # Pop header
    result_data.pop(0)

    user_list = sorted(set([item[1] for item in result_data]), key=lambda x: int(x))
    content_list = sorted(set([item[2] for item in result_data]), key=lambda x: int(x))

    return result_data, user_list, content_list


def add_data_from_new_file(new_file: TextIO, result_data: list, user_list: list, content_list: list, additional_user_size: int, remove_size: int) -> Tuple[list, list]:
    reader = csv.reader(new_file)

    new_data = list()
    for line in reader:
        new_data.append(line)

    # Pop header
    new_data.pop(0)

    new_user_dict = defaultdict(int)

    for row in new_data:
        if row[2] in content_list:
            new_user_dict[row[1]] = int(new_user_dict[row[1]]) + 1

    sorted_new_user = sorted(new_user_dict.items(), key=lambda x: x[1], reverse=True)

    new_user_list = list()
    current_iteration = 0
    for count, user_data in enumerate(sorted_new_user):
        if remove_size <= count and user_data[0] not in user_list and current_iteration < additional_user_size:
            new_user_list.append(user_data[0])
            current_iteration += 1
    new_user_list.sort(key=lambda x: int(x))

    user_list = sorted(user_list + new_user_list, key=lambda x: int(x))

    for value in new_data:
        if value[1] in new_user_list and value[2] in content_list:
            result_data.append(value)

    return result_data, user_list


# Make dataframe for training
def make_df(result_data: list, user_list: list, content_list: list) -> Tuple[pd.DataFrame, pd.DataFrame]:
    index = user_list
    columns = content_list
    df_index_list = pd.DataFrame(index=index, columns=columns)

    for value in result_data:
        df_index_list.loc[value[1]:value[1], value[2]] = float(value[3])

    columns = ['user', 'content', 'rating']
    df_user_content = pd.DataFrame(columns=columns)
    for value in result_data:
        df_user_content.loc[value[0]] = [value[1], value[2], float(value[3])]

    return df_index_list, df_user_content


# User-based collaborative filtering
def train(df_index_list: pd.DataFrame, df_user_content: pd.DataFrame, user_list: list, content_list: list) -> Tuple[pd.DataFrame, pd.DataFrame]:
    reader = surprise.dataset.Reader(rating_scale=(1, 5))
    data_folds = surprise.dataset.DatasetAutoFolds(reader=reader, df=df_user_content)
    trainset = data_folds.build_full_trainset()
    sim_options = {
        'name': 'Cosine',   # Cosine similarity
        'user_based': 'True'    # User-based CF
    }
    algo = surprise.KNNBasic(sim_options=sim_options)
    algo.fit(trainset)

    trained = list()  # For only predicted rating matrix
    trained_full = list()  # For complete matrix
    for user in user_list:
        for content in content_list:
            try:
                rating = df_index_list.loc[user, content]
                if pd.isnull(rating):
                    raise KeyError
                trained_full.append([user, content, rating])
            except KeyError:
                predict = algo.predict(str(user), str(content)).est
                trained.append([user, content, predict])
                trained_full.append([user, content, predict])

    # Only predicted rating matrix
    index = user_list
    columns = content_list
    trained_df_index_list = pd.DataFrame(index=index, columns=columns)
    for value in trained:
        trained_df_index_list.loc[value[0]:value[0], value[1]] = value[2]

    # Complete matrix
    index = user_list
    columns = content_list
    trained_full_df_index_list = pd.DataFrame(index=index, columns=columns)
    for value in trained_full:
        trained_full_df_index_list.loc[value[0]:value[0], value[1]] = value[2]

    return trained_df_index_list, trained_full_df_index_list


def content_update(result_data: list, user_list: list, content_list: list) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_index_list, df_user_content = make_df(result_data, user_list, content_list)
    trained_df_index_list, trained_full_df_index_list = train(df_index_list, df_user_content, user_list, content_list)

    return trained_df_index_list, trained_full_df_index_list


def main():
    data_file: TextIO = open(sys.argv[1], "r")
    new_file: TextIO = open(sys.argv[2], "r")
    output_file_trained: TextIO = open(sys.argv[3], "w")
    output_file_completed: TextIO = open(sys.argv[4], "w")
    additional_user_size: int = int(sys.argv[5])
    remove_size: int = int(sys.argv[6])

    result_data, user_list, content_list = get_data_from_csv(data_file)
    result_data, user_list = add_data_from_new_file(new_file, result_data, user_list, content_list, additional_user_size, remove_size)

    trained_df_index_list, trained_full_df_index_list = content_update(result_data, user_list, content_list)

    trained_df_index_list.to_csv(output_file_trained, sep=',')
    trained_full_df_index_list.to_csv(output_file_completed, sep=',')

    data_file.close()
    new_file.close()
    output_file_trained.close()
    output_file_completed.close()


if __name__ == "__main__":
    main()
