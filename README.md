# CF-Updater
Collaborative Filtering (CF) Updater Module for Analysis in Mobile Networks

## Introduction
CF-Updater is a implementation of *a*STEAM Project (Next-Generation Information Computing Development Program through the National Research Foundation of Korea (NRF) funded by the Ministry of Science and ICT). The function of this module is to infer users' ratings for unrated contents, which thus provide an complete rating matrix. This module also updates the user-content rating matrix based on CF, in response to mobile networks environment in which new users are added.

## Requirements and Dependencies
- `python >= 3.7`
- `pandas >= 1.1`
- `surprise >= 0.1`

## Instructions
* Prepare datasets that contains a set of contents and a set of users who have reacted to some of the contents. (`data format: ['id', 'user', 'content', 'rating']`)
* Execute `cf-updater.py`

```shell script
$ python cf-updater.py DATA_FILE NEW_FILE INFERRED_FILE_PATH COMPLETED_FILE_PATH NEW_USER_SIZE REMOVE_SIZE
```

`DATA_FILE`: data file's path (`base`)

`NEW_FILE`: new file's path (`including new users' information`)

`INFERRED_FILE_PATH`: trained output file's path (`only inferred ratings`)

`COMPLETED_FILE_PATH`: completed(merged) output file's path

`NEW_USER_SIZE`: number of users to be added

`REMOVE_SIZE`: removing some users and contents to prevent matrix from becoming too dense (*same as uc-filter.py*)