import pandas as pd

# +-----------+-----------+------------+--------------+
# | player_id | device_id | event_date | games_played |
# +-----------+-----------+------------+--------------+
# | 1         | 2         | 2016-03-01 | 5            |
# | 1         | 2         | 2016-05-02 | 6            |
# | 2         | 3         | 2017-06-25 | 1            |
# | 3         | 1         | 2016-03-02 | 0            |
# | 3         | 4         | 2018-07-03 | 5            |
# +-----------+-----------+------------+--------------+
def player_first_login(activity: pd.DataFrame) -> pd.DataFrame:
    return activity.groupby('player_id').agg({ 'event_date': min }).rename(columns={ 'event_date': 'first_login' }).reset_index()

input_table = { 'player_id': [1, 1, 2, 3, 3], 'device_id':[2, 2, 3, 1, 4], 'event_date': ['2016-03-01', '2016-05-02', '2017-06-25', '2016-03-02', '2018-07-03'], 'games_played': [5, 6, 1, 0, 5] }
expected_table = { 'player_id': { 0: 1, 1: 2, 2: 3 }, 'first_login': { 0: '2016-03-01', 1: '2017-06-25', 2: '2016-03-02' } }
result_table = player_first_login(pd.DataFrame(input_table)).to_dict()
print(f'input: {input_table}\nexpected: {expected_table}\nresult: {result_table}')
assert (result_table == expected_table)
