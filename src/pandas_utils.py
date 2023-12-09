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
    pass

input_table = None
expected_table = None
result_table = None
print(f'input: {input_table}, expected: {expected_table}, result: {result_table}')
assert (result_table = expected_table)
