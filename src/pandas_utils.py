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

# Employee
# +-------------+---------+
# | Column Name | Type    |
# +-------------+---------+
# | empId       | int     |
# | name        | varchar |
# | supervisor  | int     |
# | salary      | int     |
# +-------------+---------+
# Bonus
# +-------------+------+
# | Column Name | Type |
# +-------------+------+
# | empId       | int  |
# | bonus       | int  |
# +-------------+------+
def low_bonus_employees(employees: pd.DataFrame, bonuses: pd.DataFrame, exc_limit: int) -> pd.DataFrame:
    employees_bonuses = pd.merge(employees, bonuses, how='left', on='empId')
    employees_bonuses = employees_bonuses.fillna(0)
    employees_bonuses['bonus'] = employees_bonuses['bonus'].astype(int)
    result = employees_bonuses[employees_bonuses['bonus'] < exc_limit]
    result.loc[:'bonus'] = result.loc[:'bonus'].replace(0, None)
    return result[['name', 'bonus']]


input_table = { 'player_id': [1, 1, 2, 3, 3], 'device_id':[2, 2, 3, 1, 4], 'event_date': ['2016-03-01', '2016-05-02', '2017-06-25', '2016-03-02', '2018-07-03'], 'games_played': [5, 6, 1, 0, 5] }
expected_table = { 'player_id': { 0: 1, 1: 2, 2: 3 }, 'first_login': { 0: '2016-03-01', 1: '2017-06-25', 2: '2016-03-02' } }
result_table = player_first_login(pd.DataFrame(input_table)).to_dict()
print(f'input: {input_table}\nexpected: {expected_table}\nresult: {result_table}')
assert (result_table == expected_table)
print()

employees = { 'empId': [3, 1, 2, 4], 'name':['Brad', 'John', 'Dan', 'Thomas'], 'supervisor': [None, 3, 3, 3], 'salary': [4000, 1000, 2000, 4000] }
bonuses = { 'empId': [2, 4], 'bonus': [500, 2000] }
expected = { 'name': { 0: 'Brad', 1: 'John', 2: 'Dan' }, 'bonus': { 0: None, 1: None, 2: 500 } }
result = low_bonus_employees(pd.DataFrame(employees), pd.DataFrame(bonuses), 1000).to_dict()
print(f'employees: {employees}\nbonuses: {bonuses}\nexpected: {expected}\nresult: {result}')
assert (result == expected)
print()
