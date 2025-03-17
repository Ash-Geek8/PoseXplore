import sqlite3
conn = sqlite3.connect('angle_data.db')
c = conn.cursor()
def dview():
    c.execute('Select * from angle_data')
    row = c.fetchall()
    for i in row:
        for j in i:
            print(j,end='\t')
        print()
def ddel():
    c.execute('drop table angle_data')
#ddel()
dview()