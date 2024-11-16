import csv
import mysql.connector
import pandas as pd

db = 'BS_AMF22_5M'
cnx = mysql.connector.connect(user='traffic', password='traffic1',
                              host='172.21.62.196',
                              database=db)

cursor = cnx.cursor()

query = f"SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = '{db}'"

cursor.execute(query)

tables = cursor.fetchall()

with open('results.csv', 'w', newline='') as csvfile:
    fieldnames = ['DB_name', 'table_name', 'port', 'col_name', 'col_min', 'col_max', 'col_avg', 'col_var']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for table_name in tables:
        table_name = table_name[0]
        print("table_name:", table_name)
        table_df =pd.DataFrame()
        # Get the columns for the table
        cursor.execute(f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = '{db}' AND TABLE_NAME = '{table_name}'")
        columns = [column[0] for column in cursor.fetchall()]
        e=0
        for column in columns:
            if column in ['DATETIME','SYSTEM','PORT']:  # Skip the 'PORT' column
                continue

            print("column_name", column)
            
            # Check if the column exists in the table
            if column not in columns:
                print(f"Column '{column}' does not exist in table '{table_name}'. Skipping.")
                continue

            # Get the unique ports in the table
            cursor.execute(f"SELECT DISTINCT PORT FROM `{table_name}`")
            ports = [port[0] for port in cursor.fetchall()]
            
            cursor.execute(f"SELECT * FROM `{table_name}`")
            df = cursor.fetchall()
            df = pd.DataFrame(df, columns=columns)
            #print(df.head())
            
            for port in ports:
                # Calculate min, max, avg, and var for each unique port and column
                
                tmp_df  = df.loc[df["PORT"]==port,column]
                print("tmp_df",port)
                print(tmp_df.shape)
                
                
                col_min = tmp_df.min()
                col_max = tmp_df.max()
                col_avg = tmp_df.mean()
                col_var = tmp_df.var()
                writer.writerow({'DB_name': db, 'table_name': table_name, 'port': port, 'col_name': column, 'col_min': col_min, 'col_max': col_max, 'col_avg': col_avg, 'col_var': col_var})

                row_df = pd.DataFrame({'DB_name': db, 'table_name': table_name, 'port': port, 'col_name': column, 'col_min': col_min, 'col_max': col_max, 'col_avg': col_avg, 'col_var': col_var ,'shape':[tmp_df.shape]},index=[e])
                table_df = pd.concat([table_df,row_df],axis=0)
                e += 1
        table_name2 = table_name.replace("|","_")
        table_df.to_csv(f"D:/table_df/{table_name2}_{db}.csv")

cursor.close()
cnx.close()