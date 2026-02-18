import pandas as pd
import sqlite3

# Connect to SQLite database
cnx = sqlite3.connect('db.sqlite')

# Read CSV
df = pd.read_csv('anime_dataset_cleaned.csv')

# Save to SQLite table (replace if exists)
df.to_sql('anime', cnx, if_exists='replace', index=False)

# Close connection
cnx.close()
print("CSV imported to SQLite successfully!")
