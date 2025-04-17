import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\jay\Downloads\archive (17)\Banglore_traffic_Dataset.csv")

print("Before cleaning:")
print(df.info())
print("\nMissing values before cleaning:\n", df.isnull().sum())

str_cols = df.select_dtypes(include='object').columns
df[str_cols] = df[str_cols].apply(lambda x: x.str.strip())

df = df[df['Area Name'].notnull() & (df['Area Name'] != '')]

if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

fill_mean = ['Traffic Volume', 'Average Speed']
for col in fill_mean:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mean())

if 'Congestion Level' in df.columns:
    df['Congestion Level'] = df['Congestion Level'].fillna(df['Congestion Level'].median())

unique_areas = df['Area Name'].unique()
print("Unique Area Names:\n", unique_areas)

if 'Traffic Volume' in df.columns:
    avg_traffic_per_area = df.groupby('Area Name')['Traffic Volume'].mean().sort_values(ascending=False)
    print("\nAverage Traffic Volume per Area:\n", avg_traffic_per_area)
    
print("\nAfter cleaning:")
print(df.info())
print("\nMissing values after cleaning:\n", df.isnull().sum())

plt.figure(figsize=(12, 6))
avg_traffic_per_area.plot(kind='bar', color='skyblue')
plt.title("Average Traffic Volume per Area")
plt.xlabel("Area Name")
plt.ylabel("Average Traffic Volume")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

df.to_csv("/mnt/data/Banglore_traffic_Dataset_cleaned.csv", index=False)

df.head()
