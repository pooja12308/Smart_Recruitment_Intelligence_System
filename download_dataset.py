import pandas as pd

url = "https://raw.githubusercontent.com/611noorsaeed/Resume-Screening-App/main/UpdatedResumeDataSet.csv"

df = pd.read_csv(url)

print(df.shape)
print(df.columns)

df.to_csv("UpdatedResumeDataSet.csv", index=False)

print("Dataset downloaded successfully!")