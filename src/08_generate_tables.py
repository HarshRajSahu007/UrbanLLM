import pandas as pd

IN_FILE="data/results/routed_complaints.csv"

df=pd.read_csv(IN_FILE)

# Table 1: Dataset distribution
table1=df["category"].value_counts().reset_index()
table1.columns= ["Complaint Category","Number of Samples"]
table1.to_csv("paper/tables/table1_dataset_distribution.csv",index=False)

# Table 2: Priority distribution
table2=df["priority_level"].value_counts().reset_index()
table2.columns= ["Priority Level","Number of Complaints"]
table2.to_csv("paper/tables/table2_priority_distribution.csv",index=False)

# Table 3: Department routing distribution
table3=df["assigned_department"].value_counts().reset_index()
table3.columns= ["Assigned Department","Number of Complaints"]
table3.to_csv("paper/tables/table3_department_routing.csv",index=False)

print("Saved all paper tables.")