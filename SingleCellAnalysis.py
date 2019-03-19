import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


dataset = pd.read_csv("SingleCellRNA_edit.csv")

gene_exp = dataset.iloc[:, 1:2554].values  
cell_stage = dataset.iloc[:,2554].values

#convert int to string for classification    
cell_stage = list(map(str, cell_stage))

#split data into training and testing set

train_test = train_test_split(gene_exp, cell_stage, test_size=0.2, random_state=0)  
gene_expTrain = train_test[0]
gene_expTest = train_test[1]
cell_stageTrain = train_test[2]
cell_stageTest = train_test[3]

#feature scaling (optional)

#sc = StandardScaler()  
#GE_train = sc.fit_transform(gene_expTrain)  
#GE_test = sc.transform(gene_expTest)

#Random Forest 

rfc = RandomForestClassifier(n_estimators=20, class_weight = {'1':1, '2':1, '3':1, '4':1, '5':5})
rfc.fit(gene_expTrain,cell_stageTrain)

# predictions

rfc_predict = rfc.predict(gene_expTest)

#evaluate predictions

print(confusion_matrix(cell_stageTest,rfc_predict))  
print(classification_report(cell_stageTest,rfc_predict))  
print(accuracy_score(cell_stageTest,rfc_predict)) 
