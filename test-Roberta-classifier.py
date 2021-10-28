import pandas as pd
from simpletransformers.classification import ClassificationModel
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

args={'train_batch_size':8, 'learning_rate': 3e-6, 'num_train_epochs': 1, 'max_seq_length': 512, 'overwrite_output_dir': True}
model = ClassificationModel("roberta", "outputs/checkpoint-2000")
from sklearn.model_selection import train_test_split
train_df = pd.read_csv("train.csv",index_col=False)
#print(train_df.columns())
train_df=train_df[['text','labels']]
train, test = train_test_split(train_df, test_size=0.2)

#model.train_model(train)

print(test.labels.unique())
predictions, raw_outputs = model.predict(list(test.text.values))
y_true = test.labels.values
y_pred = predictions
print(classification_report(y_true, y_pred, target_names=[ str(i) for i in test.labels.unique()]))
