import pandas as pd
from simpletransformers.classification import ClassificationModel
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

args={'train_batch_size':8, 'learning_rate': 3e-6, 'num_train_epochs': 1, 'max_seq_length': 512, 'overwrite_output_dir': True}
model = ClassificationModel(
    "roberta", "roberta-base",num_labels=3,args=args,
)
from sklearn.model_selection import train_test_split
train_df = pd.read_csv("train.csv",index_col=False)
#print(train_df.columns())
train_df=train_df[['text','labels']]
train, test = train_test_split(train_df, test_size=0.2)

model.train_model(train)


predictions, raw_outputs = model.predict(list(test.text.values))
print('Accuracy on Test Data:', accuracy_score(test['labels'],predictions))
print('F1-Score on Test Data:', f1_score(test['labels'],predictions))
print('Confusion Matrix on Test Data:\n', confusion_matrix(test['labels'],predictions))
print('Classification on Test Data:\n', classification_report(test['labels'],predictions))
