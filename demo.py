import streamlit as st
import pandas as pd
from simpletransformers.classification import ClassificationModel
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

args={'train_batch_size':8, 'learning_rate': 3e-6, 'num_train_epochs': 1, 'max_seq_length': 512, 'overwrite_output_dir': True}
model = ClassificationModel("bert", "checkpoint-7512-epoch-2/")

def predict(newsArticle):
    outputs = ["Left", "Centre", "Right"]
    prediction, raw_output = model.predict(newsArticle)
    return outputs[prediction]

st.title("Identifying Political Bias in a news article")
user_input = st.text_area("News article to test", "")
if st.button('Run Model') and user_input:
    bias = predict(user_input)
    st.write("### Given News article is " + bias + " biased")