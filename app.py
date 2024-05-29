import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
port_stem = PorterStemmer()
vectorization = TfidfVectorizer()
import nltk
nltk.data.path.append('/')

vector_form = pickle.load(open('vector.pkl', 'rb'))
load_modelTree=pickle.load(open('modelTree.pkl', 'rb'))
load_LogisticRegression_model=pickle.load(open('LogisticRegression_model.pkl', 'rb'))
load_RandomForestClassifier_model=pickle.load(open('RandomForestClassifier_model.pkl', 'rb'))
load_svm_model=pickle.load(open('svm_model.pkl', 'rb'))

def stemming(content):
    con=re.sub('[^a-zA-Z]', ' ', content)
    con=con.lower()
    con=con.split()
    con=[port_stem.stem(word) for word in con if not word in stopwords.words('english')]
    con=' '.join(con)
    return con

def fake_news(news):
    news=stemming(news)
    input_data=[news]
    vector_form1=vector_form.transform(input_data)
    Treeprediction = load_modelTree.predict(vector_form1)
    if Treeprediction == 0:
        Treeprediction = -1
    Treeprediction=Treeprediction*0.8865384615384615

    LogisticRegressionprediction = load_LogisticRegression_model.predict(vector_form1)
    if LogisticRegressionprediction <0.5:
        LogisticRegressionprediction = -1
    else:
        LogisticRegressionprediction = 1
    LogisticRegressionprediction=LogisticRegressionprediction*0.9490384615384615

    RandomForestClassifierprediction = load_RandomForestClassifier_model.predict(vector_form1)
    if RandomForestClassifierprediction == 0:
        RandomForestClassifierprediction = -1
    RandomForestClassifierprediction=RandomForestClassifierprediction*0.9204326923076923

    svm_modelprediction = load_svm_model.predict(vector_form1)
    if svm_modelprediction == 0:
        svm_modelprediction = -1
    svm_modelprediction=svm_modelprediction*0.9620192307692308

    prediction=Treeprediction+LogisticRegressionprediction+RandomForestClassifierprediction+svm_modelprediction
      
    return prediction



if __name__ == '__main__':
    st.title('Fake News Classification app ')
    st.subheader("Input the News content below")
    sentence = st.text_area("Enter your news content here", "",height=200)
    predict_btt = st.button("predict")
    if predict_btt:
        prediction_class=fake_news(sentence)
        print(prediction_class)
        
        if prediction_class >0:
            st.warning('Unreliable')
        if prediction_class <0:
            st.success('Reliable')