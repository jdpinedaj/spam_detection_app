# %% Importing libraries
import joblib
import re
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
from lime.lime_text import LimeTextExplainer
import streamlit.components.v1 as components

# %% Header
st.write("# Spam-detection Model")

# Adding text input
message_text = st.text_input("Enter a message to evaluate if it is spam.")

# %% Loading the model
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text) 
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text

model = joblib.load('spam_classifier.joblib')


# %% Generating and Displaying predictions
def classify_message(model, message):  
    label = model.predict([message])[0]
    spam_prob = model.predict_proba([message])  
    return {'label': label, 'spam_probability': spam_prob[0][1]}

if message_text != '':
	result = classify_message(model, message_text)
	st.write(result)
	
	explain_pred = st.button('Explain Predictions')

#%% Using lime to explain the results
	if explain_pred:
		with st.spinner('Generating explanations'):
			class_names = ['ham', 'spam']
			explainer = LimeTextExplainer(class_names=class_names)
			exp = explainer.explain_instance(message_text, 
				model.predict_proba, num_features=10)
			components.html(exp.as_html(), height=800)
    
# %%
