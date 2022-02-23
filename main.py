#! python36

#1 Importing libraries and downloading packages

import nltk #nltk is a natural language tool kit
import numpy as np

# downloading model to tokenize message
nltk.download('punkt')
# downloading stopwords
nltk.download('stopwords')
# downloading wordnet, which contains all lemmas of english language (lemmas are used to shorten the words Ex: removing 'ing' from downloading etc)
nltk.download('wordnet')
# Got an error while running fixed it by adding:
nltk.download('omw-1.4')

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

from nltk.stem import WordNetLemmatizer
print(stop_words)

#2 Function to clean text

def clean_corpus(corpus):
  # lowering every word in text
  corpus = [ doc.lower() for doc in corpus]
  cleaned_corpus = []
  
  stop_words = stopwords.words('english')
  wordnet_lemmatizer = WordNetLemmatizer()

  # iterating over every text
  for doc in corpus:
    # tokenizing text
    tokens = word_tokenize(doc)
    cleaned_sentence = [] 
    for token in tokens: 
      # removing stopwords, and punctuation
      if token not in stop_words and token.isalpha(): 
        # applying lemmatization
        cleaned_sentence.append(wordnet_lemmatizer.lemmatize(token)) 
    cleaned_corpus.append(' '.join(cleaned_sentence))
  return cleaned_corpus

#3 Loading and cleaning intents

import json
with open('intents.json', 'r') as file:
  intents = json.load(file)

corpus = []
tags = []

for intent in intents['intents']:
    # taking all patterns in intents to train a neural network
    for pattern in intent['patterns']:
        corpus.append(pattern)
        tags.append(intent['tag'])

cleaned_corpus = clean_corpus(corpus)
print(cleaned_corpus)

#4 Vectorizing intents

from sklearn.feature_extraction.text import TfidfVectorizer # Converting the words of patterns into numbers

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(cleaned_corpus)

from sklearn.preprocessing import OneHotEncoder  # Converting the words of tags into numbers

encoder = OneHotEncoder()
y = encoder.fit_transform(np.array(tags).reshape(-1,1))

#5 Training neural network

# Training the modal
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential([
                    Dense(128, input_shape=(X.shape[1],), activation='relu'),
                    Dropout(0.2), # removing certain neurons
                    Dense(64, activation='relu'),
                    Dropout(0.2),
                    Dense(y.shape[1], activation='softmax')
])
# "loss='categorical_crossentropy'" finds the difference between the actual output given by the modal and the desired output 
# and depending on the difference "optimizer='adam'" reassigns the weights
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(X.toarray(), y.toarray(), epochs=20, batch_size=1)
# epochs=20 means we are running the modal 20 times to make it more accurate(matching the patterns(customer's input) with the tag - intent)

'''
#6 Classifying messages to intent
> If the intent probability does not match with any intent, then send it to no answer.
> Get Intent
> Perform Action
'''

# if prediction for every tag is less than 0.4, then we want to classify that message as noanswer
INTENT_NOT_FOUND_THRESHOLD = 0.40

def predict_intent_tag(message):
  message = clean_corpus([message])
  X_test = vectorizer.transform(message)
  #print(message)
  #print(X_test.toarray())
  y = model.predict(X_test.toarray()) # trying to match customer's input to the tag - intent
  #print (y)
  # if probability of all intent is low, classify it as noanswer
  if y.max() < INTENT_NOT_FOUND_THRESHOLD:
    return 'noanswer'
  
  prediction = np.zeros_like(y[0])
  prediction[y.argmax()] = 1
  tag = encoder.inverse_transform([prediction])[0][0]
  return tag

print(predict_intent_tag('How you could help me?')) # here the pattern(customer's input) matches the tag - options => intent
print(predict_intent_tag('swiggy chat bot'))
print(predict_intent_tag('Where\'s my order'))

import random
import time 

def get_intent(tag):
  # to return complete intent from intent tag
  for intent in intents['intents']:
    if intent['tag'] == tag:
      return intent

def perform_action(action_code, intent):
  # funition to perform an action which is required by intent
  
  if action_code == 'CHECK_ORDER_STATUS':
    print('\n Checking database \n')
    time.sleep(2)
    order_status = ['in kitchen', 'with delivery executive']
    delivery_time = []
    return {'intent-tag':intent['next-intent-tag'][0],
            'order_status': random.choice(order_status),
            'delivery_time': random.randint(10, 30)}
  
  elif action_code == 'ORDER_CANCEL_CONFIRMATION':
    ch = input('BOT: Do you want to continue (Y/n) ?')
    if ch == 'y' or ch == 'Y':
      choice = 0
    else:
      choice = 1
    return {'intent-tag':intent['next-intent-tag'][choice]}
  
  elif action_code == 'ADD_DELIVERY_INSTRUCTIONS':
    instructions = input('Your Instructions: ')
    return {'intent-tag':intent['next-intent-tag'][0]}

#7 Complete chat bot

while True:
  # get message from user
  message = input('You: ')
  # predict intent tag using trained neural network
  tag = predict_intent_tag(message)
  # get complete intent from intent tag
  intent = get_intent(tag)
  # generate random response from intent
  response = random.choice(intent['responses'])
  print('Bot: ', response)

  # check if there's a need to perform some action
  if 'action' in intent.keys():
    action_code = intent['action']
    # perform action
    data = perform_action(action_code, intent)
    # get follow up intent after performing action
    followup_intent = get_intent(data['intent-tag'])
    # generate random response from follow up intent
    response = random.choice(followup_intent['responses'])
    
    # print randomly selected response
    if len(data.keys()) > 1:
      print('Bot: ', response.format(**data))
    else:
      print('Bot: ', response)

  # break loop if intent was goodbye
  if tag == 'goodbye':
    break
