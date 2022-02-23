# Swiggy-Chatbot
NLP, Neural networks

https://user-images.githubusercontent.com/91654378/155391920-107107d7-416b-489f-88cb-a2e74635514a.mp4

# About

---

A chatbot is artificial intelligence (AI) software that can imitate a natural language discussion (or chat) with a user via messaging apps, websites or mobile apps. What is the significance of chatbots? A chatbot is frequently described as one of the most advanced and promising forms of human-machine interaction. Chatbots can automatically simulate interactions with customers based on a set of predefined conditions or events.

From a technology standpoint, a chatbot is simply the next step in the evolution of a Question Answering system that uses Natural Language Processing (NLP).

<p align="center">
<img alt="Visual Studio Code" width="700px" src="https://github.com/harshithvh/Swiggy-Chatbot/blob/main/images/img1.png" />
</p><br>

Today's customers want immediate resolution of their queries.
Chatbots can be used to assist users for a specific task, whenever they want. And as chatbots do not get tired or bored, they can be employed to provide customer service round the clock. Chatbot applications streamline interactions between people and services, enhancing customer experience.

# Types of Chatbots
# 1 Generative Based

---

Generative chatbots use a combination of supervised learning, unsupervised learning & reinforcement learning. A generative chatbot is an open-domain chatbot that creates unique language combinations rather than selecting from a list of pre-defined responses. Retrieval-based systems are limited to predefined responses. Chatbots that use generative methods can generate new dialogue based on large amounts of conversational training data.

<p align="center">
<img alt="Visual Studio Code" width="700px" height="600px" src="https://github.com/harshithvh/Swiggy-Chatbot/blob/main/images/img2.png" />
</p><br>

# 2 Retrieval Based

---

Retrieval based chatbots, employ techniques such as keyword matching, machine learning, and deep learning to find the most appropriate response. These chatbots, regardless of technology, solely deliver predefined responses and do not generate fresh output. From a database of predefined responses, the chatbot is trained to offer the best possible response. The responses are based on previously collected data.

<p align="center">
<img alt="Visual Studio Code" width="700px" height="600px" src="https://github.com/harshithvh/Swiggy-Chatbot/blob/main/images/img3.png" />
</p><br>

Since the Generative Based Chatbots require huge amount of data to train on and construct the appropriate response, we will be using Retrieval Based approach for our chatbot.

# Technology used
# Natural Language Processing

---

Natural language processing is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data.

We will be using a four-step process to transform our "content" into a form that could be understood by a computer algorithm and with which it can extract meaningful insights.

# #1 Tokenization

---

Tokenization is the process of breaking down sentence or paragraphs into smaller chunks of words called tokens.

<img align="left" alt="Visual Studio Code" width="820px" src="https://github.com/harshithvh/Swiggy-Chatbot/blob/main/images/img4.png" />

# #2 Stop Words Removal

---

On removal of some words, the meaning of the sentence doesn't change, like and, am. Those words are called stop-words and should be removed before feeding to any algorithm. In datasets, some non-stop words repeat very frequently. Those words too should be removed to get an unbiased result from the algorithm.

<img align="left" alt="Visual Studio Code" width="820px" src="https://github.com/harshithvh/Swiggy-Chatbot/blob/main/images/img5.png" />

# #3 Lemmatization

---

Lemmatization is the process of converting a word to its base form. It considers the context and converts the word to its meaningful base form, whereas stemming just removes the last few characters, often leading to incorrect meanings and spelling errors.

For example, lemmatization would correctly identify the base form of ‘moving’ to ‘move’.

‘Moving’ -> Lemmatization -> ‘Move’

# #4 Vectorization

---

After tokenization, and stop words removal, our "content" are still in string format. We need to convert those strings to numbers based on their importance (features). We use TF-IDF vectorization to convert those text to vector of importance. With TF-IDF we can extract important words in our data. It assign rarely occurring words a high number, and frequently occurring words a very low number.

You can learn more about it from: https://en.wikipedia.org/wiki/Tf-idf

# Intent Classification

---

Intent recognition is a form of natural language processing (NLP), a subfield of artificial intelligence. 
Intent classification or intent recognition is the task of taking a written or spoken input, and classifying it based on what the user wants to achieve. Intent recognition forms an essential component of chatbots and finds use in sales conversions, customer support, and many other areas.

<img align="left" alt="Visual Studio Code" width="820px" src="https://github.com/harshithvh/Swiggy-Chatbot/blob/main/images/img6.png" />

In the above example, three different users are inquiring about their order status using different sentences but the underlying intent is to check what's the current status of their order.
