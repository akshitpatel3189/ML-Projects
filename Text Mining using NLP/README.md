#   Text Mining using NLP

In this project, I explore different aspects of text mining and natural language processing  
(NLP) and apply it to analyze emotions in text data.

# Tools and Libraries

I am using Jupyter Notebook for this project.
For running this project I have used the following Python libraries
- pandas
- spcy
- NumPy
-  matplotlib
- lda
- pprint
- bokeh
- nltk
- sklearn
- emoji


# Description:

I have used the IMDB movie reviews dataset which is provided in the dataset_link.txt file.

- First I perform **exploratory data analysis**. In this, I check the distribution of sentiments/emotions, the number of unique words, the average review length, etc. Then I preprocess the text data by tokenizing the reviews/tweets, removing stop words, and applying stemming/lemmatization. Afterward, I perform **part-of-speech (POS) tagging** on the preprocessed text data.

- After preprocessing I did Sentiment Analysis in Two ways.
 
1.  **Sentiment Analysis with Bag of Words (BoW) and TF-IDF:** where I converted the preprocessed IMDB movie reviews into a matrix of token counts with the  CountVectorizer from sklearn and trained a logistic regression classifier on BoW and TF-IDF representations with accuracy, precision, recall, and F1-score. Then compare the performances of  BoW and TF-IDF.
2. **Sentiment Analysis with Word Embeddings:** Where I used the Word2Vec model, which convert the preprocessed movie reviews into vectors and trains a logistic regression classifier with accuracy, precision, recall, and F1-score. Then compare the performances of  BoW, TF-IDF, and Word2Vec.

I also did **Emotion Classification** on the SemEval-2018 Task 1 dataset. I built two classification models to classify tweets into various emotion categories: one using the BoW method and one using Word2Vec and compared the performance of the two models. After that perform topic modeling using Latent Dirichlet Allocation (LDA). Visualize the top words for each topic represented based on these words.
