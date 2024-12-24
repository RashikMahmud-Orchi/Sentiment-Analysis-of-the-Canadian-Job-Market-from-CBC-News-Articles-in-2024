# Sentiment Analysis of the Canadian Job Market from CBC News Articles in 2024

This project focuses on analyzing sentiment trends in the Canadian job market by leveraging textual data from CBC News articles. Using advanced Natural Language Processing (NLP) techniques and machine learning models, the study aims to uncover insights into public perception and sentiment shifts in 2024.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
4. [Technologies Used](#technologies-used)
5. [Installation](#installation)
6. [Usage](#usage)
7. [References](#references)


---

## Introduction

Understanding the sentiment of the Canadian job market can provide valuable insights into economic trends, public perception, and policy implications. This project processes, analyzes, and classifies news articles into sentiment categories (positive, negative, and neutral) using various embedding techniques and machine learning models. The analysis explores which embedding method and model provide the most accurate results for downstream classification tasks.

---

## Dataset

### Files:
1. `data.xlsx` - Contains unlabeled articles directly scraped using Selenium.
2. `labeled_data` - Contains data labeled by GPT-4 Mini+ with manual corrections for improved accuracy.

### Preprocessing Steps:
- Removal of duplicates and irrelevant data.
- Merging `Title` and `Description` fields into `Articles`.
- Lowercasing text, punctuation removal, and stopword filtering.
- Lemmatization applied for normalization.

---

## Methodology

### Embedding Techniques:
1. **Transformer-based Embeddings**:
   - Generated using DistilBERT, providing contextual embeddings for the text.
   - Output: \( E_{\text{DistilBERT}} \in \mathbb{R}^{n \times 768} \).
2. **Static Embeddings**:
   - Created using spaCy’s `en_core_web_sm` model for fixed 96-dimensional embeddings.

### Models:
- **Machine Learning**:
  - Logistic Regression
  - Random Forest
  - Classificatin and Regression Tree
  - Naïve Bayes
  - SVM
  - XGBoost
- **Deep Learning**:
  - Long Short-Term Memory (LSTM)

### Evaluation Metrics:
- Macro and Weighted Precision, Recall, F1-score.

---


## Technologies Used

- **Programming Language**: Python 3.11.9
- **Libraries**:
  - TensorFlow
  - Torch
  - spaCy
  - scikit-learn
  - transformers
  - pandas, NumPy, NLTK, TextBlob

---

## Installation

1. Install Python version <3.12 or check the requiremtns for torch and tensorflow version support
2. Clone this repository:
   ```bash
   git clone git@git.cs.dal.ca:courses/2024-fall/nlp-course/p-15.git
   ```
3. Set up Virtual Environment in cloned repository
4. Install the requiremtns.txt using the commanad
   ```   
   pip install -r requirements.text
   ```



## Authors

1. Rashik Mahmud Orchi(rs868141@dal.ca)
2. Ying Du (yn881054@dal.ca)



##  References

The following code snippets were used as reference to generate our project. 


1.  Andrey Shtrauss[Kaggle](https://www.kaggle.com/code/shtrausslearning/news-sentiment-based-trading-strategy)- to make the pipeline class for  training and evalution of Machine learning learning models. But in our code we changed the functions to incorporate macro and wighted evalution metrics and also to inlcude XGboot and Naive Bayes model. 

```
class nlp_evals:
    
    def __init__(self,df,corpus,label,
                 spacy_model='en_core_web_sm',
                 title='accuracy evaluation'
                ):
        
        self.df = deepcopy(df)
        self.corpus = corpus
        self.label = label
        self.spacy_model = spacy_model
        self.embeddings = self.get_embeddings()
        self.seed = 32
        self.num_folds = 4
        self.title = title

    def get_embeddings(self):
        
        # NLP pipline
        nlp = spacy.load(self.spacy_model)
        if(self.spacy_model is 'en_core_web_sm'):
            embedding_dims = 96
        elif(self.spacy_model is 'en_core_web_lg'):
            embedding_dims = 300
        
        # average embedding vector for each document
        all_vectors = np.array([np.array([token.vector for token in nlp(s) ]).mean(axis=0)*np.ones((embedding_dims)) \
                                   for s in self.df[self.corpus]])
        print(all_vectors.shape)
        print('embeddings loaded!')
        return all_vectors
        
    def tts(self,ratio=0.1):
        
        # split out validation dataset for the end
        Y = self.df[self.label]
        X = self.embeddings

        X_train, X_test, y_train, y_test = train_test_split(X, Y, 
                                                            test_size=ratio, 
                                                            random_state=32)
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train 
        self.y_test = y_test
        print('train/test split!')
        
    def define_models(self,models):
        self.models = models
        print('models set!')
        
    def kfold(self):
        
        self.results = []
        self.names = []
        self.test_results = []
        self.train_results = []
        self.cv_results = []
        
        lX_train = deepcopy(self.X_train)
        lX_test = deepcopy(self.X_test)
        ly_train = deepcopy(self.y_train.to_frame())
        ly_test = deepcopy(self.y_test.to_frame())
        
        print('model, cv mean, cv std, train, test')

        for name, model in self.models:
            
            # cross validation on training dataset
            kfold = KFold(n_splits=self.num_folds, shuffle=True,random_state=self.seed)
            cv_results = cross_val_score(model, 
                                         self.X_train, self.y_train,
                                         cv=kfold,
                                         scoring='accuracy')
            self.results.append(cv_results)
            self.names.append(name)
            self.cv_results.append(cv_results.mean())

           # Full Training period
            res = model.fit(self.X_train, self.y_train)
            ytrain_res = res.predict(self.X_train)
            acc_train = accuracy_score(ytrain_res,self.y_train)
            self.train_results.append(acc_train)

            # Test results
            ytest_res = res.predict(self.X_test)
            acc_test = accuracy_score(ytest_res, self.y_test)
            self.test_results.append(acc_test)    

            msg = "%s: %f (%f) %f %f" % (name, 
                                         cv_results.mean(), 
                                         cv_results.std(), 
                                         acc_train, 
                                         acc_test)
            
            ly_train[f'{name}_train'] = ytrain_res
            ly_test[f'{name}_test'] = ytest_res
            
            print(msg)
            print(confusion_matrix(ytest_res, self.y_test))
            
        
        self.ly_train = ly_train
        self.ly_test = ly_test
            
        print('evaluation finished!')
        
    def plot_results(self):

        ldf_res = pd.DataFrame({'cv':self.cv_results,
                                'train':self.train_results,
                                'test':self.test_results})
        
        plot_df = ldf_res.melt()
        local_names = deepcopy(self.names)
        local_names = local_names * 3
      
        plot_df['names'] = local_names
        
        ptable = pd.pivot_table(plot_df,
                                values='value',
                                index='variable',
                                columns='names')

        fig,ax = plt.subplots(figsize=(5,1.5))
        sns.heatmap(ptable,annot=True,
                    fmt=".2f",
                    ax=ax,
                    cmap='crest')
        plt.title(self.title)
```

2.  Rebeen Hamad[Medium](https://medium.com/@rebeen.jaff/what-is-lstm-introduction-to-long-short-term-memory-66bd3855b9ce) - to employ the LSTM Based deep learning strucuture based on 





