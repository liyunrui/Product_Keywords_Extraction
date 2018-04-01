Product Keywords Extraction
===
My solution for the Shopee Technical assesment. In order to help seller find the most effective keywords, I designed a keyword recommendation system mainly using NLP and ML techniques.

Task Overview
===
The task is how to extract best keywords: Given a seller’s product name and category of the product, to recommend a list best keywords for this product.

The task is quite different from common keywords extraction task such as search engine. For example, given a bunch of keywords queried by users, to find relevant document with those words.

The goal of the task also makes it interesting and challenging: What is definition of the best keywords? Let's saying if we knew, then How do we automatically judge whether the extracted keywords are the best keywords?

Data 
===
There are two datasets: sample_data.csv and test_data.csv, which are both provided by Shopee.

> a. sample_data.csv contains 10303 rows of user's prior view log.
> b. test_data.csv contains 200 rows of product name and category.

Below is the full data schema 
> **sample_data.csv** (10k rows)
> - Product Name: description of product from seller
> - Category: category of the product
> - Query: the words that users used to look for product
> - Event: click or impression
> - Date: date when user look for this product
> 
> **test_data.csv** (200 rows)
> - Product Name
> - Category


The Approach
===

The task was reformulated as a regression prediction task: Given a seller's product name, category of the product, and the user's prior view log data, to predict what ratio that users used this extracted word to look for products is [0~1]. 

**Keywords Prediction** - which words extracted by Jieba will most likely to be the words that users used to look for products? This model depends on both the product name and words.

![](https://i.imgur.com/9rztjzw.png)


Here is a diagram of my keywords recommendation system flow.

![](https://i.imgur.com/tNKZVTk.png)

The keywords probabilities are a weighted average of the outputs from the xgboost models. Then, the final list of best keywords are chosen by using these probabilities and choosing the keywords subset with maximum expected F1-score.

In the below, we’re going to break our approach into  a couple of chunks to elaborate.
### Text Preprocessing
In order to reduce the words without too much information, there are basically two steps to filter raw data: 
- Normalizing: Make different word with same meaning become a single word. For example, I turn "iphone 7" into "iphone7". 
- Remove stopwords: Since product name is much shorter than document, I don't use common stopwords. Here, I self-defined the stopwords are the words never were searched by users.

### Keywords Extraction ([Jieba](https://github.com/isuhao/jieba))
Since there are lot of new words in our task, this makes segmentation or tokenization harder. But Jieba has the ability to add your own new words. For example, I self-defined "手機殼","維多利亞的秘密", "細肩帶", and etc. It can ensure a higher rate of correct segmentation. That's why I chose Jieba to extract keywords.

### Predictive Modeling ([XGboost](https://github.com/dmlc/xgboost))
Although Jieba did well on keywords extraction, it was done based-on TF-IDF weights, which may not be the words that our customers used to look for products. So I created two different gradient boosted tree models to predict ratio that users used the word to look for products, then averaged their predictions together for reducing variance.

### F1 Maximization ([Reference](https://www.kaggle.com/mmueller/f1-score-expectation-maximization-in-o-n))
To convert these probabilities into binary Yes/No scores of which words will be used to look for products by users, I feed them into a special F1 Score Maximization algorithm. Later on, I will talk about the details in my presentation.



Feature Engineering
===
I created four types of features.

1. **Category features** - What is the product’s category?
2. **Product Name x word features** - What’s the relationship between a specific product name and words extracted?
3. **Hot Search features** - How well the extracted words is searched by users?
4. **Semantic features** - What is the semantics of the extracted words?


The details about how they were generated, please see directory [py_feature](https://github.com/liyunrui/Product_Keywords_Extraction/tree/master/py_feature).

## Which features were the most useful?

For the keywords prediction model, out of 261 features I created, we can see that the **top 10 important** features were...
![](https://i.imgur.com/re3ph07.png)

Explain the feature with top importance:

**Product Name x word features**
- weights_given_product_name: the tf-idf weights of words extracted by Jieba given a product name
- position_given_product_name: the position of extracted words givne a product name. For example, a product name, "蘋果手機大特賣", then the position of extracted words of "蘋果" is 1, the position of extracted words of "手機" is 2, and so on.
- len_extracted_words: the length of extracted words given a product name. Take the above example to interpret, the length of extracted words given "蘋果手機大特賣" is 3 since it has separately 3 extracted words: "頻果", "手機", and "大特賣".

**Hot Search features**
- log_count_in_hot_search: how many times the extracted words were used to look for product by users. Also, I performed log-transformation on this feature, because it's good to ML algorithm when features with large value, practically.

**Semantic features**
I built 250-dimensional semantic space using  [gensim](https://radimrehurek.com/gensim/models/word2vec.html), trained on [wiki corpus](https://dumps.wikimedia.org/zhwiki/20180101/).

- dim1: 1-dimensional semantic vector of the extracted word.
- dim2: 2-dimensional semantic vector of the extracted word.

Note: 
Here I need some preprocessing to wiki corpus for conversion between Traditional Chinese and Simplified Chinese, please refer to [OpenCC](https://github.com/BYVoid/OpenCC). However, to save time, I put pre-traiend model, called word2vec.model, in the directory [word2vec](https://github.com/liyunrui/Product_Keywords_Extraction/tree/master/word2vec).


Experiments
===
| What I did | Validating RMSE |
| ------ | ----------- |
| num of features less 10|~0.200 |
| semantic features + hot search features|~0.095|
| Remove Stopwords |0.065|



Summary
===
1. I designed an algorithm integrating Jieba, Gensim, and Xgboost to implement a keyword recommendation system.

2. Rather than returning keywords probabilities, I applied f1 score maximization to convert them into binary numbers (Yes: It's keyword/No, it's not keyword).

3. It still has some room for further improvement. Like we can see in experiments, the more effor we put into text preprocessing, the better result we got. Also, if time permits, I could try keywords expansion and find a better way to convert probabilities into list of best keywords.



Submission
===
The lists of recommended keywords for the test set was save as **recommended_keywords.xlsx** and put into directory [output/sub/final](https://github.com/liyunrui/Product_Keywords_Extraction/tree/master/output/sub/final).


How to run
===
The following command should be step-by-step executed in the terminal.
- cd py_feature
- python3 run.py
- cd ../py_model
- python3 run.py

Note that:
It may take approximately few hours to train model.

Requirements
===
MacOS Sierra or Linux-like system(recommended), Python 3.6.4

Python packages:
- numpy==1.13.3
- pandas==0.21.0
- xgboost==0.6
- scikit-learn==0.19.1
- scipy==1.0.0
- jieba==0.39
- gensim==3.4.0




