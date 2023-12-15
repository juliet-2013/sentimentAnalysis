# Taming-the-TwitterVerse: Politicians-Embrace-Sentiment-Analysis-to-Shape-Public-Perception
Enhancing Political Brand Reputation Through Twitter Sentiment Analysis

---

>  **Project Authors:** Richard Taracha || Juliet Thuku || Eva Kiio

> **Flatiron School Data Science Program**

>  **Date:** 14/12/2023
> 
![Taming the TwitterVerse](https://github.com/TarachaR/Taming-the-TwitterVerse-Politicians-Embrace-Sentiment-Analysis-to-Shape-Public-Perception/assets/67068918/18cd6417-dd21-450c-9814-fb7b02b2943c)

<p align="right"><i><small>Image by: Richard Taracha</small></i></p>

---
</br>

> ###  Table of Contents
1. [Executive Summary](#background-information) 
2. [Problem Statement](#understanding-the-context)
3. [Proposed Solution](#understanding-the-context)
4. [Data Pertinence and Attribution](#understanding-the-context)
5. [Formulating the Benchmark of Success](#understanding-the-context)
6. [Project Structure & Deliverables](#project-deliverable)
7. [Recording the Experimental Design](#recording-the-experimental-design)
8. [Results](#project-deliverable)


</br>

---

</br>

> ## 1. Executive Summary

***In the dynamic world of politics, public perception plays a crucial role in shaping the success or failure of a political figure. Leveraging the power of social media, particularly Twitter, has become an indispensable tool for political leaders to connect with their constituents, gauge public sentiment, and build a strong brand identity. However, effectively managing and understanding the vast amount of data generated on Twitter can be a daunting task. This is where Twitter sentiment analysis using NLP (Natural Language Processing) emerges as a powerful solution.***

</br>

---
> ## 2. Problem Statement

**Our client, a prominent political figure, seeks to enhance their brand reputation and strengthen their connection with their followers on Twitter. They recognize the importance of understanding public sentiment and identifying emerging trends in online conversations. However, manually analyzing the sheer volume of tweets directed at their account is a time-consuming and resource-intensive endeavor.**

</br>

---
> ## 3. Proposed Solution

**We propose implementing a Twitter sentiment analysis NLP project to automatically extract and analyze sentiment from tweets directed from the client's Twitter account. This solution will provide valuable insights into the public's perception of the client's policies, actions, and overall brand image.**

</br>

---
> ## 4. Data Pertinence and Attribution

**Data Source:**

- Twitter account: Dr. Miguna Miguna (@MigunaMiguna)
- Date range: January 1, 2019 - April 28, 2022
- Number of tweets: 43,479
- Attribution: The data used in this project was scraped from Twitter using Twint, an unofficial Twitter scraper. The data is publicly available on Twitter and is not owned by the author of this project.
- Data Dictionary:

  | Field | Description |
    |-|-|
    | id | Unique identifier for the tweet |
    | conversation_id | Unique identifier for the conversation thread |
    | created_at | Date and time the tweet was created |
    | time | Time zone of the tweet |
    | user_id | Unique identifier for the user who posted the tweet |
    | username | Twitter handle of the user who posted the tweet |
    | name | Name of the user who posted the tweet |
    | place | Location of the user who posted the tweet (if available) |
    | tweet | Content of the tweet |
    | language | Language of the tweet |  
    | mentions | Twitter handles mentioned in the tweet |
    | urls | URLs mentioned in the tweet |
    | photos | Photos attached to the tweet |
    | replies_count | Number of replies to the tweet |
    | retweets_count | Number of retweets of the tweet |
    | likes_count | Number of likes on the tweet |
    | hashtags | Hashtags used in the tweet |
    | cashtags | Cashtags used in the tweet |
    | link | URL of the tweet |
    | retweet | Whether the tweet is a retweet |
    | quote_url | URL of the quoted tweet (if applicable) |
    | video | Whether the tweet includes a video |
    | thumbnail | Thumbnail image for the video (if applicable) |
    | near | Location near which the tweet was posted (if available) |
    | geo | Geolocation coordinates of the tweet (if available) |
    | source | Application used to post the tweet |  
    | user_rt_id | Unique identifier of the retweeted tweet (if applicable) |
    | user_rt | Whether the user retweeted the tweet |
    | retweet_id | Unique identifier of the retweeted tweet (if applicable) |  
    | reply_to | Unique identifier of the tweet being replied to (if applicable) |
    | retweet_date | Date and time the tweet was retweeted (if applicable) |
    | translate | Translated text of the tweet (if applicable) |
    | trans_src | Source language of the translation (if applicable) |  
    | trans_dest | Destination language of the translation (if applicable) |

</br>

---
> ## 5. Formulating the Benchmark of Success

The core benchmarks for success in this NLP Twitter analysis project will be: 

a) Building a classifier accuracy model with at least 80% precision in detecting tweet sentiment (positive, negative, neutral)

b) Providing actionable insights by identifying major influencers, topics, and events impacting changes in sentiment

c) Creating easy-to-understand data visualisations and reports to clearly communicate insights to our client. 

By developing a high-performance classification model and emphasizing practical, strategic business insights, our solution will enable our client to efficiently gauge public perception, track brand reputation, and guide future decisions - achieving their core goal of understanding and connecting with their followers at scale.

</br>

---
> ## 6. Project Structure & Deliverables
> 
The following are the required deliverables for this project:
- A **GitHub repository**
- A **Jupyter Notebook**
- A **non-technical presentation**
- A **Deployed Streamlit Web Application** [Click Me!](https://tamingthetwitterverse.streamlit.app/)
  
![Screenshot (106)](https://github.com/TarachaR/Taming-the-TwitterVerse-Politicians-Embrace-Sentiment-Analysis-to-Shape-Public-Perception/assets/67068918/a52db26a-1610-4ad4-be5f-848691476a52)


</br>

---
> ## 7. Recording the Experimental Design
Adapting the methodology for a Twitter sentiment analysis project involves focusing on the specific aspects related to text data and sentiment classification. Here's a revised methodology for your Twitter sentiment analysis project:

1. **Data Collection and Preprocessing:**
    Gather tweets using relevant usrenames or keywords.
    Preprocess the text data by removing mentions, hashtags, and URLs, and handle special characters.
    Explore the dataset to understand its structure, size, and language-specific nuances.
    Clean the data by addressing issues like spelling errors and removing irrelevant information.

2. **Exploratory Data Analysis (EDA):**
    Perform EDA to understand the distribution of sentiment classes in the dataset.
    Analyze word frequencies, common phrases, and trends in positive, negative, and neutral tweets.
    Identify any patterns or correlations between tweet features and sentiment labels.

3. **Data Visualization:**
    Utilize word clouds, bar charts, and pie charts to visually represent the distribution of sentiments.
    Create visualizations that showcase the most frequent words associated with each sentiment class.

4. **Text Representation:**
    Convert the text data into a suitable format for machine learning models, using TF-IDF.
    Explore different text representation techniques and choose the one that best captures the nuances of sentiment in tweets.

5. **Machine Learning Modeling:**
    Build and train various sentiment analysis models, such as Complement Naive Bayes, Random Forests, and Logistic Regression.
    Evaluate model performance using metrics like accuracy, precision, recall, and F1 score. Here out metric of choice was the Average Macro-Recall Score.
    Address the challenge of imbalanced classes by employing techniques like Random Oversampling.
    Consider the interpretability of the models, especially if explaining the results is important.

6. **Hyperparameter Tuning and Model Selection:**
    Fine-tune the hyperparameters of the chosen model to optimize performance.
    Select the model that demonstrates the best balance between precision and recall for sentiment classification.

7. **Deployment:**
    Deploy the chosen model as a Streamlit Web application to perform sentiment analysis on incoming tweets.

8. **Results Interpretation and Reporting:**
    Interpret the model predictions and analyze any misclassifications.
    Provide insights into the most influential features or words contributing to each sentiment class.
    Summarize the overall findings in a clear and concise manner suitable for sharing on Twitter or other platforms.




</br>

---
> ## 8. Results
1. Based on the Explore section of our Analysis:

   - **The user frequently tweets about Kenyatta and Uhuru, who are the president and the former president of Kenya, respectively. This suggests that he is very interested or involved in Kenyan politics and that he may have a critical or oppositional stance towards them.**

  - **He often uses the word despot, which means a tyrant or a dictator, to describe or refer to Kenyatta or Uhuru. This implies that he has a negative or hostile attitude towards them and that he may accuse them of abusing their power or violating human rights.**

  - **He also tweets a lot about Kenyans, which could indicate that he cares about the people of Kenya and their welfare, or that he wants to appeal to them as a potential leader or influencer.**

  - **Some of the other common words he uses are must, revolution, fight, and resist, which suggest that he has a strong or radical opinion on certain issues and that he calls for action or change from his followers or the public.**

2. Predictive Modelling Results:


| Model                         | Average Macro Recall | Negative Recall | Neutral Recall | Positive Recall | Overall Accuracy | F1-Score |
| :---------------------------- | :------------------: | --------------: | -------------- | --------------- | ----------------- | -------- |
| Baseline-Logistic Regression  |        0.80          |           0.92  | 0.74           | 0.74            | 0.81              | 0.81     |
| Random Forest                 |        0.81          |           0.75  | 0.95           | 0.73            | 0.82              | 0.82     |
| Tuned Random Forest           |        0.60          |           0.43  | 0.99           | 0.40            | 0.65              | 0.62     |
| Oversampled Random Forest     |        0.81          |           0.76  | 0.93           | 0.74            | 0.83              | 0.82     |
| Tuned Oversampled Random Forest|        0.60          |           0.43  | 0.99           | 0.40            | 0.65              | 0.62     |
| Logistic Regression           |        0.81          |           0.78  | 0.88           | 0.77            | 0.82              | 0.82     |
| Tuned LR                      |        0.90          |           0.88  | 0.95           | 0.87            | 0.91              | 0.91     |
| Oversampled LR                |        0.84          |           0.81  | 0.90           | 0.80            | 0.84              | 0.84     |
| Tuned Oversampled LR          |        0.90          |           0.89  | 0.95           | 0.87            | 0.91              | 0.91     |
| Complement NB                 |        0.73          |           0.78  | 0.64           | 0.77            | 0.72              | 0.72     |
| Tuned Complement NB           |        0.74          |           0.78  | 0.66           | 0.77            | 0.73              | 0.73     |
| Oversampled Complement NB     |        0.71          |           0.81  | 0.54           | 0.78            | 0.69              | 0.69     |
| Tuned Oversampled Complement NB|        0.71          |           0.81  | 0.54           | 0.78            | 0.69              | 0.69     |

**The summary table shows the performance metrics of different machine learning models on a sentiment analysis task. The models are evaluated based on the average macro recall, negative recall, neutral recall, positive recall, overall accuracy, and F1-score.**

**Based on the summary table, the best performing models are the Tuned `Logistic Regression` and `Tuned Oversampled Logistic Regression`, which have the highest values for all the metrics. These models are logistic regression models that have been tuned using some optimization techniques, such as grid search to find the best hyperparameters. They also use Random oversampling, which is a technique to balance the class distribution by creating synthetic samples of the minority class. These models can accurately classify the sentiments of the data, and they have a good balance between precision and recall for each class.**

(a)  TUNED LOGISTIC REGRESSION MODEL:

![Tuned_LR](https://github.com/TarachaR/Taming-the-TwitterVerse-Politicians-Embrace-Sentiment-Analysis-to-Shape-Public-Perception/assets/67068918/561f324b-97dd-44e1-9baf-bae51cfbcfcc)

(b)  OVER-SAMPLED TUNED LOGISTIC REGRESSION MODEL:

![Tuned_Oversample_LR](https://github.com/TarachaR/Taming-the-TwitterVerse-Politicians-Embrace-Sentiment-Analysis-to-Shape-Public-Perception/assets/67068918/91c3e1b2-b3ca-4e13-afa9-a67d886d9f77)


**The worst performing models are `Tuned RandomForest` and `Tuned Oversampled Random Forest`, which have the lowest values for most of the metrics. These models are random forest models that have been tuned using some optimization techniques, but they have a poor performance on the sentiment analysis task. They have a very low recall for the negative and positive classes, which means that they fail to identify most of the negative and positive sentiments in the data. They also have a low accuracy and F1-score, which means that they have a poor balance between precision and recall.**

(c) TUNED RANDOM FOREST MODEL:

![Tuned_RF](https://github.com/TarachaR/Taming-the-TwitterVerse-Politicians-Embrace-Sentiment-Analysis-to-Shape-Public-Perception/assets/67068918/5b1e9e9d-14f0-4522-b5ce-33c84a53e607)

(d) OVER-SAMPLED TUNED RANDOM FOREST MODEL:

![Tuned_Oversampled_RF](https://github.com/TarachaR/Taming-the-TwitterVerse-Politicians-Embrace-Sentiment-Analysis-to-Shape-Public-Perception/assets/67068918/634a6d51-aed0-46e8-9035-ea439664ed48)

3. Feature Importance - Tuned LR Model

![feature_importance](https://github.com/TarachaR/Taming-the-TwitterVerse-Politicians-Embrace-Sentiment-Analysis-to-Shape-Public-Perception/assets/67068918/fc85a5bf-619d-4bbd-964d-49718aceee3b)

**On the left of the graph, we can see how each word in the tweet affects the model's prediction. If the tweet had words such as "happy", "excellent" or "innocent" for example, the model's prediction was pushed towards the tweet being positive while words such as "pathetic", "desperate", or "stupid" pushed the prediction towards negative.**

**It is also worth noting that since this is a multiclass model, it is more difficult to interpret how each word is affecting the prediction, however, it still provides insight into important words that the user (Dr. Miguna Miguna) should keep an eye out for.**

</br>

---
> ## 8. Recommendations
As discussed in the Explore Section, in light of the insights we provided above, our recommendations for the user (Dr. Miguna Miguna) are as follows:

- **The user could try to engage more with their audience by asking them questions, conducting polls, or inviting feedback. This could help them understand their views and needs better, as well as to build trust and rapport with them.**

- **The user could also diversify their topics and sources of information by tweeting about other relevant issues, such as the economy, health, education, or culture. This could help you to broaden your perspective and appeal to a wider range of followers.**

- **The user could be more respectful and constructive in your criticism of Kenyatta and Uhuru, and avoid using inflammatory or abusive language. This could help you to avoid legal troubles, as well as to maintain your credibility and reputation.**

- **The user could also acknowledge the positive aspects or achievements of Kenyatta and Uhuru, and offer suggestions for improvement or alternatives. This could help the user to show that they are fair and balanced, and that they are willing to work with them for the common good of Kenya.**

- </br>

---
> ## 9. Limitations & Future Work

- **Some text data is going to be more negative or more positive than others. By creating a scale from very negative to somewhat negative to neutral to somewhat positive to very positive, more nuance will be able to be found in the sentiment analysis, and actions can be taken based on the severity of the situation.**

- **Sarcasm and negation can pose significant challenges to sentiment analysis models. In sarcastic text, people express their negative sentiments using positive words, which can easily confuse sentiment analysis models unless they are specifically designed to handle these cases.**

- **There is plenty of other publicly available text data that can be acquired and monitored for sentiment. This data may be on other social media platforms or public forums, or could be product reviews. While product reviews often have an associated rating, that rating may differ from the overall sentiment of the review. Classifying this other data will require a new model because its structure would differ from a tweet.**

- **The dynamic and ever-evolving nature of social media data can present difficulties for maintaining and updating sentiment analysis models and algorithms.**

- **As next steps, if we would like to generalize these models for different applications, we would definitely gather more data from Twitter and potentially other sources. Additionally, if the data had to be labeled by humans, we would set guidelines for what each class of tweet would consist of with examples to make sure that the labels didn't solely rely on emotions. Furthermore, taking the average of sentiment labels for each tweet would result in more accurate labels.**

- **Use cross-validation and regularization to prevent overfitting and underfitting, which are common problems in machine learning. Overfitting is when the model performs well on the training data but poorly on the test data, while underfitting is when the model performs poorly on both the training and test data.**

- **Use ensemble methods, such as bagging or boosting, to combine multiple models and improve the performance. Ensemble methods can reduce the variance and bias of the models, and increase the diversity and stability of the predictions.**

- **Lastly, the performance of the models could be greatly improved by rethinking this project with Neural Networks. In the future we would use Deep Learning to classify tweets.**

