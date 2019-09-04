![picture](title.png)
![Link to Leaderboard](https://datahack.analyticsvidhya.com/contest/india-ml-hiring-hackathon-2019/pvt_lb)

**Approach** :

I divided my solution to sections.

1. Data Interpretation:
    - Dataset is highly skewed (imbalanced) which is quite obvious since the category of problem is like that. The extent of skewness is, class-0 contains 99.45% of samples and remaining  0.005%(approx) to class-1
    - Feature list along with description includes:
        1. There are in total 27 features to play around with.
          1. First instincts suggests few features carry more importance than others (even if they can be discarded by feature engg) like Credit\_Score,loan\_amount, previous few months of delinquencies, issuing institution (for trend) etc.
          2. It&#39;s less likely that all the feature contribute to the target variable prediction so we&#39;ll find more of that in Feature engg section
    2. There is a mix of categorical as well as continuous features in the data which has to be handled smartly and efficiently (in core analysis)

2. Analysis:(Step by step is given in the ipython notebook)
    - After Data Interpretation I tried finding correlation between features(both continuous and categorical), turned out it existed between features, so now we could use it in Feature Engineering phase.
        - For finding correlation between continuous one's I used Spearman and for categorical I used Cramer&#39;s V and Theils\_U(non-symmetrical)
    - Did some analysis on few features that had correlation. For eg I dropped origination\_date and first\_payment\_date and instead made a new column that was difference between these two because the dates weren&#39;t significant enough than the time it took to make the first installment.

2. Feature Engineering:
    - I implemented Recursive Feature Elimination using the Bruteforce model of XGBoost and tried finding features that are relevant.
    - Also with optimized RandomForest model(explained below), RFE was again called in sort of a backtracked fashion.


1. Model Building:
    - Since the data is highly skewed I used Focal Loss technique to tune my algorithms to tackle it.
    - Focal loss weights were computed statistically for opt F1-score
class\_weight = {0:1.6,1:180}

    - Started with XGBoostClassifier , in xgboost we can set focal loss by altering scale\_pos\_weight parameter which you can compute by

    scale\_pos\_weight = 100 - ( (num\_pos\_samples /total\_samples) \* 100 )
The f1-score on training data was 0.9973 (this was the test submission so i didn&#39;t checked on eval or test set)

    - Moving further I switched to RandomForest since it works pretty well on these sort of problems.
I trained this using StratifiedKfold cross-validation geting a cross-validation F1-score in the range 0.74-0.79 on different configurations

    - I combined GridSearchCV to find optimal params for XG and RF and used them in an Ensemble way. Multiple model prediction and Stacking them in column way and leveraging every prediction was done.

    - Did some feature engineering using Recursive Feature Elimination using RF(which gave me the best f1-score on AV) to find out the most optimal subset of features.

  3. Finally my f1-score on public leaderboard went 0.336... and I finished 98th out of 4000(approx) participants.

 ##### Final Model:
Best result I got was with combining RF+GridsearchCV+StratifiedKfold training. The complete Classification report including Precision Recall scores and Confusion Matrix can be found in the ipython notebook itself.

So this roadmap clearly describes how I reached to my final model which was RandomForestClassifier tuned using GridSearchCV and training on StratifiedKfold cross-validation splits.

#### Vision:

I had some ideas to implement but unfortunately couldn&#39;t so I putting this in scope here:

1.  By looking at the predictions it was evident that the classifier were able to predict some cases with high probability and some with close to 0.5,0.5. So by looking at those cases(the data) more depth about them can be learned.

2.  Feature engineering wasn&#39;t implemented at full capacity so maybe testing more techniques, better features that contributes more to the prediction can be implemented.

3. Recently heard about IsolationForest Algorithm by H2O and people are recommending it for highly imbalanced classification problems.

Key Takeaways:

1. Highly skewed set of problems like this one or Fraud Detection problems has a lot of ways to solve just by altering the view of data
  1. I read about SMOTE, Focal Loss and Data Augmentation Techniques, though there are a lot of more statistic one&#39;s like Over and Under-Sampling, kinds of Synthetic Sampling but the above one&#39;s seemed more usable and practical.
  2. Hyperparameter Optimization in these set of problems is very important like for eg one could have introduced a new parameter by **EMI/ Cost of living** which is something called FOIR (Fixed Obligations to Income Ratio) .
2. Feature Engineering and knowing the data in deep is necessary to solve a problem, because if you don&#39;t know what you are solving then the solution wouldn&#39;t be that significant.
