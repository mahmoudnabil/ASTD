ASTD: Arabic Sentiment Tweets Dataset
===============================================

This dataset contains over 10k Arabic sentiment tweets classified into four classes subjective positive, subjective negative,
subjective mixed, and objective. Two sets of baseline sentiment analysis experiments are supported with the dataset.

Contents:
---------

- README.md: this file

- data/
                      
	  - Tweets.txt: a tab separated file containing the "cleaned up" tweets. It contains over 10k tweet. The format is:
		             
		             review<TAB>rating
		             

	  
	  - 4class-balanced-train/test/validation.txt: text file containing indices of tweets 
		             (from the Tweets.txt file) that are in the training/test/validation
		             sets. Balanced means the number of reviews in the 
		             positive/negative/mixed/objective classes are equal.
		             
	  - 4class-unbalanced-train/test.txt: the same, but the sizes of the calsses 
		             are not equal.
		             




- python/
  
   - AraTweet.py: the main interface to the dataset. Contains functions that can
              read/write training and test sets.
              
   - twitter_experiments.py: a Python script containing the code used to generate the baseline experiments
              
              
   - Defiantions.py: a python file contain the definations for the used classifiers

   - Utilities.py: a python file contain the some reading functions and classifier performance measure functions.


Demo
---------
In order to replicate the splits with different test/train/validation precent 



>AraSent=AraTweet()

>(body,rating) = AraSent.read_clean_reviews()

>AraSent.split_train_validation_test(self, rating, percent_test, percent_valid,
                                balanced="unbalanced")


In order to try new classifier just add it to "classifiers" list in Definations.py
then run twitter_experiments.py
   
Reference
---------
To be Added EMNLP2015
