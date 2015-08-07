# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 16:27:03 2015
"""

import codecs

import numpy as np
import pandas as pd
import re

class AraTweet:
    def __init__(self):
        self.REVIEWS_PATH = "../data/"
        self.DELETED_REVIEWS_FILE = "deleted_reviews.tsv"
        self.CLEAN_REVIEWS_FILE = "Tweets.txt"
    # Copied from the PyArabic package.
    def arabicrange(self):
        """return a list of arabic characteres .
        Return a list of characteres between \u060c to \u0652
        @return: list of arabic characteres.
        @rtype: unicode;
        """
        mylist = [];
        for i in range(0x0600, 0x00653):
            try :
                mylist.append(unichr(i));
            except ValueError:
                pass;
        return mylist;

    # cleans a single review
    def clean_raw_review(self, body):
         # patterns to remove first
        pat = [\
            (u'http[s]?://[a-zA-Z0-9_\-./~\?=%&]+', u''),  # remove links
            (u'www[a-zA-Z0-9_\-?=%&/.~]+', u''),
#            u'\n+': u' ',                     # remove newlines
            (u'<br />', u' '),  # remove html line breaks
            (u'</?[^>]+>', u' '),  # remove html markup
#            u'http': u'',
            (u'[a-zA-Z]+\.org', u''),
            (u'[a-zA-Z]+\.com', u''),
            (u'://', u''),
            (u'&[^;]+;', u' '),
            (u':D', u':)'),
#            (u'[0-9/]+', u''),
#            u'[a-zA-Z.]+': u'',
#            u'[^0-9' + u''.join(self.arabicrange()) + \
#                u"!.,;:$%&*%'#(){}~`\[\]/\\\\\"" + \
#                u'\s^><\-_\u201D\u00AB=\u2026]+': u'',          # remove latin characters
            (u'\s+', u' '),  # remove spaces
            (u'\.+', u'.'),  # multiple dots
            (u'[\u201C\u201D]', u'"'),  # “
            (u'[\u2665\u2764]', u''),  # heart symbol
            (u'[\u00BB\u00AB]', u'"'),
            (u'\u2013', u'-'),  # dash
        ]

        # patterns that disqualify a review
        remove_if_there = [\
            (u'[^0-9' + u''.join(self.arabicrange()) + \
                u"!.,;:$%&*%'#(){}~`\[\]/\\\\\"" + \
                u'\s\^><\-_\u201D\u00AB=\u2026+|' + \
                u'\u0660-\u066D\u201C\u201D' + \
                u'\ufefb\ufef7\ufef5\ufef9]+', u''),  # non arabic characters
        ]

        # patterns that disqualify if empty after removing
        remove_if_empty_after = [\
            (u'[0-9a-zA-Z\-_]', u' '),  # alpha-numeric
            (u'[0-9' + u".,!;:$%&*%'#(){}~`\[\]/\\\\\"" + \
                u'\s\^><`\-=_+]+', u''),  # remove just punctuation
            (u'\s+', u' '),  # remove spaces
        ]

        # remove again
        # patterns to remove
        pat2 = [\
#            u'[^0-9' + u''.join(self.arabicrange()) + \
#                u"!.,;:$%&*%'#(){}~`\[\]/\\\\\"" + \
#                u'\s^><\-_\u201D\u00AB=\u2026]+': u'',          # remove latin characters
        ]

        skip = False

        # if empty body, skip
        if body == u'': skip = True

        # do some subsitutions
        for k, v in pat:
            body = re.sub(k, v, body)

        # remove if exist
        for k, v in remove_if_there:
            if re.search(k, body):
                skip = True

        # remove if empty after replacing
        for k, v in remove_if_empty_after:
            temp = re.sub(k, v, body)
            if temp == u" " or temp == u"":
                skip = True

        # do some more subsitutions
        if not skip:
            for k, v in pat2:
                body = re.sub(k, v, body)

        # if empty string, skip
        if body == u'' or body == u' ':
            skip = True

        if not skip:
            return body
        else:
            return u""

    # Read raw reviews from file and clean and write into clean_reviews
    def clean_raw_reviews(self):
        # input file
        in_file = codecs.open(self.REVIEWS_PATH + self.RAW_REVIEWS_FILE,
                              'r', encoding="utf-8")
        reviews = in_file.readlines()

        # Output file: rating<tab>content
        out_file = open(self.REVIEWS_PATH + self.CLEAN_REVIEWS_FILE,
                        'w', buffering=100)
        deleted_file = open(self.REVIEWS_PATH + self.DELETED_REVIEWS_FILE,
                            'w', buffering=100)

        counter = 1
        for i in xrange(0, len(reviews)):
            review = reviews[i]
            skip = False

#           # If line starts with #, then skip
#            if review[0] == u"#": continue

            # split by <tab>
            parts = review.split(u"\t")

            # rating is first part and body is last part
            rating = parts[0]
            review_id = parts[1]
            user_id = parts[2]
            book_id = parts[3]
            body = parts[4].strip()

            # clean body
            body = self.clean_raw_review(body)
            if body == u"": skip = True

            if i % 5000 == 0:
                print "review %d:" % (i)

            # write output
            line = u"%s\t%s\t%s\t%s\t%s\n" % (rating, review_id, user_id,
                                              book_id, body)
            if not skip:
                out_file.write(line.encode('utf-8'))
                counter += 1
            else:
                deleted_file.write(line.encode('utf-8'))




    # Read the reviews file. Returns a tuple containing these lists:
    #   rating: the rating 1 -> 5
    #   review_id: the id of the review
    #   user_id: the id of the user
    #   book_id: the id of the book
    #   body: the text of the review
    def read_review_file(self, file_name):

        reviews = codecs.open(file_name, 'r', 'utf-8').readlines()

        # remove comment lines and newlines
        reviews = [re.sub(("^\s+"), "", r) for r in reviews]
        reviews = [re.sub(("\s+$"), "", r) for r in reviews]
        # parse
        rating = list()
        body = list()

        for review in reviews:
            # split by <tab>
            parts = review.split(u"\t")

            # body is first part and rating is last part
            body.append(parts[0])
            rating.append(parts[1])

        return (body, rating)

    def read_clean_reviews(self):
        return self.read_review_file(self.REVIEWS_PATH +
                                     self.CLEAN_REVIEWS_FILE)

    def read_raw_reviews(self):
        return self.read_review_file(self.REVIEWS_PATH + self.RAW_REVIEWS_FILE)


    # Splits the data-set into a training/validation/test sets in the setting of using 4
    # classes and balanced or unbalanced settings
    def split_train_validation_test(self, rating, percent_test, percent_valid,
                                balanced="unbalanced"):
        np.random.seed(1234)

        rating = np.array(rating)
        # length
        num_reviews = len(rating)
        review_ids = np.arange(0, num_reviews)



        review_ids_pos=review_ids[rating=='POS']
        review_ids_neg=review_ids[rating=='NEG']
        review_ids_neutral=review_ids[rating=='NEUTRAL']
        review_ids_obj=review_ids[rating=='OBJ']

        np.random.shuffle(review_ids_pos)
        np.random.shuffle(review_ids_neg)
        np.random.shuffle(review_ids_neutral)
        np.random.shuffle(review_ids_obj)

        if balanced == "unbalanced":
            ntest_pos = np.floor(len(review_ids_pos)*percent_test)
            ntest_neg = np.floor(len(review_ids_neg)*percent_test)
            ntest_neutral = np.floor(len(review_ids_neutral)*percent_test)
            ntest_obj = np.floor(len(review_ids_obj)*percent_test)

            nvalid_pos = np.floor(len(review_ids_pos) * percent_valid)
            nvalid_neg = np.floor(len(review_ids_neg) * percent_valid)
            nvalid_neutral = np.floor(len(review_ids_neutral) * percent_valid)
            nvalid_obj = np.floor(len(review_ids_obj) * percent_valid)



            test_ids = np.concatenate([review_ids_pos[0:ntest_pos] \
                       ,review_ids_neg[0:ntest_neg]\
                       ,review_ids_obj[0:ntest_obj]\
                       ,review_ids_neutral[0:ntest_neutral]])

            validation_ids = np.concatenate([review_ids_pos[ntest_pos:ntest_pos+nvalid_pos] \
                       ,review_ids_neg[ntest_neg:ntest_neg+nvalid_neg]\
                       ,review_ids_obj[ntest_obj:ntest_obj+nvalid_obj]\
                       ,review_ids_neutral[ntest_neutral:ntest_neutral+nvalid_neutral]])

            train_ids = np.concatenate([review_ids_pos[ntest_pos+nvalid_pos:] \
                       ,review_ids_neg[ntest_neg+nvalid_neg:]\
                       ,review_ids_obj[ntest_obj+nvalid_obj:]\
                       ,review_ids_neutral[ntest_neutral+nvalid_neutral:]])

        elif balanced == "balanced":
            sizes=l=[len(review_ids_pos),len(review_ids_neg),len(review_ids_neutral),len(review_ids_obj)]
            min_size = min(sizes)

            ntest = np.floor(min_size * percent_test)
            nvalid = np.floor(min_size * percent_valid)

            test_ids = np.concatenate([review_ids_pos[0:ntest] \
                       ,review_ids_neg[0:ntest]\
                       ,review_ids_obj[0:ntest]\
                       ,review_ids_neutral[0:ntest]])

            validation_ids = np.concatenate([review_ids_pos[ntest:ntest+nvalid] \
                       ,review_ids_neg[ntest:ntest+nvalid]\
                       ,review_ids_obj[ntest:ntest+nvalid]\
                       ,review_ids_neutral[ntest:ntest+nvalid]])

            train_ids = np.concatenate([review_ids_pos[ntest+nvalid:min_size] \
                       ,review_ids_neg[ntest+nvalid:min_size]\
                       ,review_ids_obj[ntest+nvalid:min_size]\
                       ,review_ids_neutral[ntest+nvalid:min_size]])



        train_file = self.REVIEWS_PATH + "4class-" + balanced + "-train.txt"
        test_file = self.REVIEWS_PATH + "4class-" + balanced + "-test.txt"
        validation_file = self.REVIEWS_PATH + "4class-" + balanced + "-validation.txt"

        open(train_file, 'w').write('\n'.join(map(str, train_ids)))
        open(test_file, 'w').write('\n'.join(map(str, test_ids)))
        open(validation_file, 'w').write('\n'.join(map(str, validation_ids)))

        return (train_ids, test_ids)

    # Reads a training or test file. The file contains the indices of the
    # reviews from the clean reviews file.
    def read_file(self, file_name):
        ins = open(file_name).readlines()
        ins = [int(i.strip()) for i in ins]

        return ins

    # A helpter function.
    def set_binary_klass(self, ar):
        ar[(ar == 1) + (ar == 2)] = 0
        ar[(ar == 4) + (ar == 5)] = 1

    # A helpter function.
    def set_ternary_klass(self, ar):
        ar[(ar == 1) + (ar == 2)] = 0
        ar[(ar == 4) + (ar == 5)] = 1
        ar[(ar == 3)] = 2

    # Returns (train_x, train_y, test_x, test_y)
    # where x is the review body and y is the rating (1->5 or 0->1)
    def get_train_test(self, klass="2", balanced="balanced"):
        (rating, a, b, c, body) = self.read_clean_reviews()
        rating = np.array(rating)
        body = pd.Series(body)

        train_file = (self.REVIEWS_PATH + klass + "class-" +
            balanced + "-train.txt")
        test_file = (self.REVIEWS_PATH + klass + "class-" +
            balanced + "-test.txt")

        train_ids = self.read_file(train_file)
        test_ids = self.read_file(test_file)

        train_y = rating[train_ids]
        test_y = rating[test_ids]
        train_x = body[train_ids]
        test_x = body[test_ids]

        if klass == "2":
            self.set_binary_klass(train_y)
            self.set_binary_klass(test_y)
        return (train_x, train_y, test_x, test_y)

    # Returns (train_x, train_y, test_x, test_y)
    # where x is the review body and y is the rating (1->5 or 0->1)
    def get_train_test_validation(self, klass="4", balanced="balanced"):

        (body,rating) = self.read_clean_reviews()
        rating = np.array(rating)
        body = pd.Series(body)

        train_file = (self.REVIEWS_PATH + klass + "class-" +
            balanced + "-train.txt")
        test_file = (self.REVIEWS_PATH + klass + "class-" +
            balanced + "-test.txt")
        validation_file = (self.REVIEWS_PATH + klass + "class-" +
            balanced + "-validation.txt")

        train_ids = self.read_file(train_file)
        test_ids = self.read_file(test_file)
        validation_ids = self.read_file(validation_file)

        train_y = rating[train_ids]
        test_y = rating[test_ids]
        valid_y = rating[validation_ids]

        train_x = body[train_ids]
        test_x = body[test_ids]
        valid_x = body[validation_ids]

        return (train_x, train_y, test_x, test_y, valid_x, valid_y)



AraSent=AraTweet()
(body,rating)=AraSent.read_clean_reviews()
AraSent.split_train_validation_test(rating,0.2, 0.2,"unbalanced")


