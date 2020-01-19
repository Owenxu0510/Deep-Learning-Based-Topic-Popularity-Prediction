# Deep-Learning-Based-Topic-Popularity-Prediction
This paper proposed a network which can predict whether a Sina Weibo topic will become popular.

This is not the whole project, but we still can achieve good result. The precision here is 83.27%. We use several characteristics of a topic to predict its future popularity, in this case, we use SVM. 

Then we consider the text message as an important characteristic, so we build a Fasttext classifier before we apply SVM, in order to focus on the text information.

Detailed information is in the paper.

newcontent1.xls contains all the topics and features we used. Also fasttext and svm are included here.
