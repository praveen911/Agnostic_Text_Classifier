from time import time
import regex as re
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split
import spacy
from nltk.corpus import stopwords
from sklearn.naive_bayes import GaussianNB
import pickle

stopwords = stopwords.words('english')
nlp = spacy.load('en_core_web_sm')
filename = " "
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model.perceptron import Perceptron
from sklearn.linear_model.ridge import RidgeClassifierCV
from sklearn.linear_model.stochastic_gradient import SGDClassifier

app = Flask(__name__)


@app.route('/')
def firstpage():
    return render_template('dataset.html')


@app.route('/train', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        path = request.files.get('myFile')

        df = pd.read_csv(path, encoding="ISO-8859-1")

        filename = request.form['filename']

        str1 = request.form['feature']
        str2 = request.form['label']

        if str1 in list(df) and str2 in list(df):
            y = df[str2]
            X = df[str1]
        else:
            return render_template('nameError.html')

        x = []
        for subject in X:
            result = re.sub(r"http\S+", "", subject)
            replaced = re.sub(r'[^a-zA-Z0-9 ]+', '', result)
            x.append(replaced)
        X = pd.Series(x)

        X = X.str.lower()
        """
        texts = []
        for doc in X:
            doc = nlp(doc, disable=['parser', 'ner'])
            tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
            tokens = [tok for tok in tokens if tok not in stopwords]
            tokens = ' '.join(tokens)
            texts.append(tokens)

        X = pd.Series(texts)
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True)

        tfidfvect = TfidfVectorizer(ngram_range=(1, 1))
        X_train_tfidf = tfidfvect.fit_transform(X_train)

        start = time()
        clf1 = LinearSVC()
        clf1.fit(X_train_tfidf, y_train)
        pred_SVC = clf1.predict(tfidfvect.transform(X_test))

        a1 = accuracy_score(y_test, pred_SVC)
        end = time()
        print("accuracy SVC: {} and time: {} s".format(a1, (end - start)))

        start = time()
        clf2 = LogisticRegression(n_jobs=-1, multi_class='multinomial', solver='newton-cg')
        clf2.fit(X_train_tfidf, y_train)
        pred_LR = clf2.predict(tfidfvect.transform(X_test))
        a2 = accuracy_score(y_test, pred_LR)
        end = time()
        print("accuracy LR: {} and time: {}".format(a2, (end - start)))

        start = time()
        clf3 = RandomForestClassifier(n_jobs=-1)

        clf3.fit(X_train_tfidf, y_train)
        pred = clf3.predict(tfidfvect.transform(X_test))
        a3 = accuracy_score(y_test, pred)
        end = time()
        print("accuracy RFC: {} and time: {}".format(a3, (end - start)))

        start = time()
        clf4 = MultinomialNB()

        clf4.fit(X_train_tfidf, y_train)
        pred = clf4.predict(tfidfvect.transform(X_test))
        a4 = accuracy_score(y_test, pred)
        end = time()
        print("accuracy MNB: {} and time: {}".format(a4, (end - start)))

        start = time()
        clf5 = GaussianNB()

        clf5.fit(X_train_tfidf.toarray(), y_train)
        pred = clf5.predict(tfidfvect.transform(X_test).toarray())
        a5 = accuracy_score(y_test, pred)
        end = time()
        print("accuracy GNB: {} and time: {}".format(a5, (end - start)))

        start = time()
        clf6 = LogisticRegressionCV(n_jobs=-1)
        clf6.fit(X_train_tfidf, y_train)
        pred_LR = clf6.predict(tfidfvect.transform(X_test))
        a6 = accuracy_score(y_test, pred_LR)
        end = time()
        print("accuracy LRCV: {} and time: {}".format(a6, (end - start)))

        start = time()
        clf7 = AdaBoostClassifier()
        clf7.fit(X_train_tfidf, y_train)
        pred_LR = clf7.predict(tfidfvect.transform(X_test))
        a7 = accuracy_score(y_test, pred_LR)
        end = time()
        print("accuracy ABC: {} and time: {}".format(a7, (end - start)))

        start = time()
        clf8 = BernoulliNB()

        clf8.fit(X_train_tfidf.toarray(), y_train)
        pred = clf8.predict(tfidfvect.transform(X_test).toarray())
        a8 = accuracy_score(y_test, pred)
        end = time()
        print("accuracy BNB: {} and time: {}".format(a8, (end - start)))

        start = time()
        clf9 = Perceptron(n_jobs=-1)

        clf9.fit(X_train_tfidf.toarray(), y_train)
        pred = clf9.predict(tfidfvect.transform(X_test).toarray())
        a9 = accuracy_score(y_test, pred)
        end = time()
        print("accuracy Per: {} and time: {}".format(a9, (end - start)))
        start = time()
        clf10 = RidgeClassifierCV()

        clf10.fit(X_train_tfidf.toarray(), y_train)
        pred = clf10.predict(tfidfvect.transform(X_test).toarray())
        a10 = accuracy_score(y_test, pred)
        end = time()
        print("accuracy RidCV: {} and time: {}".format(a10, (end - start)))

        start = time()
        clf11 = SGDClassifier(n_jobs=-1)

        clf11.fit(X_train_tfidf.toarray(), y_train)
        pred = clf11.predict(tfidfvect.transform(X_test).toarray())
        a11 = accuracy_score(y_test, pred)
        end = time()
        print("accuracy SGDC: {} and time: {}".format(a11, (end - start)))
        start = time()
        clf12 = SGDClassifier(n_jobs=-1)

        clf12.fit(X_train_tfidf.toarray(), y_train)
        pred = clf12.predict(tfidfvect.transform(X_test).toarray())
        a12 = accuracy_score(y_test, pred)
        end = time()
        print("accuracy XGBC: {} and time: {}".format(a12, (end - start)))

        acu_list = [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12]
        max_list = max(acu_list)

        if max_list == a1:
            pickle.dump(clf1, open(filename + '_model', 'wb'))
        elif max_list == a2:
            pickle.dump(clf2, open(filename + '_model', 'wb'))
        elif max_list == a3:
            pickle.dump(clf3, open(filename + '_model', 'wb'))
        elif max_list == a4:
            pickle.dump(clf4, open(filename + '_model', 'wb'))
        elif max_list == a5:
            pickle.dump(clf5, open(filename + '_model', 'wb'))
        elif max_list == a6:
            pickle.dump(clf6, open(filename + '_model', 'wb'))
        elif max_list == a7:
            pickle.dump(clf7, open(filename + '_model', 'wb'))
        elif max_list == a8:
            pickle.dump(clf8, open(filename + '_model', 'wb'))
        elif max_list == a9:
            pickle.dump(clf9, open(filename + '_model', 'wb'))
        elif max_list == a10:
            pickle.dump(clf10, open(filename + '_model', 'wb'))
        elif max_list == a11:
            pickle.dump(clf11, open(filename + '_model', 'wb'))
        elif max_list == a12:
            pickle.dump(clf12, open(filename + '_model', 'wb'))

        pickle.dump(tfidfvect, open(filename + '_tfidfVect', 'wb'))

        return render_template("result.html", ac1=a1, ac2=a2, ac3=a3, ac4=a4, ac5=a5, ac6=a6, ac7=a7, ac8=a8, ac9=a9,
                               ac10=a10, ac11=a11, ac12=a12)


@app.route('/connect', methods=['POST', 'GET'])
def connect():
    return render_template('predict.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        path_predict = request.files.get('myFile_predict')

        df_1 = pd.read_csv(path_predict, encoding="ISO-8859-1")

        str3 = request.form['feature_predict']
        filename_2 = request.form['filename_2']

        if str3 in list(df_1):
            Z = df_1[str3]
        else:
            return render_template('nameError.html')

        """
        blob = []
        for doc in Z:
            doc = nlp(doc, disable=['parser', 'ner'])
            tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
            tokens = [tok for tok in tokens if tok not in stopwords]
            tokens = ' '.join(tokens)
            blob.append(tokens)

        Z = pd.Series(blob)
        """
        loaded_model = pickle.load(open(filename_2 + '_model', 'rb'))
        loaded_tfidf = pickle.load(open(filename_2 + '_tfidfVect', 'rb'))

        Z_predict = loaded_tfidf.transform(Z)

        predict_model = loaded_model.predict(Z_predict)

        df_1['label'] = predict_model
        name = request.form['csv_name']
        df_1.to_csv(name, sep=',')

        texts = df_1['text']
        labels = df_1['label']
        data_print = [[text, lab] for text, lab in zip(texts, labels)]

        return render_template("result1.html", data_print=data_print)


if __name__ == '__main__':
    app.run(debug=True)
