import numpy as np
import pandas as pd
import itertools
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt

from google.colab import files # Загрузка файла с диска
uploaded = files.upload()

df = pd.read_csv('fake_news.csv')

print(df.head())

'''Plotting the dataset    Построение набора данных
Since we cant plot text data we will plot bar chart of the count of the Fake vs Real
Поскольку мы не можем построить текстовые данные, мы посмотрим на
соотношения поддельных и реальных данных.'''
print(df.label.value_counts())

X_train, X_test, y_train, y_test = train_test_split(df['text'], df.label, test_size=0.1, random_state=7)

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and Transform training data and testing data
# Подгонка и преобразование данных обучения и тестирования
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

# Initialize a PassiveAggressiveClassifier
# Инициализируйте пассивно-агрессивный классификатор
model = PassiveAggressiveClassifier(max_iter=100)
model.fit(tfidf_train, y_train)

# Predict on the testing data   Прогнозировать на основе данных тестирования
y_pred = model.predict(tfidf_test)

# Finding Accuracy  Определение точности
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {round(score*100, 2)}%')

#  Creating Classification Report  Создание отчета о классификации
print('\n отчет о классификации:\n', classification_report(y_test, y_pred))

# Creating Confusion Matrix    Создание матрицы путаницы (ошибок)
confusion_matrix = metrics.confusion_matrix(y_test, y_pred, labels=['FAKE','REAL'])

# преобразовать таблицу в отображение матрицы путаницы (ошибок)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])
cm_display.plot()
plt.show()