from bs4 import BeautifulSoup
import requests
from tables import comp
from flask import Flask, flash, redirect, render_template, request, url_for
import pandas as pd
from tables import tensor_results1, data_recommendations1
app = Flask(__name__)


url = 'https://t.me/s/moex_news_parser'
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")
tags = soup.find_all('div', class_ = ['tgme_widget_message_wrap js-widget_message_wrap','tgme_header_right_column'])
news = []
@app.route('/', methods=['GET', 'POST'])
def data():
    str_news = []
    sorted = []
    for i in tags:
        if i not in news:
            news.append(i)
    for t in news:
        t = str(t)
        str_news.append(t)
    newlist2=pd.DataFrame(str_news,columns=["news"])
    sa3 = request.args.get('sa3')
    if sa3:
        for p in str_news:
            if sa3 in p:
                sorted.append(p)
    else:
        sorted = '<h2>Не найдено</h2>'

    return render_template('base.html',text=news, comp=comp, sorted = sorted, table1=[tensor_results1.to_html(classes='table1', header = True)], title1=tensor_results1.columns.values, table2=[data_recommendations1.to_html(classes='table2', header = True)], title2=data_recommendations1.columns.values)


@app.route('/')
def index():
   return render_template('base.html',**locals())

app.run(debug=True)
