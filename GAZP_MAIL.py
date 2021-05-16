from flask import Flask, render_template
from bs4 import BeautifulSoup
import requests
import pandas as pd
import re

def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext

with open("messages4.html", encoding='utf-8') as fp:
    soup = BeautifulSoup(fp, "html.parser")
tags = soup.find_all('span')
messages_text = []
cleaned_messages = []
mes = []
mes2 = []
list = ['Газпром', 'Mail', 'Мейл','GAZP', 'MAIL', 'MAILDR', 'ГАЗПРОМ','Gazprom', 'GAZPROM']
list2 = ['Сбербанк', 'СБЕРБАНК', 'SBER']
for i in tags:
  i = str(i)
  messages_text.append(i)
for message in messages_text:
        cleaned_messages.append(cleanhtml(message))
for i in cleaned_messages:
    i = re.sub("^\s+|\n|\r|\s+$", '', i)
    if len(i)>=10 & len(i)<=120:
            mes.append(i)
for k in mes:
    for b in list2:
        if b in k:
            mes2.append(k)
print(mes2)
