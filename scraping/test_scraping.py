# これがcoding:UTF-8
import urllib.request, urllib.error as ul2
# URLにアクセスするとHTMLが帰ってくる
with urllib.request.urlopen("https://tenki.jp/forecast/3/14/4310/11225/") as url:
    html = url.read()
from bs4 import BeautifulSoup
from datetime import datetime
import time

# htmlをBeautifulSoupで扱う
soup = BeautifulSoup(html, "html.parser")

# today要素の検出 最初の大枠の要素を摘出
today = soup.find(class_="today-weather")

# p要素全てを摘出する→全てのp要素が配列に入ってかえされます→[<p class="m-wficon triDown"></p>, <p class="l-h...
p = soup.find_all("p")
span = soup.find_all("span")

# print時のエラーとならないように最初に宣言しておきます。
today_weather = ""
# for分で全てのp要素の中からClass="p"となっている物を探します
for tag in p:
    # classの設定がされていない要素は、tag.get("class").pop(0)を行うことのできないでエラーとなるため、tryでエラーを回避する
    try:
        # tagの中からclass="n"のnの文字列を摘出します。複数classが設定されている場合があるので
        # get関数では配列で帰ってくる。そのため配列の関数pop(0)により、配列の一番最初を摘出する
        # <p class="hoge" class="foo">  →   ["hoge","foo"]  →   hoge
        string_ = tag.get("class").pop(0)

        # 摘出したclassの文字列に weather-telop と設定されているかを調べます
        if string_ in "weather-telop":
            # mkc-stock_pricesが設定されているのでtagで囲まれた文字列を.stringであぶり出します
            today_weather = tag.string
            # 摘出が完了したのでfor分を抜けます
            break
    except:
        # パス→何も処理を行わない
        pass

today_temp = ""
for tag in span:
    # classの設定がされていない要素は、tag.get("class").pop(0)を行うことのできないでエラーとなるため、tryでエラーを回避する
    try:
        # tagの中からclass="n"のnの文字列を摘出します。複数classが設定されている場合があるので
        # get関数では配列で帰ってくる。そのため配列の関数pop(0)により、配列の一番最初を摘出する
        # <p class="hoge" class="foo">  →   ["hoge","foo"]  →   hoge
        string_ = tag.get("class").pop(0)

        # 摘出したclassの文字列に weather-telop と設定されているかを調べます
        if string_ in "value":
            # mkc-stock_pricesが設定されているのでtagで囲まれた文字列を.stringであぶり出します
            today_temp = tag.string
            # 摘出が完了したのでfor分を抜けます
            break
    except:
        # パス→何も処理を行わない
        pass

# 摘出した日経平均株価を出力します。
print ("天気は" + today_weather)
print ("最高気温は" + today_temp + "度")