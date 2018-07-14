# coding:UTF-8
import urllib.request, urllib.error as ul2
# URLにアクセスするとHTMLが帰ってくる
with urllib.request.urlopen("http://www.nikkei.com/markets/kabu") as url:
    html = url.read()
from bs4 import BeautifulSoup
from datetime import datetime
import csv
import time

time_flag = True

#無限実行
while True:
    if datetime.now().minute != 59:
        # 59分ではないので、１分待機。
        time.sleep(58)
        continue
    # csvを追記モードで開く。ここでcsvファイルを開くのは、ファイルが大きくなった時に、csvを開くのに時間がかかるから
    f = open('nikkei_heikin.csv', 'a')
    writer = csv.writer(f, lineterminator = '¥n')

    # 59分になったが、正確な時間に測定するために５９秒になるまでは、抜け出せないようにする。
    while datetime.now().second != 59:
        time.sleep(1)
    time.sleep(1)

    csv_list = []

    # 現在時刻の取得
    time_ = datetime.now().strftime("%Y/%m/%d %H:%M:%S")

    # １カラム目に時間を挿入
    csv_list.append(time_)

    
    soup = BeautifulSoup(html, "html.parser")
    # print (soup.title.string)

    # span要素全てを抽出する。＝＞全てのspan要素が配列に入って返される。
    span = soup.find_all("span")

    # print時のエラーを防ぐ
    nikkei_heikin = ""

    # for文ですべてのspan要素の中から、class = "mkc - stock_prices"と生ってるものを探す
    for tag in span:
        # classの設定がなされていない要素は、tag.get("class").pop(0)を行う事のできないのでエラーとなる。そこで、tryでerrorを回避
        try:
            # tagの中から、class＝nの文字を摘出する。複数classeが設定されている場合があるので、
            # get関数では、配列で返ってくる、そのための配列の関数pop（0）より、配列の一番最初を摘出する
            # <span class ="hoge" class="foo"> => ["hoge", "foo"] => hoge
            string_ = tag.get("class").pop(0)

            # 摘出したclasseの文字列にmkc-stock_pricesが含まれているかを調べる
            if string_ in "mkc-stock_prices":
                # 設定されているので、tagで囲まれた文字列を、.stringであぶり出す
                nikkei_heikin = tag.string
                break
        except:
            # パス　＝＞何も処理を行わない
            pass
    print (nikkei_heikin)

# このプログラムで大体の応用が聞いてしまうのはすごい。