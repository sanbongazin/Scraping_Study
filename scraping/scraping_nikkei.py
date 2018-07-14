# coding:UTF-8
import urllib.request, urllib.error as ul2
# URLにアクセスするとHTMLが帰ってくる
with urllib.request.urlopen("http://www.nikkei.com") as url:
    s = url.read()
from bs4 import BeautifulSoup

# htmlをBeautifulsoupで扱う
soup = BeautifulSoup(s, "html.parser")

# タイトル要素を取得。 ＝＞<title>経済、株価、ビジネス、政治のニュース:日経電子版</>・・・
title_tag = soup.title

# 要素の文字列取得
title = title_tag.string

# タイトル要素の取得
print (title_tag)

# タイトル文字列の取得
print (title)