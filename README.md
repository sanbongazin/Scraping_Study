---
title: 初学者用チュートリアルの選択肢を探ってみた
tags: チュートリアル 初心者
author: ushironoko
slide: false
---

何か新しい言語とかフレームワークを学ぶ時ってやっと構文や組み込み関数覚えたところで目的先行じゃない場合広い海原にほっぽりだされるわけですよ。次何すればいいんだろう？:rolling_eyes:と。

なのでそういう時何すればいいのか考えてみたい。

つまり今回の要件は ***初学者がやると良いチュートリアルを探る*** です。特に言語間で実装を比較できるものであればより良い。

というわけで世間で良く使われている教材を見ていきます。リンク先は古いものもありますので、参考程度に。

## FizzBuzz
[C#でFizzBuzz](https://qiita.com/Nucleareal/items/d2ebf32f7556d4ff7398)
[DartでFizzBuzz](https://qiita.com/loasnir/items/0c57cf9ec52474453079)
[Erlangでfizzbuzz](https://qiita.com/shiro01/items/c19ff58ba1598c3adefd)
[[Rust] RustでFizzBuzz書いてみる](https://qiita.com/yagince/items/6f9c082f639d0c833f79)

基本中の基本、FizzBuzzです。言語チュートリアルを終えたら脳みそに定着しているか確認も兼ねてまずやってみると良いと思います。ボリューム的には実装するだけなら30分もあればできてしまうでしょう。そこから突き詰めるかは本人次第ですが、そこまで根深くやるものでもないと思います。

## ブラックジャック

[プログラミング入門者からの卒業試験は『ブラックジャック』を開発すべし](https://qiita.com/hirossyi73/items/cf8648c31898216312e5)
[【PHP編】初心者卒業試験　ブラックジャック](https://qiita.com/qwertyuiopngsdfg/items/652c874b9da807c7a6a0#_reference-2338c734b07339de117c)
[Vue.jsでブラックジャックを作ってみた](https://qiita.com/t2kojima/items/88a924fa3807909e0488#_reference-a50ae51b9680b940b6d3)
[Kotlinでブラックジャック作ってみた](https://qiita.com/matyahiko2831/items/901554ba7c5680c06bc9)

シンプルなルールと気を付けなければハマる箇所があるという点、webか否かにかかわらず使える点でいい教材。

新規に言語を学ぼうという人にとっても同じルール同じ流れを別言語で実装するという事になるので、色々比較出来て良いかと思います。

ただ、アルゴリズムの訓練的な要素が強く、何かしらのサービスに繋げていきたい人にとってはベクトルが若干違うかもしれませんね。GUIベースで作っていくのが良いかも。多少ハードルが高くなるので完全な初心者には辛いかな？本当に「入門者からの卒業」向けという所でしょうか。

## TODOリスト

[Vue.jsとRailsでTODOアプリのチュートリアルみたいなものを作ってみた](https://qiita.com/naoki85/items/51a8b0f2cbf949d08b11)
[ReactでTODOリストを作る](https://qiita.com/NanayaKto/items/6fc4b5056c109c26b1ec)
[Django REST frameworkでTodoアプリを作ってみる](https://qiita.com/KojiOhki/items/5be98eeae72dca2260bc)
[[Swift] FirebaseでToDoリストを作ってみた for clean architecture](https://qiita.com/eKushida/items/68f21e2e43914f394b43)

ブラックジャックとの違いはデータの永続化まで学びやすい点、GUIベースの学習をしやすい点、アルゴリズムが簡潔になりやすい点でしょうか。アプリケーション開発の入門なら一度やっておいて損はないでしょう。

私もよくTODOリストアプリの実装は時間も取られないしやってみたりしますが、ユーザー由来の入力データ受け入れ、保持、出力、その間にするべき事（正規化やデータの構造化等）を１アプリケーション内で学べて良いなと思う反面成果物への物足りなさを感じる事があります。

如何に簡単に基本的な実装を実現できるかさくっと体験する時に有用だと思います。フレームワーク向け。でもボリューム不足なのでがっつり学ぶ時には追加の何かを求める事になりそうです。

## チャットアプリ
[Vue.js2とFirebaseでLINEライクなチャットアプリを作ってみる。](https://qiita.com/unotovive/items/5e6a94b4885c6e57dc84)
[PlayFrameworkでチャットアプリ](https://qiita.com/hys-rabbit/items/32967e8a76b2c894ead1)
[SpringBootでチャットアプリを作る](https://qiita.com/duke-gonorego/items/7700ada96bf3804b19e3)
[Swiftでソケット通信するチャットアプリ](https://qiita.com/ytakzk/items/c0a3af0f1b9e5a349d05)

定番ですね。TODOリストの制作を終えたらチャットに手を出していい頃でしょう。いきなりやるのはちょっとハードル高め。非同期な実装をするか否かで難易度がぐっと変わります。2ch(5ch)かLINEか的な。こちらもフレームワークを使った実装が良いでしょう。

特にLINEライクな実装にはWebSocketの利用などライブラリが絡んでくることが考えられます。生実装悪くありませんがチュートリアルとしては辛い。


## Twitter(っぽい)アプリ
[初めてのRuby on Rails。Twitterみたいなサイトを作成しよう](https://qiita.com/soushiy/items/f49c466bd3ef78059912)
[Vue.js + ElectronでTwitterクライアントを作った ](https://razokulover.hateblo.jp/entry/2017/04/03/125053)
[C#でTwitterアプリを作る 第0回](http://laco0416.hatenablog.com/entry/2014/03/24/204509)
[Pythonでサクッと簡単にTwitterAPIを叩いてみる](https://qiita.com/ogrew/items/0b267f57b8aaa24f1b73)

かの有名なRailsチュートリアルでも採用されています。API叩くくらいなら気軽に行けますがクライアントの実装までやると本格的なアプリケーション開発になってきます。中級者以上向けかもしれません。一本のアプリ開発でとても多くの事を学べて個人的にかなりおすすめ。腰を据えてやりたい人は手を出してみましょう。ドキュメントも豊富な方。

## Webスクレイピング
[Python Webスクレイピング 実践入門](https://qiita.com/Azunyan1111/items/9b3d16428d2bcc7c9406)
[RubyでSeleniumを使ってスクレイピング](https://qiita.com/tomerun/items/9cb81d7a98150ff22f53)
[PHPネイティブのDOMによるスクレイピング入門](https://qiita.com/mpyw/items/c0312271819baee09132)
[goqueryでお手軽スクレイピング！](https://qiita.com/yosuke_furukawa/items/5fd41f5bcf53d0a69ca6)

こちらも一般的な学習方法。フレームワークに依存しない実装を学ぶ場合に良いです。

権利や負荷的な注意事項があるので初学者向けじゃないかもしれません。中級者入門編といった感じ。初学者でも例えば2chに投稿された画像をコマンド一発で回収したい！とか思った時成果物を使えるので実用的という意味でも一度やってみる事をおすすめします。


## おまけ ： どういう学習方法が良いのか

完全に個人の意見ですがチュートリアルで部分的な学習の進め方はやめた方がいいと思ってます。例えばHTMLを学ぶ→JavaScriptを学ぶ→CSSを学ぶ、のような分割した学習方法。

何故かというと、アプリケーション開発の流れにノイズが混じる可能性があるからです。あいつとあいつは相性が悪い、あいつより今はこっちの方が主流、これはもうすぐ非推奨になるから今のうちにこっちを・・・みたいな。経験ある方多いと思います。

さらに、毎回どれをどうやってと悩む事になります。分割した分だけ教材を探す羽目になって結構辛いです。

アプリケーション開発において何を使うか悩むフェーズは初学者にとってかなり先の話なので、スムーズに進めるために初めから敷かれたレールの上を走るだけの学習をした方が良い、と思ったりします。

そういう意味ではプログラミング学習サイトを利用するのも一つの手ですね。悩む事がぐっと減るし、最近の学習サービスは進捗率とかレベルとかでモチベーションの維持を促してくれるので。

先にこれ勉強したら何が出来るようになるのか掲示してくれるなら私も全力で支持するんですが、そういうサイト少ないんですよね。

占いの雑誌みたいに色々選んでいったらそれに合った学習チャートが始まるとかできたら面白いですよね。web → 静的 → ホームページ → 言語選択 → ・・・みたいな。かなり難しそうですが。

## おわりに

こんな勉強方法いいよ！というのがあればコメントで教えて頂けると幸いです。技術者はいつだって初心者になれるので、良いチュートリアルの情報は思っているより需要あると思います。
































