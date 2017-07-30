from reppy.cache import RobotsCache
from bs4 import BeautifulSoup
from selenium import webdriver
from urllib.request import urlopen
import os
import requests

"""References
・Pythonでスクレイピング for Javascript
    http://qiita.com/_akisato/items/2daafdbc3de544cf6c92
・BSを使ってスクレイピング
    http://qiita.com/rusarusa/items/d7f014ba80d6fe7a3e07
・PythonでWEB上の画像をまとめてダウンロード
    http://www.dyesac.com/pythonでweb上の画像をまとめてダウンロード/
・画像クローラー
    http://qiita.com/komakomako/items/dd380f980e56e70fa321

Targets:
・https://reverb.com/jp/marketplace/electric-guitars
・https://www.yahoo.co.jp
"""

# (1) クロールするurlを決める
target_url = "https://www.yahoo.co.jp"

# (2) robot.txtを読み込むため際に使用するインスタンスの作成
robots = RobotsCache(100)

# (3) もし、robot.txtを読み込んでみて、クロール許可をもらえたら、先の処理に進む
if robots.allowed(target_url, 'python program'):
    # (4) Javascriptで生成されたコードでもクロールできるようにPhatomJSインスタンスを作成する
    driver = webdriver.PhantomJS()
    # (5) 作成したインスタンスのGetリクエストを呼ぶメソッドに対象のurlを引数として与え、domの情報を手に入れる
    driver.get(target_url)
    # <selenium.webdriver.phantomjs.webdriver.WebDriver (session="b140b9a0-74d3-11e7-b434-8b9f5b309f17")>
    # type(driver)
    # <class 'selenium.webdriver.phantomjs.webdriver.WebDriver'>

    # (6) 先ほど取得したdomの情報をutf-8でエンコードして、クロール対象ページの情報をbyte型として保持する
    html = driver.page_source.encode('utf-8')
    # type(html)
    # <class 'bytes'>

    # html = requests.get(target_url)
    # < Response [200]>
    # type(html)
    # <class 'requests.models.Response'>
    # html = html.text
    # type(html)_
    # <class 'str'>

    # (7) BSに先ほど取得したbyte型のdom情報を引数として渡して, lxmlとしてdom情報を再構築
    soup = BeautifulSoup(html, "lxml")
    # type(soup)
    # <class 'bs4.BeautifulSoup'>

    # for i in soup.find_all('img'):
    #     print(i['src'])

    # (8) 取得したdom情報から、最初に発見できるimgタグのsrcを取得する
    img_url = soup.find('img')['src']
    # urlopenを使用する際に例外が発生するかもしれないのでtry exceptブロックで囲む
    try:
        # (9) imgタグのsrc == url をurllib.request.urlopenの引数として渡して、写真情報を取得する
        img = urlopen(img_url)

        # (10) 今回は、urlを"/"で分解してあげて、最後の部分をファイル名として使用する
        #   ex:
        #       url: https://s.yimg.jp/images/top/sp2/cmn/logo-170307.png
        #       img_url.split('/'): ['https:', '', 's.yimg.jp', 'images', 'top', 'sp2', 'cmn', 'logo-170307.png']
        #       img_url.split('/')[-1]: 'logo-170307.png'
        file_name = img_url.split('/')[-1]

        # (11) 現在作業しているディレクトリに同じ名前のファイルが存在しているかを確認する
        if not os.path.exists(file_name):
            # (12) 先ほど取得したファイル名で新しいファイルを作成してあげる
            with open(file_name, 'wb') as f:
                # (13) 開いたファイルに写真情報をバイト文字列で書き込みしてあげる
                f.write(img.read())
    except:
        print("エラーが発生したよ(ﾟ∀ﾟ)")
