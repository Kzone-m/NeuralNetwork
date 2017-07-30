from reppy.cache import RobotsCache
from bs4 import BeautifulSoup
from selenium import webdriver
from urllib.request import urlopen
import os, time

# (1) クロールするurlを決める
# target_url = "https://www.yahoo.co.jp"


# (2) robot.txtを読み込むため際に使用するインスタンスの作成
robots = RobotsCache(100)

# (3) もし、robot.txtを読み込んでみて、クロール許可をもらえたら、先の処理に進む
if robots.allowed(target_url, 'python program'):

    # (4) Javascriptで生成されたコードでもクロールできるようにPhatomJSインスタンスを作成する
    driver = webdriver.PhantomJS()

    # (5) 作成したインスタンスのGetリクエストを呼ぶメソッドに対象のurlを引数として与え、domの情報を手に入れる
    driver.get(target_url)

    # (6) 先ほど取得したdomの情報をutf-8でエンコードして、クロール対象ページの情報をbyte型として保持する
    html = driver.page_source.encode('utf-8')

    # (7) BSに先ほど取得したbyte型のdom情報を引数として渡して, lxmlとしてdom情報を再構築
    soup = BeautifulSoup(html, "lxml")

    # (8) 取得したdom情報から、特定のimgタグのリストを取得する
    for i in soup.find_all('img'):
        # (9) 特定のimgタグリストから一つずつsrc == urlを取得する
        img_url = i['src']
        print(str(i) + " times")
        print("img_url: ", img_url)
        try:
            print("here3")
            # (10) 取得したurlを元に写真情報を取得する
            img = urlopen(img_url)

            # (11) urlを元に保存するときのファイル名を決めてあげる
            file_name = img_url.split('/')[-1]

            # (12) 保存する際に、作業中のディレクトリに同じファイル名が存在していないか確認してあげる
            if not os.path.exists(file_name):
                # (13) 先ほど指定したファイル名で画像を保存するためのファイルを作成してあげる
                with open(file_name, 'wb') as f:
                    # (14) バイト文字列で画像をファイルに書き込みしてあげる
                    f.write(img.read())

            # (15) クロール先にのサーバーに迷惑がかからないようにするため、一つの処理が終わる度に2秒間停止してあげる
            time.sleep(2)
        except:
            print("エラーが発生したよ(ﾟ∀ﾟ)")

