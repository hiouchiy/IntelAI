# Resnet50量子化デモ用環境構築手順
これはIntel TensorFlowおよびIntel® OpenVINO™ Toolkitを用いたResnet50の量子化デモ用の環境構築ガイドです。

デモの本編はJupyter Notebook上にて実施されるので、当該Notebookを立ち上げるまでに必要な手順を記しています。

## 前提条件
- Ubuntu 18.04
- Docker 最新版（インストール方法は[こちら](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-18-04)を参照）

## Jupyter Notebookを立ち上げるまで

### ベアメタルマシン、または、VMへSSHでログイン
- ssh ユーザーID@IPアドレス

### Dockerイメージをダウンロード＆Dockerコンテナ起動
- docker pull hiouchiy/openvino2020r3-configured-on-cpu-with-inteltf1.15.2
- docker run -it -p 8008:8008 --privileged -v ~/imagenet:/imagenet hiouchiy/openvino2020r3-configured-on-cpu-with-inteltf1.15.2

>注:
>このコンテナには、Python v3.6、Intel TensorFlow v1.15.2、Intel® OpenVINO™ Toolkit 2020R3、および周辺ライブラリやツールがプリインストールされています。どのように作ったかを知りたい場合は最後の「おまけ（Dockerイメージを作るまで）」を参照下さい。

>注:
>このデモでは、ホストマシンにImageNet Large Scale Visual Recognition Competition 2012のValidationデータ（50,000枚）を準備しています。本データは[ImageNet](http://www.image-net.org/challenges/LSVRC/2012/)の公式サイトからダウンロードください。または、適当なデータをご利用ください。

### Githubから必要なリポジトリをクローン
- cd ~
- git clone https://github.com/IntelAI/models.git models && cd models 
- git checkout v1.5.0 (※今回はIntel TensorFlow v1.15.2 を使うため)
- cd ~
- git clone https://github.com/intel/lp-opt-tool.git ilit && cd ilit/examples/tensorflow/image_recognition
- wget https://raw.githubusercontent.com/hiouchiy/IntelAI/master/tensorflow_quantization/how_to_quantize_with_ilit.ipynb

### OpenVINOのAccuracyChekerとPost-training Optimization Toolkitをセットアップ
- cd /opt/intel/openvino/deployment_tools/open_model_zoo/tools/accuracy_checker
- python3 setup.py install
- cd /opt/intel/openvino/deployment_tools/tools/post_training_optimization_toolkit
- python3 setup.py install
- source /opt/intel/openvino/bin/setupvars.sh

### Jupyter Notebookを起動
- cd ~
- KMP_AFFINITY=granularity=fine,compact,1,0 KMP_BLOCKTIME=1 KMP_SETTINGS=1 OMP_NUM_THREADS=サーバーの物理コア数 numactl -N 0 -m 0 nohup jupyter notebook --ip 0.0.0.0 --allow-root > /dev/null 2>&1 &

>注:
>ちなみにこの4つの環境変数はIntel Tensorflow用です。それぞれ下記のような意味があります。更に詳しく知りたい場合は[こちら](https://software.intel.com/content/www/us/en/develop/articles/maximize-tensorflow-performance-on-cpu-considerations-and-recommendations-for-inference.html)へ。
>   - KMP_AFFINITY・・・
>   - KMP_BLOCKTIME・・・
>   - KMP_SETTINGS・・・
>   - OMP_NUM_THREADS・・・1ソケットの物理コア数

### Webブラウザからアクセス
- https://<ホストマシンのIPアドレス>:8008
- パスワードは「intel」
- quantizationフォルダ内の「QuantizationHowto.ipynb」をクリック
- あとはNotebook内の指示に従う

## おまけ（Dockerイメージを作るまで）
- mkdir workspace
- cd workspace
- [こちら](https://raw.githubusercontent.com/hiouchiy/IntelAI/master/tensorflow_quantization/Dockerfile)のDockerfileをダウンロード
- docker build . -t イメージ名
- docker run -it -p 8008:8008 --privileged "イメージ名"
- [こちら](https://qiita.com/JIwatani/items/ae1acb0878610fef3da8)を参考にJupyter Notebookをコンテナ上にインストール
- ctrl+p, ctrl+qでコンテナを一旦抜ける（コンテナを終了はしない）
- docker ps -a で今抜けたコンテナのIDを確認
- docker commit 今抜けたコンテナのID 新イメージのタグ名
- docker login
- docker push 新イメージのタグ名

## 参考
- [Intel Tensorflow Quantization Tool](https://github.com/IntelAI/tools/blob/master/api/README.md#quantization-python-programming-api-quick-start)
- [Intel Tensorflow Quantization Tool（日本語訳）※非公式](https://github.com/hiouchiy/IntelAI/blob/master/tensorflow_quantization/tf_quantization_ja.md)
- [OpenVINO Post-training Optimization Toolkit](https://docs.openvinotoolkit.org/latest/_README.html)
