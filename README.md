これはOpenFDTDのVersion 4.2.3 (2025/02/01)　をSYCLによりIntelや他のGPUで実行可能にするためのリポジトリです。
"https://ss023804.stars.ne.jp/OpenFDTD/index.html" よりダウンロードしたものにmasterブランチのファイルをマージ
しています。

実行するデバイスを指定するオプション" -txp <num> "を追加しました。ofd_syclを実行時に
ofd_sycl -txp 1 data/sample/1st_sample.ofd
のように指定して、sycl-lsで得られたデバイスの番号<num>を指定して実行します。

===== 

This is for making SYCL executable of OpenFDTD Version 4.2.3 (2025/02/01), which will run on Intel and other GPUs.
It is downloaded from "https://ss023804.stars.ne.jp/OpenFDTD/index.html" and the files in master branch are merged.

The option "-txp <num>" to specify the execution device has been added. When running ofd_sycl, you can execute it by specifying the device number <num> obtained via sycl-ls, for example:
ofd_sycl -txp 1 data/sample/1st_sample.ofd
This allows explicit selection of SYCL-compatible devices (e.g., GPUs, CPUs) listed in the sycl-ls output.
