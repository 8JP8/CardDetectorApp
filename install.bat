pip install pyinstaller
python ./pyinstaller.exe --noconsole --icon=images/icons/icon.ico --exclude-module "images" --add-data "images;images" --exclude-module "config" --add-data "config.ini;config.ini" --exclude-module "ui" --add-data "main.ui;main.ui" --noconfirm CardDetectorProgram.py
