pip install pyinstaller
python ./pyinstaller.exe --noconsole --icon=images/icon.ico --exclude-module "images" --add-data "images;images" --exclude-module "config" --add-data "config.ini;." --exclude-module "ui" --add-data "main.ui;." --noconfirm CardDetectorProgram.py

