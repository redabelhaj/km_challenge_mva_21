#/bin/bash
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
python3 main.py --config configs/config_aug_whitening.yaml
python3 main.py --config configs/config_aug.yaml
python3 main.py --config configs/config_whitening.yaml
python3 main.py --config configs/config.yaml
python3 ensemble_final.py