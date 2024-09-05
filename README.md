# mlflow-quickstart

![image](https://github.com/user-attachments/assets/ea1f70db-845f-4e9b-b49e-bc5ec05e7282)

## Introduce
MLflow의 전반적인 기능을 실습. MLflow 공식 문서를 참조하여 작성하였다.

Fashion-Mnist을 학습 시키는 코드에 MLflow를 적용하였다.

local 환경에서 MLflow가 아닌 실제 MLOps에 적용하기 위함으로 Docker를 필수적으로 설치해준다.

## Install
python 3.8 버전을 사용하였다.

필요 라이브러리들을 설치해준다. torch 와 torchvision은 CUDA 11.8을 기준으로 설치하였으며 각 환경에 맞게 설치해준다.
[pytorch install 명령어 확인](https://pytorch.org/get-started/previous-versions/)


```shell
pip install mlflow
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
pip install optuna plotly
pip install optuna-dashboard
```

