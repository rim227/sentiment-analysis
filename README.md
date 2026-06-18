# 텍스트 감성 분석 시스템

## 프로젝트 개요
텍스트 데이터 정제 여부가 감성 분석 모델 성능에 미치는 영향을
실험적으로 검증하는 웹 기반 시스템입니다.
정제 전·후 모델 결과를 동시에 비교하여 데이터 품질의 중요성을 시각화했습니다.

## 주요 기능
- 텍스트 정제 파이프라인 구현
  - 특수문자 제거
  - 불용어 처리 (konlpy Okt)
  - 반복 문자 정규화
- 정제 전·후 감성 분석 결과 비교
  - 정제 전: 긍정 55% → 정제 후: 긍정 78% (+23%p 향상)
- 단계별 텍스트 변화 시각화
- 분석 히스토리 자동 저장

## 사용 기술
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/ScikitLearn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)

## 실행 방법
```bash
pip install streamlit scikit-learn pandas konlpy matplotlib
streamlit run app.py
```

## 프로젝트 구조
```
sentiment-analysis/
├── app.py
├── model.py
├── preprocess.py
└── README.md
```
