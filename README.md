# AI / Machine Learning 실습

머신러닝과 데이터 분석을 직접 손으로 익히기 위해 진행한 실습 모음입니다.
데이터 전처리부터 모델 학습, 성능 평가·해석까지의 전체 흐름을 스스로 수행하는 데 목표를 두었습니다.

## 프로젝트

### 1. 타이타닉 생존 예측 (`titanic_survival_prediction.py`)
- **대회**: DACON 타이타닉 생존 예측 (연습 대회)
- **모델**: Random Forest Classifier (scikit-learn)
- **주요 작업**
  - 이름에서 호칭(Title)을 추출해 새로운 특징 생성
  - 나이 결측치를 호칭별 평균으로 보완, 그 외 결측치는 최빈값·중앙값으로 처리
  - One-Hot Encoding으로 범주형 변수 변환, train/test 컬럼 정렬
- **결과**: 제출 정확도 **약 0.767**

### 2. 서울 부동산 실거래가 군집 분석 (`seoul_realestate_kmeans.py`)
공개 실거래 데이터를 직접 전처리·군집화하고 시각화한 자기주도 프로젝트입니다.
- **모델**: K-Means Clustering (k=4), StandardScaler 정규화
- **주요 작업**
  - 파생 변수 생성: 건축 경과 연수(`building_age`), 단위 면적당 가격(`price_per_sqm`)
  - 정규화 후 K-Means로 4개 군집 분류
  - 군집 중심점과 재건축 연한(30년) 기준선을 함께 시각화하여 데이터 특성 해석
- **데이터**: 2025년 서울 아파트 실거래 데이터 (`7_seoul_2025_transaction.csv`)

## 그 외 실습
- Decision Tree + GridSearchCV 하이퍼파라미터 튜닝
- K-Means 군집 결과를 실제 타깃과 대조해 평가
- CNN 기반 이미지 분류 (MNIST, TensorFlow/Keras)

## 사용 기술
`Python` · `Pandas` · `NumPy` · `scikit-learn` · `TensorFlow/Keras` · `matplotlib` · `Google Colab`
