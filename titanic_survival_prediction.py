"""
타이타닉 생존 예측 (DACON 연습 대회)
- 모델: Random Forest Classifier
- 특징 생성: 이름에서 호칭(Title) 추출
- 결측치 처리: 호칭별 평균 나이 / 최빈값 / 중앙값
- 인코딩: One-Hot Encoding
- 제출 정확도: 약 0.767
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 1. 데이터 로드
colab_path = ''  # Colab에서 드라이브 마운트 시 경로 변경
df_train = pd.read_csv(colab_path + 'train.csv')
df_test = pd.read_csv(colab_path + 'test.csv')


# 2. 전처리 함수 (특징 생성 + 결측치 처리 + 인코딩)
def preprocess_data(df):
    df = df.copy()

    # 이름에서 호칭(Title) 추출 및 그룹화
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
         'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})

    # 나이 결측치: 호칭별 평균 -> 전체 평균 보완
    df['Age'] = df['Age'].fillna(df.groupby('Title')['Age'].transform('mean'))
    df['Age'] = df['Age'].fillna(df['Age'].mean())

    # 기타 결측치 처리
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    # 불필요한 컬럼 제거
    df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

    # 원-핫 인코딩
    df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Title'], drop_first=True)
    return df


# 3. 전처리 적용 및 학습/테스트 컬럼 정렬
y_train = df_train['Survived']
X_train = preprocess_data(df_train.drop(columns=['Survived']))
X_test = preprocess_data(df_test)
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# 4. 모델 학습
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)

# 5. 예측 및 제출 파일 저장
pred_y = rf.predict(X_test)
df_submission = pd.read_csv(colab_path + 'submission.csv')
df_submission['Survived'] = pred_y
df_submission.to_csv(colab_path + 'harksu_submission.csv', index=False)
print("예측 완료 - 'harksu_submission.csv' 생성")
