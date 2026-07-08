"""
서울 부동산 실거래가 군집 분석 (자기주도 프로젝트)
- 파생 변수 생성: 건축 경과 연수, 단위 면적당 가격
- 정규화(StandardScaler) 후 K-Means(k=4) 군집화
- 군집 중심점 + 재건축 연한(30년) 기준선 시각화
- 데이터: 7_seoul_2025_transaction.csv (2025 서울 아파트 실거래)
"""

# Colab 한글 폰트 설치 (최초 1회 실행 후 [런타임] -> [세션 다시 시작])
# !sudo apt-get install -y fonts-nanum && sudo fc-cache -fv && rm -rf ~/.cache/matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 폰트/스타일 설정
plt.rc('font', family='NanumBarunGothic')
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

# 1. 데이터 로드 및 결측치 제거
df = pd.read_csv('7_seoul_2025_transaction.csv')
df = df.dropna()

# 2. 파생 변수 생성
df['dealAmount_num'] = df['dealAmount'].str.replace(',', '').astype(float)  # 거래금액(문자->실수)
df['building_age'] = 2025 - df['buildYear']                                # 건축 경과 연수
df['price_per_sqm'] = df['dealAmount_num'] / df['excluUseAr']              # 단위 면적당 가격

# 3. 정규화 + K-Means 군집화
features = df[['building_age', 'price_per_sqm']]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(features_scaled)

# 4. 시각화
plt.figure(figsize=(14, 8))
colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
for i in range(4):
    c = df[df['cluster'] == i]
    plt.scatter(c['building_age'], c['price_per_sqm'],
                color=colors[i], label=f'Cluster {i}', alpha=0.6,
                edgecolors='white', s=60)

# 군집 중심점 강조 (정규화 역변환)
centers = scaler.inverse_transform(kmeans.cluster_centers_)
for i, center in enumerate(centers):
    plt.scatter(center[0], center[1], color='crimson', marker='X',
                s=350, edgecolor='black', linewidth=2, zorder=5)
    plt.text(center[0] + 1.5, center[1] + 150, f'Core {i}',
             fontsize=13, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

# 재건축 연한(30년) 기준선
plt.axvline(x=30, color='indigo', linestyle='--', linewidth=2, label='재건축 연한 (30년)')

plt.title('2025 서울 부동산 가치 군집 분석: 건축 연수 대비 단위 면적당 가격',
          fontsize=18, fontweight='bold', pad=20)
plt.xlabel('건축 경과 연수 (년)', fontsize=14, fontweight='bold')
plt.ylabel('단위 면적당 가격 (만원/㎡)', fontsize=14, fontweight='bold')
plt.legend(title='Segments', fontsize=11, loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
