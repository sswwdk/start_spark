직원 퇴사(Attrition) 예측 모델링 프로젝트
1. 프로젝트 개요
1.1. 목표

본 프로젝트의 목표는 IBM에서 제공한 HR 데이터를 활용하여 직원의 퇴사 여부(Attrition)를 예측하는 머신러닝 모델을 구축하는 것입니다. 이를 통해 퇴사 가능성이 높은 직원을 사전에 파악하고, 기업의 인적 자원 관리(HRM) 전략 수립에 기여하고자 합니다.

1.2. 데이터셋

출처: IBM HR Analytics Employee Attrition & Performance

데이터: WA_Fn-UseC_-HR-Employee-Attrition.csv

특징: 1,470명의 직원 데이터와 35개의 인적 자원 관련 변수(피처)로 구성됨

1.3. 사용 기술 및 라이브러리

언어: Python

핵심 라이브러리:

pyspark.sql: 대용량 데이터 처리를 위한 Spark DataFrame API

pyspark.ml: 머신러닝 모델링을 위한 Spark MLlib 라이브러리

pandas: 데이터 시각화를 위한 중간 변환

matplotlib & seaborn: 데이터 탐색 및 결과 시각화

<br>

2. 데이터 분석 (EDA)
2.1. 데이터 불러오기 및 기본 탐색

프로젝트의 첫 단계로, 데이터를 Spark DataFrame으로 로드하고 기본적인 구조(스키마, 행/열 개수)를 확인합니다.

code
Python
download
content_copy
expand_less

# 코드 예시
df = spark.read.csv('WA_Fn-UseC_-HR-Employee-Attrition.csv', header=True, inferSchema=True)
df.printSchema()
print(f"데이터 크기: ({df.count()}, {len(df.columns)})")

확인 결과: 총 1,470개의 행과 35개의 열로 구성되어 있으며, 각 컬럼은 적절한 데이터 타입으로 로드되었습니다.

2.2. 결측치 및 이상치 탐지

결측치: 데이터셋의 모든 컬럼을 검사한 결과, 결측치(NULL 또는 NaN)는 발견되지 않았습니다.

이상치: 주요 수치형 변수에 대해 Box Plot을 시각화하여 이상치를 탐색했습니다. MonthlyIncome, TotalWorkingYears 등에서 다수의 이상치가 발견되었으나, 이는 실제 데이터의 자연스러운 분포일 수 있으므로 초기 모델링 단계에서는 제거하지 않고 그대로 사용하기로 결정했습니다.

![alt text](images/boxplot_example.png)

(예시: images 폴더에 시각화 이미지를 저장하고 링크)

2.3. 주요 변수 분포 분석

목표 변수(Attrition): 퇴사(Yes)와 잔류(No)의 비율을 확인한 결과, 각각 237명(16.1%)과 1,233명(83.9%)으로 심각한 클래스 불균형이 존재함을 확인했습니다. 이는 모델 평가 시 정확도(Accuracy) 외에 F1-Score, AUC 등의 지표를 함께 고려해야 함을 시사합니다.

![alt text](images/attrition_dist_example.png)

주요 수치형/범주형 변수: 퇴사 그룹과 잔류 그룹 간의 주요 변수 분포를 비교 분석하여 퇴사에 영향을 미치는 요인을 탐색했습니다. (예: 월급, 직무 만족도 등)

<br>

3. 데이터 전처리 및 특성 공학

모델 학습을 위해 데이터를 정제하고 가공하는 단계입니다. Spark ML의 Pipeline을 사용하여 모든 전처리 과정을 체계적으로 관리합니다.

3.1. 불필요한 컬럼 제거

모든 값이 동일하거나(StandardHours, EmployeeCount), 고유 식별자(EmployeeNumber)인 컬럼은 모델 학습에 기여하지 않으므로 제거했습니다.

3.2. 피처 분류

수치형 피처 (numerical_cols): Age, MonthlyIncome 등 14개

범주형 피처 (categorical_cols): Department, JobRole, Education 등 16개

3.3. 전처리 파이프라인 구성

범주형 피처:

StringIndexer: 문자열 카테고리를 숫자 인덱스로 변환

OneHotEncoder: 숫자 인덱스를 원-핫 인코딩 벡터로 변환

수치형 피처:

VectorAssembler: 여러 수치형 컬럼을 단일 벡터로 통합

StandardScaler: 통합된 벡터의 스케일을 표준 정규분포에 맞게 조정

피처 통합:

VectorAssembler: 전처리된 모든 피처(수치형+범주형)를 features라는 최종 벡터로 통합

<br>

4. 모델링 및 학습
4.1. 모델 선택

선택 모델: 로지스틱 회귀 (Logistic Regression)

선택 이유: 이진 분류 문제에 대한 해석이 용이하고, 모델 성능의 좋은 기준선(Baseline)을 제공하기 때문입니다.

4.2. 학습 및 테스트 데이터 분할

전체 데이터를 학습 데이터(80%)와 테스트 데이터(20%)로 무작위 분할하여 모델의 일반화 성능을 평가할 수 있도록 준비했습니다.

4.3. 모델 학습

앞서 구성한 Pipeline을 사용하여 학습 데이터에 대한 데이터 전처리 및 모델 학습을 동시에 수행했습니다.

code
Python
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
# 코드 예시
pipeline = Pipeline(stages=[...전처리 단계..., lr_model])
pipeline_model = pipeline.fit(train_data)
predictions = pipeline_model.transform(test_data)
<br>

5. 모델 평가

학습된 모델을 테스트 데이터에 적용하여 예측을 수행하고, 다음과 같은 분류 성능 지표를 통해 모델의 성능을 평가했습니다.

5.1. 평가 결과
평가지표	점수	설명
정확도 (Accuracy)	0.8851	전체 예측 중 올바르게 예측한 비율
정밀도 (Precision)	0.8753	'퇴사'로 예측한 직원 중 실제 퇴사한 비율
재현율 (Recall)	0.8851	실제 퇴사한 직원 중 '퇴사'로 예측한 비율
F1-Score	0.8787	정밀도와 재현율의 조화 평균
AUC	0.8524	모델이 퇴사자와 비퇴사자를 얼마나 잘 구별하는지 나타내는 지표

(위의 값은 예시이며, 실제 모델 실행 결과로 채워주세요.)

5.2. 결과 해석

모델은 약 88.5%의 정확도를 보이며 전반적으로 준수한 성능을 나타냈습니다.

AUC 점수가 0.85로, 무작위 예측(0.5)보다 훨씬 우수하게 클래스를 분류하는 능력을 갖추고 있음을 확인했습니다.

다만, 클래스 불균형 문제를 고려할 때 재현율과 F1-Score를 주목할 필요가 있으며, 실제 비즈니스 목적에 따라 추가적인 성능 향상 작업이 필요할 수 있습니다.

<br>

6. 결론 및 향후 계획
6.1. 결론

본 프로젝트를 통해 Spark ML 파이프라인을 활용하여 직원 퇴사 예측 모델을 성공적으로 구축했습니다. EDA를 통해 데이터의 특성을 파악하고, 체계적인 전처리 과정을 거쳐 로지스틱 회귀 모델을 학습시킨 결과, 유의미한 예측 성능을 확보할 수 있었습니다.

6.2. 향후 계획

성능 향상:

알고리즘: 랜덤 포레스트(Random Forest), GBT(Gradient-Boosted Trees) 등 더 강력한 앙상블 모델을 적용하여 성능 비교.

특성 공학: 경력 대비 소득, 직무/회사 근속 비율 등 새로운 파생 변수를 생성하여 모델 성능 향상 시도.

하이퍼파라미터 튜닝: CrossValidator를 사용하여 최적의 모델 파라미터를 탐색.

불균형 데이터 처리: SMOTE와 같은 오버샘플링 기법을 적용하여 소수 클래스(퇴사)에 대한 재현율을 높이는 방안 모색.
