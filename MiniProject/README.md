# 직원 퇴사(Attrition) 예측 모델링 프로젝트

## 1. 프로젝트 개요

### 1.1. 목표

IBM에서 제공한 HR 데이터를 기반으로 직원의 퇴사 여부(Attrition)를 예측하는 머신러닝 모델을 구축합니다. 이를 통해 퇴사 가능성이 높은 직원을 사전에 파악하고, 기업의 인적 자원 관리(HRM) 전략 수립에 기여하는 것이 목적입니다.

### 1.2. 데이터셋

* **출처:** IBM HR Analytics Employee Attrition & Performance
* **파일:** `WA_Fn-UseC_-HR-Employee-Attrition.csv`
* **구성:** 총 1,470명의 직원 데이터, 35개 변수 포함

### 1.3. 사용 기술 및 라이브러리

* **언어:** Python
* **주요 라이브러리:**

  * `pyspark.sql`: 대용량 데이터 처리를 위한 Spark DataFrame API
  * `pyspark.ml`: 머신러닝 모델링을 위한 MLlib
  * `pandas`: 중간 변환 및 일부 데이터 처리
  * `matplotlib`, `seaborn`: 시각화

---

## 2. 데이터 분석 (EDA)

### 2.1. 데이터 불러오기 및 기본 탐색

```python
# 예시 코드
df = spark.read.csv('WA_Fn-UseC_-HR-Employee-Attrition.csv', header=True, inferSchema=True)
df.printSchema()
print(f"데이터 크기: ({df.count()}, {len(df.columns)})")
```

* **결과:** 1,470행, 35열로 구성됨. 각 컬럼은 적절한 데이터 타입으로 로드됨.

### 2.2. 결측치 및 이상치 탐지

* **결측치:** 존재하지 않음
* **이상치:** Box Plot을 통해 `MonthlyIncome`, `TotalWorkingYears` 등에서 일부 이상치 존재. 다만, 실제 분포일 가능성이 있어 그대로 유지함.

### 2.3. 주요 변수 분포 분석

* **목표 변수 (Attrition):** Yes 237명 (16.1%), No 1,233명 (83.9%) — 클래스 불균형 존재
* **시사점:** 정확도 외에도 F1-score, AUC 등 다른 평가지표 필요
* **기타 변수 분석:** 퇴사 그룹과 잔류 그룹 간 변수 분포 비교 (예: 월급, 직무 만족도 등)

---

## 3. 데이터 전처리 및 특성 공학

### 3.1. 불필요한 컬럼 제거

* `StandardHours`, `EmployeeCount`: 모든 값이 동일
* `EmployeeNumber`: 고유 식별자 → 제거

### 3.2. 피처 분류

* **수치형 피처:** `Age`, `MonthlyIncome` 등 14개
* **범주형 피처:** `Department`, `JobRole`, `Education` 등 16개

### 3.3. 전처리 파이프라인 구성

* **범주형 처리:**

  * `StringIndexer`: 문자열을 인덱스로 변환
  * `OneHotEncoder`: 인덱스를 원-핫 벡터로 변환
* **수치형 처리:**

  * `VectorAssembler`: 수치형 컬럼을 벡터로 통합
  * `StandardScaler`: 표준화
* **최종 통합:**

  * `VectorAssembler`: 전 피처를 `features` 벡터로 통합

---

## 4. 모델링 및 학습

### 4.1. 선택 모델

* **모델:** 로지스틱 회귀 (Logistic Regression)
* **이유:** 해석 용이, 이진 분류 기준선으로 적합

### 4.2. 데이터 분할

* 학습용 80%, 테스트용 20% 무작위 분할

### 4.3. 모델 학습

```python
# 예시 코드
pipeline = Pipeline(stages=[...전처리 단계..., lr_model])
pipeline_model = pipeline.fit(train_data)
predictions = pipeline_model.transform(test_data)
```

---

## 5. 모델 평가

### 5.1. 평가 지표 결과 (예시)

| 평가지표          | 점수     | 설명                         |
| ------------- | ------ | -------------------------- |
| 정확도 Accuracy  | 0.8851 | 전체 예측 중 올바르게 맞춘 비율         |
| 정밀도 Precision | 0.8753 | '퇴사'로 예측한 직원 중 실제 퇴사한 비율   |
| 재현율 Recall    | 0.8851 | 실제 퇴사자 중 모델이 '퇴사'로 잘 맞춘 비율 |
| F1-Score      | 0.8787 | 정밀도와 재현율의 조화 평균            |
| AUC           | 0.8524 | 클래스 구분 능력 (0.5보다 높을수록 좋음)  |

### 5.2. 결과 해석

* 전체적으로 정확도와 AUC가 높아 준수한 성능을 보임
* 클래스 불균형 문제로 인해 F1-score와 재현율 지표 중요성 강조

---

## 6. 결론 및 향후 계획

### 6.1. 결론

* Spark ML 기반 파이프라인을 활용해 직원 퇴사 예측 모델을 성공적으로 구축
* 데이터 탐색, 전처리, 모델 학습 및 평가 과정을 체계적으로 수행함

### 6.2. 향후 개선 방향

* **모델 고도화:**

  * Random Forest, Gradient Boosted Tree 등 다른 알고리즘 실험
  * CrossValidator를 통한 하이퍼파라미터 튜닝
* **피처 엔지니어링:**

  * 경력 대비 소득, 직무/회사 근속 비율 등 파생 변수 생성
* **불균형 대응:**

  * SMOTE 등 오버샘플링 기법 도입

---

](https://github.com/sswwdk/start_spark/blob/main/MiniProject/09_Practice.ipynb)
