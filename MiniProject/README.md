# 💼 직원 퇴사 예측 모델링 프로젝트 (Employee Attrition Prediction)

## 1. 📌 프로젝트 개요

### 1.1. 목표
IBM에서 제공한 HR 데이터를 활용하여 직원의 퇴사 여부(Attrition)를 예측하는 머신러닝 모델을 구축합니다. 퇴사 가능성이 높은 직원을 사전에 파악하여 인적 자원 관리(HRM) 전략 수립에 기여하는 것이 목적입니다.

### 1.2. 데이터셋
- **출처:** [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- **파일:** `WA_Fn-UseC_-HR-Employee-Attrition.csv`
- **구성:** 1,470명의 직원 정보, 총 35개의 인적 자원 관련 변수 포함

### 1.3. 사용 기술 및 라이브러리
- **언어:** Python
- **라이브러리:**
  - `pyspark.sql` – 대용량 데이터 처리 (Spark DataFrame)
  - `pyspark.ml` – 머신러닝 모델링 (MLlib)
  - `pandas` – 중간 데이터 조작
  - `matplotlib`, `seaborn` – 데이터 시각화

---

## 2. 🔍 데이터 탐색 (EDA)

### 2.1. 데이터 불러오기
```python
df = spark.read.csv('WA_Fn-UseC_-HR-Employee-Attrition.csv', header=True, inferSchema=True)
df.printSchema()
print(f"데이터 크기: ({df.count()}, {len(df.columns)})")
