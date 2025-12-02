# =========================================================
# [긴급 패치] Scipy 1.7+ 호환성 패치
# =========================================================
import scipy.linalg
if not hasattr(scipy.linalg, 'pinv2'):
    scipy.linalg.pinv2 = scipy.linalg.pinv
# =========================================================

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 라이브러리 임포트
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# PyExplainer 임포트
try:
    from pyexplainer_core import PyExplainer
except ImportError:
    from pyexplainer import PyExplainer

# 1. 데이터 준비
data = load_breast_cancer()

# ★★★ [핵심 수정] 컬럼 이름의 공백(" ")을 밑줄("_")로 변경 ★★★
# PyExplainer는 공백이 있는 컬럼명을 제대로 인식하지 못해 에러를 냅니다.
feature_names = [name.replace(' ', '_') for name in data.feature_names]

X = pd.DataFrame(data.data, columns=feature_names)
y = pd.Series(data.target, name="target") # y 이름 설정 유지

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

target_instance = X_test.iloc[[0]]
target_label = y_test.iloc[[0]]

# 2. 모델 정의
models = {
    "XGBoost": Pipeline([
        ('scaler', StandardScaler()), 
        ('clf', XGBClassifier(n_estimators=50, max_depth=3, eval_metric='logloss', random_state=42))
    ]),
    "LightGBM": Pipeline([
        ('scaler', StandardScaler()), 
        ('clf', LGBMClassifier(n_estimators=50, verbose=-1, random_state=42))
    ]),
    "CatBoost": Pipeline([
        ('scaler', StandardScaler()), 
        ('clf', CatBoostClassifier(iterations=50, verbose=0, allow_writing_files=False, random_state=42))
    ])
}

# 3. 실행 루프
print(f"{'='*60}")
print(f"Target Instance Index: {target_instance.index.values[0]}")
print(f"{'='*60}\n")

for name, model in models.items():
    print(f"🚀 Processing [{name}]...")
    
    try:
        # (1) 모델 학습
        model.fit(X_train, y_train)
        
        # (2) PyExplainer 초기화
        py_explainer = PyExplainer(
            X_train=X_train,
            y_train=y_train,
            indep=X_train.columns,
            dep="target", 
            blackbox_model=model,
            class_label=["Malignant", "Benign"]
        )
        
        # (3) 설명 생성
        rules = py_explainer.explain(
            X_explain=target_instance,
            y_explain=target_label,
            search_function="crossoverinterpolation",
            top_k=3
        )
        
        # (4) 결과 출력
        print(f"✅ [{name}] Explanation Success!")
        
        if hasattr(rules, 'keys'):
            if 'top_k_positive_rules' in rules and rules['top_k_positive_rules'].shape[0] > 0:
                top_rule = rules['top_k_positive_rules'].iloc[0]['rule']
                print(f"   -> Top Rule: {top_rule}")
            else:
                print("   -> (양성 규칙 없음)")
            
            if 'synthetic_predictions' in rules:
                print(f"   -> Prediction Prob: {rules['synthetic_predictions'][0]}")
        
    except Exception as e:
        print(f"❌ [{name}] Error: {e}")
        import traceback
        traceback.print_exc()
        
    print("-" * 60)