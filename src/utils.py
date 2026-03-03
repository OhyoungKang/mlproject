import os
import sys
import numpy as np
import pandas as pd
import dill

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.exception import CustomException
from src.logger import logging


# ============================================================================
# 객체 저장/로드 함수
# ============================================================================

def save_object(file_path, obj):
    """
    객체를 pickle 파일로 저장

    Parameters:
    - file_path: 저장할 파일 경로 (예: artifacts/model.pkl)
    - obj: 저장할 객체 (모델, 전처리기 등)
    """
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e,sys) from e


# ============================================================================
# n_iter size 계산 for randomized search
# ============================================================================

def calculate_param_space_size(param_grid):
    """
    파라미터 그리드의 총 조합 수 계산
    """
    if not param_grid:  # 빈 딕셔너리 체크
        return 0

    total = 1
    for value in param_grid.values():
        if isinstance(value, list):
            total *= len(value)

    return total


# ============================================================================
# 평가 지표 계산 함수
# ============================================================================

def calculate_metrics(y_true, y_pred):
    """
    회귀 모델의 모든 평가 지표 계산

    Parameters:
    - y_true: 실제값
    - y_pred: 예측값

    Returns:
    - metrics: 모든 평가 지표가 포함된 딕셔너리
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    # MAPE (평균절대백분율오차) - y_true에 0이 없을 때만
    try:
        mape = mean_absolute_percentage_error(y_true, y_pred)
    except:
        mape = None

    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape
    }

# ============================================================================
# 모델 평가 함수
# ============================================================================

def evaluate_models(X_train, y_train, X_test, y_test, models, param=None, search_type='random'):
    """
    모델 평가 함수 (모든 지표 포함)

    Parameters:
    - X_train, y_train: 훈련 데이터
    - X_test, y_test: 테스트 데이터
    - models: 모델 딕셔너리
    - param: 하이퍼파라미터 딕셔너리
    - search_type: 'grid' 또는 'random'

    Returns:
    - report: {모델명: {지표: 값}} 중첩 딕셔너리
    - best_models: {모델명: 최적화된 모델} 딕셔너리
    """
    try:
        report = {}
        best_models = {}

        for model_name, model in models.items():
            logging.info(f"{'=' * 40}")
            logging.info(f"{model_name} 처리 중...")
            logging.info('=' * 40)

            # 하이퍼파라미터가 있으면 튜닝
            if param and model_name in param and param[model_name]:
                logging.info(f"하이퍼파라미터 튜닝 중...")

                # 파라미터 공간 크기 계산
                param_space_size = calculate_param_space_size(param[model_name])
                logging.info(f"파라미터 조합: {param_space_size}개")

                if search_type == 'grid':
                    # GridSearchCV (작은 파라미터 공간)
                    logging.info("탐색 방식: GridSearchCV (전체 탐색)")
                    search = GridSearchCV(
                        estimator=model,
                        param_grid=param[model_name],
                        cv=5,
                        scoring='r2',
                        n_jobs=-1,
                        verbose=0
                    )
                else:  # random
                    # RandomizedSearchCV (큰 파라미터 공간)
                    n_iter = min(15, param_space_size)
                    logging.info(f"탐색 방식: RandomizedSearchCV ({n_iter})회 샘플")
                    search = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=param[model_name],
                        n_iter=n_iter,
                        cv=5,
                        scoring='r2',
                        random_state=42,
                        n_jobs=-1,
                        verbose=0
                    )

                search.fit(X_train, y_train)
                best_model = search.best_estimator_
                logging.info(f"최적 파라미터: {search.best_params_}")

            else:
                logging.info(f"기본 파라미터 사용")
                model.fit(X_train, y_train)
                best_model = model

            # 훈련 데이터 평가
            y_train_pred = best_model.predict(X_train)
            train_metrics = calculate_metrics(y_train, y_train_pred)

            # 테스트 데이터 평가
            y_test_pred = best_model.predict(X_test)
            test_metrics = calculate_metrics(y_test, y_test_pred)

            # 결과 저장
            report[model_name] = {
                'train': train_metrics,
                'test': test_metrics
            }
            best_models[model_name] = best_model

            # 상세 로깅
            logging.info(f"훈련 데이터 성능:")
            logging.info(f" • MAE:  {train_metrics['MAE']:.4f}")
            logging.info(f" • RMSE: {train_metrics['RMSE']:.4f}")
            logging.info(f" • R²:   {train_metrics['R2']:.4f}")

            logging.info(f"테스트 데이터 성능:")
            logging.info(f" • MAE:  {test_metrics['MAE']:.4f}")
            logging.info(f" • RMSE: {test_metrics['RMSE']:.4f}")
            logging.info(f" • R²:   {test_metrics['R2']:.4f}")

            # 과적합 체크
            overfit_gap = train_metrics['R2'] - test_metrics['R2']
            if overfit_gap > 0.15:
                logging.warning(f"과적합 감지 (R² 차이: {overfit_gap:.4f})")
            else:
                logging.info(f"과적합 없음 (R² 차이: {overfit_gap:.4f})")

        return report, best_models

    except Exception as e:
        raise CustomException(e,sys) from e


