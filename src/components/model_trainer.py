# mlproject/src/components/model_trainer.py

import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    RandomForestRegressor,
)

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

# ============================================================================
# 하이퍼파라미터 그리드
# ============================================================================

PARAM_GRID = {
    "Linear Regression": {},  # 튜닝할 파라미터 없음

    "Lasso": {
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    },

    "Ridge": {
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    },

    "K-Neighbors Regressor": {
        'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    },

    "Decision Tree": {
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['squared_error', 'absolute_error']
    },

    "Random Forest Regressor": {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    },

    "XGBRegressor": {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    },

    "CatBoosting Regressor": {
        'iterations': [100, 200],
        'depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'l2_leaf_reg': [1, 3, 5]
    },

    "AdaBoost Regressor": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0],
        'loss': ['linear', 'square', 'exponential']
    }
}

# ============================================================================
# ModelTrainerConfig & ModelTrainer
# ============================================================================

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """
        모델 훈련 및 최적화
        """
        try:
            logging.info("=" * 40)
            logging.info("데이터 분리 시작")
            logging.info("=" * 40)

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(max_iter=10000),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Random Forest Regressor": RandomForestRegressor(random_state=42),
                "XGBRegressor": XGBRegressor(verbosity=0, n_jobs=-1, random_state=42),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False, random_state=42),
                "AdaBoost Regressor": AdaBoostRegressor(random_state=42)
            }

            logging.info("=" * 40)
            logging.info("모델 평가 및 하이퍼파라미터 튜닝 시작")
            logging.info("=" * 40)

            model_report, best_models = evaluate_models(
                X_train, y_train, X_test, y_test,
                models,
                param=PARAM_GRID,
                    search_type='random'
            )

            # 결과 요약 (테스트 R² 기준 정렬)
            logging.info("=" * 40)
            logging.info("최종 결과 요약 (Test R² 기준)")
            logging.info("=" * 40)

            sorted_models = sorted(
                model_report.items(),
                key=lambda x: x[1]['test']['R2'],
                reverse=True
            )

            for rank, (name, metrics) in enumerate(sorted_models, 1):
                test_r2 = metrics['test']['R2']
                test_rmse = metrics['test']['RMSE']
                test_mae = metrics['test']['MAE']

                logging.info(f"{rank}위. {name}")
                logging.info(f"   ├─ Test R²:   {test_r2:.4f}")
                logging.info(f"   ├─ Test RMSE: {test_rmse:.4f}")
                logging.info(f"   └─ Test MAE:  {test_mae:.4f}")

            # 최고 성능 모델 선택 (Test R² 기준)
            best_model_name = sorted_models[0][0]
            best_metrics = sorted_models[0][1]
            best_model = best_models[best_model_name]
            best_model_score = best_metrics['test']['R2']

            if best_model_score < 0.6:
                raise CustomException("최고 성능 모델 없음")

            logging.info("=" * 40)
            logging.info(f"최고 성능 모델: {best_model_name}")
            logging.info(f"  • Test R²:   {best_metrics['test']['R2']:.4f}")
            logging.info(f"  • Test RMSE: {best_metrics['test']['RMSE']:.4f}")
            logging.info(f"  • Test MAE:  {best_metrics['test']['MAE']:.4f}")
            logging.info("=" * 40)

            # 모델 저장
            save_object(self.model_trainer_config.trained_model_file_path, best_model)
            logging.info(f"모델 저장 완료: {self.model_trainer_config.trained_model_file_path}\n")

            return best_model, best_model_score

        except Exception as e:
            raise CustomException(e, sys)


