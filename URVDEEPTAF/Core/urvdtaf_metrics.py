import numpy as np
import sklearn.metrics as m
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from numba import njit
from typing import Any, Union, Tuple

@njit
def c_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate concordance index.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Concordance index
    """
    summ = 0
    pair = 0

    for i in range(1, len(y_true)):
        for j in range(0, i):
            pair += 1
            if y_true[i] > y_true[j]:
                summ += 1 * (y_pred[i] > y_pred[j]) + 0.5 * (y_pred[i] == y_pred[j])
            elif y_true[i] < y_true[j]:
                summ += 1 * (y_pred[i] < y_pred[j]) + 0.5 * (y_pred[i] == y_pred[j])
            else:
                pair -= 1

    if pair != 0:
        return summ / pair
    else:
        return 0

def RMSE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Square Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        RMSE value
    """
    return np.sqrt(m.mean_squared_error(y_true, y_pred))

def MAE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MAE value
    """
    return m.mean_absolute_error(y_true, y_pred)

def CORR(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Pearson correlation coefficient.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Correlation coefficient
    """
    return pearsonr(y_true, y_pred)[0]

def SD(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Standard Deviation in linear regression.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Standard deviation
    """
    y_pred = y_pred.reshape((-1, 1))
    lr = LinearRegression().fit(y_pred, y_true)
    y_ = lr.predict(y_pred)
    return np.sqrt(np.square(y_true - y_).sum() / (len(y_pred) - 1))


def MSE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Squared Error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MSE value
    """
    return np.mean((y_true - y_pred) ** 2)

def R2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate coefficient of determination (R²).
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        R² score
    """
    return r2_score(y_true, y_pred)

