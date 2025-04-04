import typing as t
import pandas as pd
from sklearn.model_selection import train_test_split

class DataManager:
    """
    Responsible for managing and preparing data for experimentation.
    """
    
    def __init__(self, dataset, test_size, random_state):
        self.dataset = dataset
        self.test_size = test_size
        self.random_state = random_state
        
        # Initialize data attributes
        self.X_train = None
        self.X_test = None 
        self.y_train = None
        self.y_test = None
        self.prob_train = None
        self.prob_test = None
    
    def prepare_data(self) -> None:
        """
        Prepare the data by performing train-test split on features and target.
        """
        X = self.dataset.X
        y = self.dataset.target
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        # If we have original probabilities, split them too
        if hasattr(self.dataset, 'original_prob') and self.dataset.original_prob is not None:
            prob_train_idx = self.X_train.index
            prob_test_idx = self.X_test.index
            
            self.prob_train = self.dataset.original_prob.loc[prob_train_idx]
            self.prob_test = self.dataset.original_prob.loc[prob_test_idx]
        else:
            self.prob_train = None
            self.prob_test = None
    
    def get_dataset_split(self, dataset: str = 'train') -> tuple:
        """
        Get the features and target for specified dataset split.
        """
        if dataset == 'train':
            return self.X_train, self.y_train, self.prob_train
        elif dataset == 'test':
            return self.X_test, self.y_test, self.prob_test
        else:
            raise ValueError("dataset must be either 'train' or 'test'")
    
    def get_binary_predictions(self, probabilities: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
        """
        Convert probability predictions to binary predictions using a threshold.
        """
        # If we have multiple columns, assume the last one is for class 1
        prob_values = probabilities.iloc[:, -1] if len(probabilities.columns) > 1 else probabilities.iloc[:, 0]
        return (prob_values >= threshold).astype(int)
