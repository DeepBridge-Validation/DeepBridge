class Pruning(BaseEstimator, ClassifierMixin):
    def __init__(self, base_model, pruning_rate=0.5):
        """
        Construtor da classe.
        
        Args:
            base_model: Modelo base a ser podado.
            pruning_rate: Fração de pesos a serem removidos (ex: 0.5 = 50%).
        """
        self.base_model = base_model
        self.pruning_rate = pruning_rate

    def fit(self, X, y):
        """
        Treina o modelo base e aplica pruning aos seus pesos.
        """
        # Treinar o modelo base
        self.base_model_ = clone(self.base_model)
        self.base_model_.fit(X, y)
        
        # Aplicar pruning
        self._apply_pruning()
        return self

    def _apply_pruning(self):
        """
        Remove os pesos menores (por magnitude) do modelo.
        """
        coef = self.base_model_.coef_
        abs_coef = np.abs(coef)
        
        # Calcular o threshold para remover `pruning_rate`% dos pesos
        threshold = np.percentile(abs_coef, self.pruning_rate * 100)
        
        # Zerar pesos abaixo do threshold
        pruned_coef = np.where(abs_coef < threshold, 0, coef)
        self.base_model_.coef_ = pruned_coef

    def predict(self, X):
        """
        Faz previsões usando o modelo podado.
        """
        check_is_fitted(self, 'base_model_')
        return self.base_model_.predict(X)

    def predict_proba(self, X):
        """
        Retorna probabilidades (se o modelo base suportar).
        """
        check_is_fitted(self, 'base_model_')
        return self.base_model_.predict_proba(X)