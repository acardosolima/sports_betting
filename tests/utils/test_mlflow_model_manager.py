import pytest
from unittest.mock import MagicMock, patch
from src.utils.mlflow_model_manager import MLflowModelManager

class TestMLflowModelManager:
    
    @pytest.fixture
    def mock_mlflow(self):
        """
        Fixture to mock all MLflow module-level functions and the MlflowClient.
        Targeting the 'src' namespace to match project structure.
        """
        # Patching the logger and the MlflowClient class
        with patch('src.utils.mlflow_model_manager.custom_logger') as mock_log_class, \
             patch('src.utils.mlflow_model_manager.MlflowClient') as mock_client_class, \
             patch('mlflow.set_tracking_uri') as mock_uri, \
             patch('mlflow.set_experiment') as mock_exp, \
             patch('mlflow.create_experiment', return_value="exp_id_123"), \
             patch('mlflow.get_experiment_by_name') as mock_get_exp, \
             patch('mlflow.start_run') as mock_run, \
             patch('mlflow.register_model') as mock_reg:
            
            # Setup Logger mock
            mock_log_class.get_logger.return_value = MagicMock()
            
            # Setup MlflowClient mock instance
            mock_client_instance = mock_client_class.return_value
            
            # Setup start_run context manager (returns a run object with an info.run_id)
            mock_run_context = MagicMock()
            mock_run_context.info.run_id = "test_run_id"
            mock_run.return_value.__enter__.return_value = mock_run_context
            
            yield {
                "uri": mock_uri,
                "exp": mock_exp,
                "get_exp": mock_get_exp,
                "run": mock_run,
                "reg": mock_reg,
                "client": mock_client_instance
            }

    @pytest.fixture
    def manager(self, mock_mlflow):
        """Returns an instance of MLflowModelManager with mocked dependencies."""
        return MLflowModelManager(
            model_name="test_model",
            experiment_name="test_experiment"
        )

    # --- Initialization Tests ---

    def test_init_full_mode(self, mock_mlflow, manager):
        """Best Practice: Verify state and side effects during init."""
        assert manager.experiment_id == "exp_id_123"
        mock_mlflow["exp"].assert_called_with("test_experiment")

    def test_init_registry_only_mode(self, mock_mlflow):
        """Verify initialization without tracking/experiments."""
        manager = MLflowModelManager("model_a")
        assert manager.experiment_id is None
        mock_mlflow["exp"].assert_not_called()

    # --- Logging Tests ---

    @patch('mlflow.sklearn.log_model')
    @patch('mlflow.log_params')
    @patch('mlflow.log_metrics')
    def test_log_model_sklearn(self, mock_metrics, mock_params, mock_sklearn, manager):
        """Verify that logging calls the correct MLflow sub-modules."""
        params = {"alpha": 0.1}
        metrics = {"rmse": 0.5}
        
        run_id = manager.log_model(
            model=MagicMock(),
            params=params,
            metrics=metrics,
            model_type="sklearn"
        )
        
        assert run_id == "test_run_id"
        mock_params.assert_called_once_with(params)
        mock_metrics.assert_called_once_with(metrics)
        mock_sklearn.assert_called_once()

    def test_log_model_no_experiment_raises_error(self, mock_mlflow):
        """Ensure logic prevents logging when no experiment is set."""
        manager = MLflowModelManager("model_a") 
        with pytest.raises(ValueError, match="'Experiment_name' is required"):
            manager.log_model(model=MagicMock(), params={}, metrics={})

    # --- Registration & Alias Tests ---

    def test_register_model(self, manager, mock_mlflow):
        """Verify model registration returns the correct version and sets aliases."""
        mock_version = MagicMock()
        mock_version.version = "1"
        mock_mlflow["reg"].return_value = mock_version
        
        version = manager.register_model(run_id="run_123", alias="prod")
        
        assert version == "1"
        # Verify call to the mocked MlflowClient
        mock_mlflow["client"].set_registered_model_alias.assert_called()

    def test_load_model_by_alias(self, manager):
        """Verify URI formatting for alias-based loading."""
        with patch('mlflow.pyfunc.load_model') as mock_load:
            manager.load_model(alias="production")
            mock_load.assert_called_with("models:/test_model@production")

    def test_promote_to_production(self, manager, mock_mlflow):
        """Verify explicit alias promotion logic."""
        manager.promote_to_production(version="2")
        
        mock_mlflow["client"].set_registered_model_alias.assert_called_with(
            name="test_model", 
            alias="production", 
            version="2"
        )