# Copyright (C) 2026 Adriano Lima
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import warnings
from typing import Any

import mlflow
import mlflow.pyfunc
import mlflow.pytorch
import mlflow.sklearn
import mlflow.tensorflow
from mlflow.pyfunc import PyFuncModel
from mlflow.tracking import MlflowClient

from ssa.utils.logger import Logger as custom_logger


class MLflowModelManager:
    """
    Manages model lifecycle using MLFlow, including logging, registration,
    versioning, and deployment across different environments.

    This manager supports two modes:
    1. Registry-only mode: Load and manage existing models (no experiment_name needed)
    2. Full mode: Log new models + registry operations (requires experiment_name)

    Attributes:
        model_name: Name of the model in MLFlow Model Registry
        experiment_name: Name of the MLFlow experiment (optional, needed only for logging)
        client: MLFlow tracking client instance
        experiment_id: ID of the MLFlow experiment (None if registry-only mode)
    """

    def __init__(
        self,
        model_name: str,
        experiment_name: str | None = None,
        tracking_uri: str | None = None,
        artifact_location: str | None = None,
        log_level: str = "WARNING",
    ):
        """
        Initializes the MLFlow Model Manager.

        Args:
            model_name: Name to use in Model Registry. Required for all operations.
            experiment_name: Name of the MLFlow experiment. Optional - only required
                           if you plan to log new model runs. Can be omitted if you
                           only need to load/manage existing models.
            tracking_uri: URI of remote MLFlow tracking server. If None, uses local tracking.
            artifact_location: Custom location for storing artifacts (S3, GCS, etc).
            log_level: Logging verbosity level. One of: DEBUG, INFO, WARNING, ERROR, CRITICAL.
        """
        self.logger = custom_logger.get_logger(log_level=log_level, caller=self)

        self.experiment_name = experiment_name
        self.model_name = model_name
        self.client = MlflowClient(tracking_uri=tracking_uri)

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        if experiment_name:
            try:
                self.experiment_id = mlflow.create_experiment(
                    experiment_name, artifact_location=artifact_location
                )
                self.logger.debug(f"Created new experiment: {experiment_name}")
            except Exception:
                self.experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
                self.logger.debug(f"Using existing experiment: {experiment_name} (ID: {self.experiment_id})")

            mlflow.set_experiment(experiment_name)
            self.logger.info(f"MLFlow experiment configured: {experiment_name}")
        else:
            self.experiment_id = None
            self.logger.info("Initialized in registry-only mode (no experiment configured)")

    def log_model(
        self,
        model: Any,
        params: dict[str, Any],
        metrics: dict[str, float],
        model_type: str = "sklearn",
        use_autolog: bool = False,
        artifacts: dict[str, str] | None = None,
        tags: dict[str, str] | None = None,
        run_name: str | None = None,
        signature: Any | None = None,
        input_example: Any | None = None,
        pip_requirements: list[str] | None = None,
        code_paths: list[str] | None = None,
    ) -> str:
        """
        Logs a trained model along with its parameters and metrics to MLFlow.

        Args:
            model: Trained model object. Type depends on model_type parameter.
            params: Dictionary of hyperparameters used during training.
            metrics: Dictionary of evaluation metrics.
            model_type: Framework type. One of:
                       - "sklearn": scikit-learn models
                       - "pytorch": PyTorch models
                       - "tensorflow": TensorFlow/Keras models
                       - "pyfunc": Custom MLFlow PyFunc models
                       - "custom": Generic Python objects (saved as pickle)
            use_autolog: If True, uses MLFlow's autolog feature to automatically capture
                        model artifacts, parameters, and metrics. Only works with
                        sklearn, pytorch, and tensorflow. Ignored for pyfunc and custom.
            artifacts: Optional dictionary mapping artifact names to file paths.
            tags: Optional tags for categorizing the run.
            run_name: Optional name for the MLFlow run. If None, MLFlow generates one.
            signature: MLFlow model signature defining input/output schema.
            input_example: Sample input data for documentation and validation.
            pip_requirements: List of pip packages required to load the model.
            code_paths: List of Python files to include with the model.

        Returns:
            String containing the MLFlow run ID for this logged model.

        Raises:
            ValueError: If experiment_name was not provided during initialization,
                       or if model_type is not supported.
        """
        if not self.experiment_name:
            raise ValueError("'Experiment_name' is required to log models")

        # Enable autolog if requested and supported
        if use_autolog:
            if model_type == "sklearn":
                mlflow.sklearn.autolog()
                self.logger.debug("Enabled sklearn autolog")
            elif model_type == "pytorch":
                mlflow.pytorch.autolog()
                self.logger.debug("Enabled pytorch autolog")
            elif model_type == "tensorflow":
                mlflow.tensorflow.autolog()
                self.logger.debug("Enabled tensorflow autolog")
            else:
                warnings.warn(
                    f"Autolog not supported for model_type '{model_type}'.Falling back to manual logging",
                    UserWarning,
                    stacklevel=2,
                )

        with mlflow.start_run(run_name=run_name) as run:
            if params:
                mlflow.log_params(params)
            if metrics:
                mlflow.log_metrics(metrics)

            if tags:
                mlflow.set_tags(tags)

            if artifacts:
                for name, path in artifacts.items():
                    mlflow.log_artifact(path, name=name)

            if model_type == "sklearn":
                mlflow.sklearn.log_model(
                    model,
                    name="model",
                    signature=signature,
                    input_example=input_example,
                    pip_requirements=pip_requirements,
                )

            elif model_type == "pytorch":
                mlflow.pytorch.log_model(
                    model,
                    name="model",
                    signature=signature,
                    input_example=input_example,
                    pip_requirements=pip_requirements,
                    code_paths=code_paths,
                )

            elif model_type == "tensorflow":
                mlflow.tensorflow.log_model(
                    model,
                    name="model",
                    signature=signature,
                    input_example=input_example,
                    pip_requirements=pip_requirements,
                )

            elif model_type == "pyfunc":
                mlflow.pyfunc.log_model(
                    name="model",
                    python_model=model,
                    signature=signature,
                    input_example=input_example,
                    pip_requirements=pip_requirements,
                    code_path=code_paths,
                )

            elif model_type == "custom":
                import pickle
                import tempfile
                from pathlib import Path

                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_path = Path(tmp_dir) / "model.pkl"
                    with open(tmp_path, "wb") as f:
                        pickle.dump(model, f)
                    mlflow.log_artifact(str(tmp_path), artifact_path="model")

            else:
                raise ValueError(
                    f"Unsupported model_type: {model_type}. "
                    "Supported types: sklearn, pytorch, tensorflow, pyfunc or custom"
                )

            run_id = run.info.run_id
            self.logger.info(f"Model logged successfully (run_id: {run_id})")

            return run_id

    def register_model(
        self,
        run_id: str,
        alias: str | None = None,
        description: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> str:
        """
        Registers a logged model in the MLFlow Model Registry.

        Once registered, the model can be loaded by version number or alias,
        enabling proper model versioning and deployment workflows.

        Args:
            run_id: The MLFlow run ID where the model was logged.
            alias: Optional alias to assign to this version.
            description: Optional human-readable description of this model version.
            tags: Optional key-value pairs for categorizing this version.

        Returns:
            String containing the registered version number (e.g., "1", "2", "3").
        """
        model_uri = f"runs:/{run_id}/model"

        model_version = mlflow.register_model(model_uri, self.model_name)
        version = model_version.version

        self.logger.info(f"Registered model: {self.model_name} version {version}")

        if description:
            self.client.update_model_version(name=self.model_name, version=version, description=description)
            self.logger.debug(f"Set description for version {version}")

        if tags:
            for key, value in tags.items():
                self.client.set_model_version_tag(name=self.model_name, version=version, key=key, value=value)
            self.logger.debug(f"Set {len(tags)} tags for version {version}")

        if alias:
            self.set_alias(version=version, alias=alias)

        return version

    def load_model(
        self,
        version: str | None = None,
        alias: str | None = None,
        run_id: str | None = None,
    ) -> PyFuncModel:
        """
        Loads a model from the MLFlow Model Registry.

        You must specify exactly one of: version, alias, or run_id.

        Args:
            version: Specific version number to load (e.g., "1", "5", "12").
            alias: Load model with this alias (e.g., "production", "staging").
            run_id: Load model from a specific MLFlow run ID.

        Returns:
            Loaded model as an MLFlow PyFuncModel wrapper. Use .predict() method
            for inference. The wrapper handles deserialization and provides a
            consistent interface regardless of the original model type.

        Raises:
            ValueError: If none or multiple identifiers are provided.
        """
        if sum([version is not None, alias is not None, run_id is not None]) != 1:
            raise ValueError("Must specify exactly one of: version, alias, or run_id")

        if run_id:
            model_uri = f"runs:/{run_id}/model"
        elif version:
            model_uri = f"models:/{self.model_name}/{version}"
        elif alias:
            model_uri = f"models:/{self.model_name}@{alias}"

        self.logger.debug(f"Loading model from: {model_uri}")

        model = mlflow.pyfunc.load_model(model_uri)
        self.logger.info("Model loaded successfully")

        return model

    def set_alias(self, version: str, alias: str) -> None:
        """
        Sets or updates an alias for a specific model version.

        If the alias already exists on a different version, it will be moved
        to the specified version. A single version can have multiple aliases.

        Args:
            version: Version number to assign the alias to.
            alias: Alias name to set
        """
        self.client.set_registered_model_alias(name=self.model_name, alias=alias, version=version)
        self.logger.info(f"Set alias '{alias}' to version {version}")

    def delete_alias(self, alias: str) -> None:
        """
        Removes an alias from the model.

        The underlying model version is not deleted, only the alias is removed.

        Args:
            alias: Name of the alias to remove.
        """
        self.client.delete_registered_model_alias(name=self.model_name, alias=alias)
        self.logger.info(f"Deleted alias: {alias}")

    def list_versions(self) -> list[dict[str, Any]]:
        """
        Lists all registered versions of the model with their metadata.

        Returns:
            List of dictionaries, each containing:
            - version: Version number
            - run_id: MLFlow run ID where this version was logged
            - status: Current status (READY, PENDING_REGISTRATION, FAILED_REGISTRATION)
            - creation_timestamp: Unix timestamp of when version was created
            - aliases: List of aliases assigned to this version
        """
        versions = self.client.search_model_versions(f"name='{self.model_name}'")

        result = [
            {
                "version": v.version,
                "run_id": v.run_id,
                "status": v.status,
                "creation_timestamp": v.creation_timestamp,
                "aliases": v.aliases if hasattr(v, "aliases") else [],
            }
            for v in versions
        ]

        self.logger.debug(f"Found {len(result)} versions")
        return result

    def get_model_by_alias(self, alias: str) -> dict[str, Any] | None:
        """
        Retrieves metadata for the model version with a specific alias.

        Args:
            alias: Alias name to look up.

        Returns:
            Dictionary with version metadata if found, None if alias doesn't exist.
            Dictionary contains: version, run_id, status, aliases
        """
        try:
            v = self.client.get_model_version_by_alias(self.model_name, alias)
            return {
                "version": v.version,
                "run_id": v.run_id,
                "status": v.status,
                "aliases": v.aliases if hasattr(v, "aliases") else [],
            }

        except Exception as e:
            self.logger.error(f"Alias '{alias}' not found: {e}")
            return None

    def promote_to_production(self, version: str) -> None:
        """
        Promotes a model version to production by setting the 'production' alias.

        This is a convenience method equivalent to set_alias(version, "production").
        If another version currently has the 'production' alias, it will be moved.

        Args:
            version: Version number to promote.
        """
        self.set_alias(version=version, alias="production")
        self.logger.info(f"Promoted version {version} to production")

    def promote_to_staging(self, version: str) -> None:
        """
        Promotes a model version to staging by setting the 'staging' alias.

        This is a convenience method equivalent to set_alias(version, "staging").

        Args:
            version: Version number to promote.
        """
        self.set_alias(version=version, alias="staging")
        self.logger.info(f"Promoted version {version} to staging")

    def delete_version(self, version: str) -> None:
        """
        Permanently deletes a model version from the registry.

        Warning: This action cannot be undone. The model artifacts and all
        associated metadata will be removed.

        Args:
            version: Version number to delete.
        """
        self.client.delete_model_version(name=self.model_name, version=version)
        self.logger.info(f"Deleted version {version}")

    def update_description(self, version: str, description: str) -> None:
        """
        Updates the description text for a model version.

        Args:
            version: Version number to update.
            description: New description text.
        """
        self.client.update_model_version(name=self.model_name, version=version, description=description)
        self.logger.info(f"Updated description for version {version}")
