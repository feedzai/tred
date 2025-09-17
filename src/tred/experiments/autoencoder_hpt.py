import logging
from copy import deepcopy
from typing import Any

import optuna
from torch import save

from tred.datasets import BaseDataset
from tred.utils import load_class, ts_to_date

# Create a logger for this module
logger = logging.getLogger(__name__)


class Autoencoder_HPT:
    """
    Class for running hyperparameter tuning on multiple methods and datasets
    using Optuna for optimization.
    """

    def __init__(self, configs: dict):
        logger.info("Initializing HyperParameterTuning class.")
        self.splits = configs["evaluation"]["splits"]

        # Load datasets
        logger.info("Loading datasets...")
        datasets: dict[str, BaseDataset] = self._load_datasets(
            configs["datasets"], configs["schema"]
        )
        logger.info(f"Loaded {len(datasets)} dataset(s): {list(datasets.keys())}")
        self.train: dict[str, BaseDataset] = {}
        self.validation: dict[str, BaseDataset] = {}
        for name_d, dataset in datasets.items():
            self.train[name_d], self.validation[name_d] = dataset.split(
                split_time=self.splits["val_start"],
            )

        # Setup methods
        self.method_classes: dict[str, Any] = {}
        self.method_configs: dict[str, Any] = {}
        self.method_studies: dict[str, optuna.study.Study] = {}

        logger.info("Setting up methods and their Optuna studies...")
        for m_name, m_config in configs["methods"].items():
            # Load the method class
            method_class_path = m_config.pop("method_class")
            self.method_classes[m_name] = load_class(method_class_path)
            logger.debug(f"Method '{m_name}' uses class: {method_class_path}")

            # Extract method config
            self.method_configs[m_name] = m_config.get("method_confs", {}).copy()
            logger.debug(
                f"Method '{m_name}' configuration: {self.method_configs[m_name]}"
            )

            # Setup the Optuna study
            self.method_studies[m_name] = self._setup_optuna(configs["optuna"], m_name)

    def run_trials(self, n_trials: int) -> dict[str, optuna.study.Study]:
        """
        Runs Optuna optimization trials for each method.

        Parameters
        ----------
        n_trials: int
            Number of trials for each method.

        Returns
        -------
        dict[str, optuna.study.Study]
            The updated studies for each method.
        """
        for m_name, m_study in self.method_studies.items():
            logger.info(
                f"Launching optimization for method '{m_name}' with {n_trials} trials."
            )
            m_study.optimize(
                lambda trial: self._objective(m_name, trial),
                n_trials=n_trials,
            )
        logger.info("All trials completed.")
        return self.method_studies

    def _load_datasets(
        self,
        dataset_configs: dict[str, Any],
        schema_configs: dict[str, Any],
    ) -> dict[str, BaseDataset]:
        """
        Loads each dataset from its configuration and performs basic time filtering.

        Parameters
        ----------
        dataset_configs: dict[str, Any]
            The configurations for loading each dataset.
        schema_configs: dict[str, Any]
            The dataset schema details.

        Returns
        -------
        dict[str, BaseDataset]
            Mapping from dataset name to dataset object.
        """
        datasets: dict[str, BaseDataset] = {}
        for name_d, config_d in dataset_configs.items():
            logger.debug(
                f"Initializing dataset '{name_d}' from class '{config_d['dataset_class']}' at path '{config_d['path']}'."
            )

            dataset_class = load_class(config_d["dataset_class"])
            datasets[name_d] = dataset_class(path=config_d["path"], **schema_configs)
            datasets[name_d].load_data()
            datasets[name_d].select_on_condition(
                self.splits["train_start"], self.splits["val_end"]
            )

            start_t, end_t = datasets[name_d].time_range
            logger.debug(
                f"Loaded dataset '{name_d}' from {ts_to_date(start_t)} to {ts_to_date(end_t)}."
            )
        return datasets

    def _objective(self, method_name: str, trial: optuna.trial.Trial) -> float:
        """
        Optuna objective function that trains and evaluates the given method on all datasets.

        Parameters
        ----------
        method_name: str
            Name of the method to optimize.
        trial: optuna.trial.Trial
            The current trial object.

        Returns
        -------
        float
            Mean value of the optimization metric across all datasets as target.
        """
        logger.info(f"Starting trial {trial.number} for method '{method_name}'.")

        # Instantiate the method with suggested hyperparameters
        m_class = self.method_classes[method_name]
        m_configs = self._recurrent_suggest(self.method_configs[method_name], trial)
        embedding_dim = m_configs.pop("embedding_dim")
        layer_dims = m_configs["E_confs"]["layer_dims"]
        m_configs["E_confs"]["layer_dims"] = list(map(int, layer_dims.split("_"))) + [
            embedding_dim
        ]
        m_configs["D_confs"] = deepcopy(m_configs["E_confs"])
        m_configs["D_confs"]["cardinalities"] = []
        m_configs["D_confs"]["input_dim_num"] = embedding_dim
        m_configs["D_confs"]["layer_dims"] = list(map(int, layer_dims.split("_")))[::-1]
        m_configs["D_confs"]["layer_dims"] += [
            m_configs["E_confs"]["input_dim_num"]
            + sum(m_configs["E_confs"]["cardinalities"])
        ]
        method = m_class(**m_configs)

        # Fit and predict
        method.fit(self.train, reset_model=True, validation=self.validation)
        score = method.evaluate(self.validation)
        trial.set_user_attr("D_confs", str(m_configs["D_confs"]))
        trial.set_user_attr("E_confs", str(m_configs["E_confs"]))
        save(method.D.state_dict(), f"model_states/AE_{trial._trial_id:04d}_D.pt")
        save(method.E.state_dict(), f"model_states/AE_{trial._trial_id:04d}_E.pt")

        logger.info(
            f"Method '{method_name}', trial {trial.number} -> " f"Score: {score:.6f}"
        )
        return score

    def _recurrent_suggest(
        self,
        confs: dict[str, Any],
        trial: optuna.trial.Trial,
        path: list[str] = [],
    ) -> dict[str, Any]:
        """
        Recursively parses a config dictionary to suggest hyperparameters
        from Optuna if `optuna_type` is specified, or use fixed values otherwise.

        Parameters
        ----------
        confs: dict[str, Any]
            A dictionary possibly containing nested hyperparameters.
        trial: optuna.trial.Trial
            The current Optuna trial.
        path: list[str], default=[]
            The nested path until current dictionary of hyperparameters.

        Returns
        -------
        dict[str, Any]
            A dictionary with selected hyperparameters.
        """
        hps: dict[str, Any] = {}
        for k, v in confs.items():
            if isinstance(v, dict):
                new_path = path + [k]

                # Tunable hyperparameter
                if "optuna_type" in v:
                    if v["optuna_type"] == "categorical":
                        hps[k] = trial.suggest_categorical(
                            name=".".join(new_path), choices=v["choices"]
                        )
                    elif v["optuna_type"] == "float":
                        hps[k] = trial.suggest_float(
                            name=".".join(new_path),
                            low=v["low"],
                            high=v["high"],
                            step=v.get("step", None),
                            log=v.get("log", False),
                        )
                    elif v["optuna_type"] == "int":
                        hps[k] = trial.suggest_int(
                            name=".".join(new_path),
                            low=v["low"],
                            high=v["high"],
                            step=v.get("step", 1),
                            log=v.get("log", False),
                        )
                    else:
                        raise ValueError(
                            f"Optuna type should be `categorical`, `float` or `int`. "
                            f"Found: {v['optuna_type']} for config '{'.'.join(new_path)}'."
                        )

                # Nested config
                else:
                    hps[k] = self._recurrent_suggest(v, trial, new_path)

            # Fixed hyperparameter
            else:
                hps[k] = v
        return hps

    def _setup_optuna(
        self, configs: dict[str, Any], method_name: str
    ) -> optuna.study.Study:
        """
        Initializes (or loads existing) an Optuna study for a given method.

        Parameters
        ----------
        configs: dict[str, Any]
            Optuna configurations (db_name, tpe_startup_trials, etc.).
        method_name: str
            Name of the method.

        Returns
        -------
        optuna.study.Study
            The Optuna study for that method.
        """
        storage = optuna.storages.RDBStorage(
            url=f"sqlite:///{configs['db_name']}_{method_name}.db"
        )
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=configs["tpe_startup_trials"]
        )

        return optuna.create_study(
            storage=storage,
            sampler=sampler,
            study_name=configs["study_name"],
            direction=configs["direction"],
            load_if_exists=True,
        )


def main():
    parser = argparse.ArgumentParser(description="Run the Autoencoder_HPT experiment.")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        required=True,
        help="Name of device (e.g. 'cpu' or 'cuda:0').",
    )
    parser.add_argument(
        "-n", "--n_trials", type=int, required=True, help="Number of trials to run."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose mode."
    )
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
    config["methods"]["autoencoder"]["method_confs"]["use_device"] = args.device

    hpt = Autoencoder_HPT(config)
    hpt.run_trials(n_trials=args.n_trials)


if __name__ == "__main__":
    import argparse

    import yaml

    main()
