from typing import Callable

from numpy.random import randint
from pandas import DataFrame
from torch import load

from tred.datasets import BaseDataset
from tred.domain_samplers import ClusterSampler
from tred.methods import BaseMethod
from tred.transformations import CSVTransformation
from tred.utils import csv_distance, load_class, metric_wrapper, ts_to_date


class EvaluationOverTime2:
    def __init__(self, config: dict) -> None:
        """Initialise the experiment.
        During this process, all datasets and methods are instantiated.
        The datasets are loaded into memory, but the models are not yet trained.

        Parameters
        ----------
        config: dict
            A dictionary containing all the required datasets and methods configurations.
        """
        self.name: str = config["experiment_name"]
        self.config = config

        self.datasets: dict[str, BaseDataset] = {}
        self._load_data()

        self.methods: dict[str, BaseMethod] = {}
        for name_m in config["methods"]:
            method_class = load_class(config["methods"][name_m]["method_class"])
            self.methods[name_m] = method_class(
                **config["methods"][name_m]["method_confs"],
                **config["methods_common_configs"],
            )

    def evaluate_sequential_updates(self, verbose: bool = False) -> DataFrame:
        confs = self.config["evaluation_over_time"]
        metrics = {
            name: metric_wrapper(info["metric"], **info.get("config", {}))
            for name, info in confs["metrics"].items()
        }
        target_name = confs["target_domain"]
        results = []
        was_trained = {name_m: False for name_m in self.methods.keys()}

        for time_i in range(confs["splits"]["n_evaluations"]):
            if verbose:
                print(f"########### Split {time_i:2} ###########")
                print("--------------------------------")

            # Filter datasets
            datasets_Lt, datasets_Lv, datasets_U = self._filter_datasets_to_train(
                target_name,
                source_start_time=confs["splits"]["source_start_time"],
                target_start_time=confs["splits"]["target_start_time"],
                end_time=(
                    confs["splits"]["target_start_time"]
                    + time_i * confs["splits"]["evaluation_step"]
                ),
                label_delay=confs["splits"]["label_delay"],
            )
            if verbose:
                self._log_datasets_info(datasets_Lt, "D_Lt")
                self._log_datasets_info(datasets_Lv, "D_Lv")
                self._log_datasets_info(datasets_U, "D_U")
                print("--------------------------------")

            # Train methods
            self._train_methods(
                datasets_Lt,
                datasets_Lv,
                datasets_U,
                target_name,
                verbose,
                was_trained,
            )
            if verbose:
                print("--------------------------------")

            # Process target dataset for evaluation
            self.datasets[target_name].select_on_condition(
                start_time=(
                    confs["splits"]["target_start_time"]
                    + time_i * confs["splits"]["evaluation_step"]
                ),
                end_time=(
                    confs["splits"]["target_start_time"]
                    + (time_i + 1) * confs["splits"]["evaluation_step"]
                ),
                inplace=True,
            )
            if verbose:
                self._log_datasets_info(
                    {target_name: self.datasets[target_name]}, "Target dataset"
                )
                print("--------------------------------")

            # Evaluate methods
            results += self._evaluate_methods(
                metrics, target_name, time_i, verbose, was_trained
            )
            if verbose:
                print("--------------------------------")
        return DataFrame(results)

    def preprocess_data(self, verbose: bool = False) -> None:
        confs = self.config["domain_sampler"]
        if confs:
            sampler_class = load_class(confs["sampler_class"])
            distance_fn = load_class(confs["sampler_confs"].pop("distance_function"))
            sampler = sampler_class(
                distance_function=distance_fn, **confs["sampler_confs"]
            )
            datasets = sampler.sample(self.datasets, **confs["sample_confs"])
        else:
            datasets = self.datasets

        confs = self.config["transformations"]
        if confs:
            csvt = CSVTransformation(confs)
            csvt.apply_transformations(datasets)  # type: ignore[arg-type]

        self.datasets = datasets  # type: ignore[assignment]

    def _evaluate_methods(
        self,
        metrics: dict[str, Callable],
        target_name: str,
        time_i: int,
        verbose: bool,
        was_trained: dict[str, bool],
    ):
        """Evaluate all methods on the target dataset."""
        labels = self.datasets[target_name].get_y()
        timestamps = self.datasets[target_name].get_t()
        results = []
        for name_m, method in self.methods.items():
            if not was_trained[name_m]:
                continue
            if verbose:
                print(f"Evaluating {name_m}.")
            preds = method.predict(self.datasets[target_name], target_name)
            result = {
                "method": name_m,
                "target": target_name,
                "split_idx": time_i,
                "time_range": (min(timestamps), max(timestamps)),
            }
            for metric_name, metric in metrics.items():
                result[metric_name] = metric(labels, preds)
            results.append(result)
        return results

    def _filter_datasets_to_train(
        self,
        target_name: str,
        source_start_time: int,
        target_start_time: int,
        end_time: int,
        label_delay: int,
        validation_ratio: float = 0.3,
    ) -> tuple[dict, dict, dict]:
        """Filter datasets for the current time split."""
        datasets_Lt, datasets_Lv, datasets_U = {}, {}, {}
        for name_d, dataset in self.datasets.items():
            start_time = (
                target_start_time if name_d == target_name else source_start_time
            )
            dataset.select_on_condition(start_time, end_time, inplace=True)
            unlabeled_start = max(start_time, end_time - label_delay)
            L, U = dataset.split(split_time=unlabeled_start)
            U.label_available = False
            if len(L) > 0:
                Lt, Lv = L.split(
                    split_time=start_time
                    + (1 - validation_ratio) * (unlabeled_start - start_time)
                )
                datasets_Lt[name_d] = Lt
                datasets_Lv[name_d] = Lv
            if len(U) > 0:
                datasets_U[name_d] = U
        return datasets_Lt, datasets_Lv, datasets_U

    def _load_data(self):
        confs = self.config["datasets"]
        self.datasets = {}
        for name_d in confs:
            dataset_class = load_class(confs[name_d]["dataset_class"])
            self.datasets[name_d] = dataset_class(
                path=confs[name_d]["path"],
                **self.config["schema"],
            )
            self.datasets[name_d].load_data()
            self._log_datasets_info({name_d: self.datasets[name_d]}, name_d)

    def _load_pretrained_models(self):
        conf = self.config["pretrained"]
        for name_m in conf:
            for component_name in conf[name_m]:
                component = getattr(self.methods[name_m], component_name)
                state_dict = load(
                    conf[name_m][component_name],
                    map_location=self.methods[name_m].device,
                    weights_only=True,
                )
                component.load_state_dict(state_dict)

    def _log_datasets_info(
        self, datasets: dict[str, BaseDataset], var_name: str
    ) -> None:
        """Log information about datasets."""
        print(f"{var_name}:")
        for name_d, dataset in datasets.items():
            timestamps = dataset.get_t()
            start, end = min(timestamps), max(timestamps)
            y = dataset.get_y()
            print(
                f"name: {name_d:10s}; "
                f"len: {len(dataset):9d}; "
                f"positives: {0 if y is None else int(y.sum()):7d}; "
                f"from {ts_to_date(start)} to {ts_to_date(end)}."
            )

    def _train_methods(
        self,
        datasets_Lt: dict[str, BaseDataset],
        datasets_Lv: dict[str, BaseDataset],
        datasets_U: dict[str, BaseDataset],
        target_name: str,
        verbose: bool,
        was_trained: dict[str, bool],
    ):
        """Train all methods for the current time split."""
        for name_m, method in self.methods.items():
            if method.requires_target_data and target_name not in datasets_U:
                if verbose:
                    print(f"Skipping {name_m:13s} for lack of target data.")
                continue
            if method.requires_target_labels and target_name not in datasets_Lt:
                if verbose:
                    print(f"Skipping {name_m:13s} for lack of target labels.")
                continue
            if verbose:
                print(f"Fitting {name_m}.")
            method.fit(
                train_L=datasets_Lt,
                reset_model=False,
                target_name=target_name,
                train_U=datasets_U,
                validation=datasets_Lv,
            )
            was_trained[name_m] = True


def _get_n_experiment_starts(conf: dict) -> int:
    dataset_units = (conf["dataset_end"] - conf["dataset_start"]) // conf["t_unit"]
    experiment_units = (
        conf["minimum_source_units"] + conf["evaluation_units"] * conf["n_evaluations"]
    )
    return dataset_units - experiment_units + 1


def main():
    parser = argparse.ArgumentParser(description="Run the CompletePipeline experiment.")
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
        "-r",
        "--results_file",
        type=str,
        required=True,
        help="Prefix of the name of the results csv file.",
    )
    parser.add_argument(
        "--trial_start", type=int, required=True, help="Starting trial id (inclusive)."
    )
    parser.add_argument(
        "--trial_end", type=int, required=True, help="Ending trial id (exclusive)."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose mode."
    )
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
    config["methods_common_configs"]["use_device"] = args.device

    n_experiment_starts = _get_n_experiment_starts(
        config["evaluation_over_time"]["splits"]
    )
    if n_experiment_starts <= 0:
        raise ValueError(
            f"Number of valid experiment starts is non-positive ({n_experiment_starts})"
        )
    config["evaluation_over_time"]["splits"]["label_delay"] = (
        config["evaluation_over_time"]["splits"]["label_delay_units"]
        * config["evaluation_over_time"]["splits"]["t_unit"]
    )

    for i in range(args.trial_start, args.trial_end):
        c = config["evaluation_over_time"]["splits"]
        rand_start = randint(n_experiment_starts)
        c["source_start_time"] = c["dataset_start"] + rand_start * c["t_unit"]
        c["target_start_time"] = (
            c["source_start_time"] + c["minimum_source_units"] * c["t_unit"]
        )
        c["evaluation_step"] = c["evaluation_units"] * c["t_unit"]

        pipeline = EvaluationOverTime2(config)
        if config["pretrained"]:
            pipeline._load_pretrained_models()

        pipeline.preprocess_data(verbose=args.verbose)

        results = pipeline.evaluate_sequential_updates(verbose=args.verbose)
        results.to_csv(f"results/{args.results_file}_{i:03d}.csv", index=False)


if __name__ == "__main__":
    import argparse

    import yaml

    main()
