# __Supervised Dynamic Large Language Model Library__

## __Abstract__

Framework for __training supervised dynamic language models__ that implements the state-of-the-art methods for training large language models.

## __Framework Structure__

For unified training and evaluation and easy implementation of new language methods we define in our framework two basic concepts, _Model_ and _Model Trainer_.

__Model__. This concept is implemented as abstract class ``AModel`` in [supervisedllm.models]. The UML diagram of the class can be find below.

![AModel](reports/figures/AModel.png)

Every dynamic language method implemented in this framework must be a child class from [supervisedllm.models.AModel]. In order one to use all feature (training, logging, ...) of the `supervisedllm` framework one must implement the new models as sub-class of ``AModel`` class. This abstract class defines four abstract methods that are used for the training and evaluation that needed to be implemented:

  1. ``forward()``: this methods implements the infirence of the model on a data point,
  2. ``train_step()``: the procedure executed during one training step
  3. ``validate_step``: the procedure executed during one validation step
  4. ``loss()``: definition of the loss for each specific model that will be used for training.

Example implementation of the Faster R-CNN model can be found in [supervisedllm.models.baseline_models.SequentialClassifier]

__Model Trainer__.  This class is used for training models implemented in this framework. The class is handeling the model trainin, logging and booking. In order this class to be used the models must be child class from the [supervisedllm.models.AModel]. This means that the model must implement four abstract functions from the parent class (see above).

## __Installation__

In order to set up the necessary environment:

### __Virtualenv__

1. Install [virtualenv] and [virtualenvwrapper].
2. Create virtualenviroment for the project:

    ```bash
    mkvirtualenv supervised-llm
    ```

3. Install the project in edit mode:

    ```bash
    python setup.py develop
    ```

Optional and needed only once after `git clone`:

1. install several [pre-commit] git hooks with:

   ```bash
   pre-commit install
   # You might also want to run `pre-commit autoupdate`
   ```

   and checkout the configuration under `.pre-commit-config.yaml`.
   The `-n, --no-verify` flag of `git commit` can be used to deactivate pre-commit hooks temporarily.

2. install [nbstripout] git hooks to remove the output cells of committed notebooks with:

   ```bash
   nbstripout --install --attributes notebooks/.gitattributes
   ```

   This is useful to avoid large diffs due to plots in your notebooks.
   A simple `nbstripout --uninstall` will revert these changes.

Then take a look into the `scripts` and `notebooks` folders.

## __Project Organization__

```
├── AUTHORS.md              <- List of developers and maintainers.
├── CHANGELOG.md            <- Changelog to keep track of new features and fixes.
├── CONTRIBUTING.md         <- Guidelines for contributing to this project.
├── Dockerfile              <- Build a docker container with `docker build .`.
├── LICENSE.txt             <- License as chosen on the command-line.
├── README.md               <- The top-level README for developers.
├── configs                 <- Directory for configurations of model & application.
├── data
│   ├── external            <- Data from third party sources.
│   ├── interim             <- Intermediate data that has been transformed.
│   ├── processed           <- The final, canonical data sets for modeling.
│   └── raw                 <- The original, immutable data dump.
├── docs                    <- Directory for Sphinx documentation in rst or md.
├── requrements.txt         <- The python environment file for reproducibility.
├── models                  <- Trained and serialized models, model predictions,
│                              or model summaries.
├── notebooks               <- Jupyter notebooks. Naming convention is a number (for
│                              ordering), the creator's initials and a description,
│                              e.g. `1.0-fw-initial-data-exploration`.
├── pyproject.toml          <- Build configuration. Don't change! Use `pip install -e .`
│                              to install for development or to build `tox -e build`.
├── references              <- Data dictionaries, manuals, and all other materials.
├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures             <- Generated plots and figures for reports.
├── scripts                 <- Analysis and production scripts which import the
│                              actual PYTHON_PKG, e.g. train_model.
├── setup.py                <- Use `python setup.py develop` to install for
│                              development or `python setup.py bdist_wheel` to build.
├── src
│   └── kiwissenbase        <- Actual Python package where the main functionality goes.
├── tests                   <- Unit tests which can be run with `pytest`.
├── .coveragerc             <- Configuration for coverage reports of unit tests.
├── .isort.cfg              <- Configuration for git hook that sorts imports.
└── .pre-commit-config.yaml <- Configuration of pre-commit git hooks.
```

## __Minimal Example__

### __Model Training__

To train a model, use the script `scripts/train_model.py`. Which model to train, specific parameters and data/output directories are provided by `config.yaml`
files. See section [__Config Files__](#config-files) for details.

Train a model from a config by `python scripts/train_model.py --config path/to/config.yaml`. We provide some default config files:

- `configs/arxiv/bert.yaml`:
  Train Bert classifier on ArXiv dataset
- ...

Additional arguments to `scripts/train_model.py`:

- `--quiet / --verbose / --very-verbose`:
  Set log-level, i.e. number of logging messages shown during training.
- `-d / --debug`: Use to disable multiprocessing entirely. Useful for debugging.
- `-nc / --no-cuda`: Use to disable GPU-usage entirely. Useful for debugging.
- `--resume-training`: Use to resume a previous training.
- `--resume-from`: Provide save directory of previous training for resuming.

#### __Terminal Output__

![Model Training Terminal](reports/figures/training-terminal.png)

### __Starting Tensorboard__

In the framework we provide out of the box tensorboard logging. In order to see the training progress in tensorboard, first you need to start the tensorboard:

```bash
tensorboard --logdir results/logging/tensorboard
```

![Tensorboard Logging Example](reports/figures/tensorboard.png)

### __Config Files__

For reproducibility and tractability of the experiments done as well as for convenience we store all the models' hyperparameters into a __yaml__ config file.
All of the configuration files used to train the models are stored in `configs` folder. Each configuration file contains 5 main parts. The first part of the
yaml configuration file is:

```yaml
name: bert_arxiv
num_runs: 1
num_workers: 0
world_size: 1 #4 num_GPU
distributed: false #true
gpus: !!python/tuple ["0"] # ["0", "1", "2", "3"]
seed: 1
```

- `name:` Key holds the name of the experiment. The user can use any name that finds suitable for the experiment.
- `num_runs:` Number of times the experiment will be repeated.
- `num_works:` How many processes should be used for the training.
- `gpus:` Which gpus to be used.
- `seed:` Value of the initial seed.

The second part of the configuration file is the model.

```yaml
model:
  module: supervisedllm.models.baseline_models
  name: SequentialClassifier
  args:
    backbone_name: bert
    output_layers_dim: !!python/tuple [32, 32]
```

In this part the user can define which model will be used for the training as well as the hyperparameters. In the example above, we use __SequentialClassifier__ model. In order to do that we have supply the `module: supervisedllm.models.baseline_models` the python package where the model is and the `name: SequentialClassifier` name of the model. Next, in the `arg` key we set the all hyperparameters needed for the specific model.

The third part of the yaml file is the data loader part.

```yaml
data_loader:
  module: supervisedllm.data.dataloaders
  name: TopicDataLoader
  args:
    root_dir: ./data/preprocessed/arxiv
    is_dynamic: true
    use_covariates: false
    use_tmp_covariates: false
    reward_field: reward # reward_normalized
    transformer_name: bert # bert, albert, roberta
    batch_size: 6 #32 #8
    validation_batch_size: 6
    n_workers: 4
    pin_memory: true
```

The forth part is the optimizer that we are going to use during the training.

```yaml
optimizer:
  min_lr_rate: 1e-14 # used for early stopping
  gradient_norm_clipping: 1.0
  module: torch.optim
  name: SGD #Adam
  args:
    lr: 0.001
```

The last part that we have to define is the _trainer_. In this part we set all the parameters that are used for training and logging.

```yaml
trainer:
  module: supervisedllm.trainer
  name: BaseTrainingProcedure
  args:
    bm_metric: accuracy
    save_after_epoch: 1
    eval_test: false
    lr_schedulers: !!python/tuple
      - optimizer: # name of the optimizer
          counter: 1 # anneal lr rate if there is no improvement after n steps
          module: torch.optim.lr_scheduler
          name: StepLR # StepLR or MultiStepLR
          args:
            step_size: 3 # for StepLR
            gamma: 0.2
    schedulers: !!python/tuple
      - module: supervisedllm.utils.param_scheduler
        name: ExponentialScheduler
        label: beta_scheduler
        args:
          max_value: 1.0
          max_steps: 5000
          decay_rate: 0.0025
  epochs: 18 #4 #30 #20
  save_dir: ./results/saved/
  logging:
    logged_train_stats:
      !!python/tuple [
        "loss",
        "accuracy"
      ]
    logged_val_stats:
      !!python/tuple [
        "loss",
        "accuracy"
      ]
    logged_test_stats:
      !!python/tuple [
        "loss",
        "accuracy"
      ]
    tensorboard_dir: ./results/logging/tensorboard/
    logging_dir: ./results/logging/raw/
    formatters:
      verbose: "%(levelname)s %(asctime)s %(module)s %(process)d %(thread)d %(message)s"
      simple: "%(levelname)s %(asctime)s %(message)s"
```

In this library we have an object that is called _Trainer_ that is responsible for training the models and logging and creating checkpoints during training.

### __Evaluate Trained Model__

To evaluate a trained model, use the script `scripts/evaluate_model.py`. In order a model to be evaluated we need at least to _provide path to the trained model_, _path to the dataset root directory_ and the _output directory_, where the results will be stored. The script will create new dirctory with name that is concatination of the experiment name and the dataset split on which we want to evaluate the mode.

### __Example code for running evaluation:__

 ```bash
 python scripts/evaluate_model.py --model_dir path/to/model.pth --split [train|validate|test] --data-root-dir path/to/data_dir --output-dir path/to/output_dir
 ```

Additional arguments to `scripts/evaluate_model.py`:

- `--evaluation-custom-name`: Custom name for the folder where the evaluation will be stored. If not provided the name of the folder will be `experiment_name` + `dataset split`
- `--gpus`: GPUs used for evaluation.
- `--num-workers`: Number of threads used for the evaluation.
- `--quiet / --verbose / --very-verbose`: Set log-level, i.e. number of logging messages shown during training.
- `-d / --debug`: Use to disable multiprocessing entirely. Useful for debugging.
- `-nc / --no-cuda`: Use to disable GPU-usage entirely. Useful for debugging.

[supervisedllm.models]: src/supervisedllm/models/__init__.py
[supervisedllm.models.AModel]: src/supervisedllm/models/__init__.py
[supervisedllm.models.baseline_models.SequentialClassifier]: src/supervisedllm/models/baseline_models.py#273
[pre-commit]: https://pre-commit.com/
[nbstripout]: https://github.com/kynan/nbstripout
[virtualenv]: https://virtualenv.pypa.io/en/latest/installation.html
[virtualenvwrapper]: https://virtualenvwrapper.readthedocs.io/en/latest/install.html
