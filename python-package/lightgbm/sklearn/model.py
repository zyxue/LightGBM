# coding: utf-8
"""Scikit-learn wrapper interface for LightGBM."""
import copy
from inspect import signature

import numpy as np

from ..basic import Dataset, LightGBMError, _choose_param_value, _ConfigAliases
from ..compat import (
    SKLEARN_INSTALLED,
    LGBMNotFittedError,
    _LGBMCheckArray,
    _LGBMCheckSampleWeight,
    _LGBMCheckXY,
    _LGBMComputeSampleWeight,
    _LGBMModelBase,
    dt_DataTable,
    pd_DataFrame,
)
from ..engine import train
from .doc import (
    _lgbmmodel_doc_custom_eval_note,
    _lgbmmodel_doc_fit,
    _lgbmmodel_doc_predict,
)


class LGBMModel(_LGBMModelBase):
    """Implementation of the scikit-learn API for LightGBM."""

    def __init__(
        self,
        boosting_type="gbdt",
        num_leaves=31,
        max_depth=-1,
        learning_rate=0.1,
        n_estimators=100,
        subsample_for_bin=200000,
        objective=None,
        class_weight=None,
        min_split_gain=0.0,
        min_child_weight=1e-3,
        min_child_samples=20,
        subsample=1.0,
        subsample_freq=0,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=None,
        n_jobs=-1,
        silent=True,
        importance_type="split",
        **kwargs
    ):
        r"""Construct a gradient boosting model.

        Parameters
        ----------
        boosting_type : string, optional (default='gbdt')
            'gbdt', traditional Gradient Boosting Decision Tree.
            'dart', Dropouts meet Multiple Additive Regression Trees.
            'goss', Gradient-based One-Side Sampling.
            'rf', Random Forest.
        num_leaves : int, optional (default=31)
            Maximum tree leaves for base learners.
        max_depth : int, optional (default=-1)
            Maximum tree depth for base learners, <=0 means no limit.
        learning_rate : float, optional (default=0.1)
            Boosting learning rate.
            You can use ``callbacks`` parameter of ``fit`` method to shrink/adapt learning rate
            in training using ``reset_parameter`` callback.
            Note, that this will ignore the ``learning_rate`` argument in training.
        n_estimators : int, optional (default=100)
            Number of boosted trees to fit.
        subsample_for_bin : int, optional (default=200000)
            Number of samples for constructing bins.
        objective : string, callable or None, optional (default=None)
            Specify the learning task and the corresponding learning objective or
            a custom objective function to be used (see note below).
            Default: 'regression' for LGBMRegressor, 'binary' or 'multiclass' for LGBMClassifier, 'lambdarank' for LGBMRanker.
        class_weight : dict, 'balanced' or None, optional (default=None)
            Weights associated with classes in the form ``{class_label: weight}``.
            Use this parameter only for multi-class classification task;
            for binary classification task you may use ``is_unbalance`` or ``scale_pos_weight`` parameters.
            Note, that the usage of all these parameters will result in poor estimates of the individual class probabilities.
            You may want to consider performing probability calibration
            (https://scikit-learn.org/stable/modules/calibration.html) of your model.
            The 'balanced' mode uses the values of y to automatically adjust weights
            inversely proportional to class frequencies in the input data as ``n_samples / (n_classes * np.bincount(y))``.
            If None, all classes are supposed to have weight one.
            Note, that these weights will be multiplied with ``sample_weight`` (passed through the ``fit`` method)
            if ``sample_weight`` is specified.
        min_split_gain : float, optional (default=0.)
            Minimum loss reduction required to make a further partition on a leaf node of the tree.
        min_child_weight : float, optional (default=1e-3)
            Minimum sum of instance weight (hessian) needed in a child (leaf).
        min_child_samples : int, optional (default=20)
            Minimum number of data needed in a child (leaf).
        subsample : float, optional (default=1.)
            Subsample ratio of the training instance.
        subsample_freq : int, optional (default=0)
            Frequence of subsample, <=0 means no enable.
        colsample_bytree : float, optional (default=1.)
            Subsample ratio of columns when constructing each tree.
        reg_alpha : float, optional (default=0.)
            L1 regularization term on weights.
        reg_lambda : float, optional (default=0.)
            L2 regularization term on weights.
        random_state : int, RandomState object or None, optional (default=None)
            Random number seed.
            If int, this number is used to seed the C++ code.
            If RandomState object (numpy), a random integer is picked based on its state to seed the C++ code.
            If None, default seeds in C++ code are used.
        n_jobs : int, optional (default=-1)
            Number of parallel threads.
        silent : bool, optional (default=True)
            Whether to print messages while running boosting.
        importance_type : string, optional (default='split')
            The type of feature importance to be filled into ``feature_importances_``.
            If 'split', result contains numbers of times the feature is used in a model.
            If 'gain', result contains total gains of splits which use the feature.
        **kwargs
            Other parameters for the model.
            Check http://lightgbm.readthedocs.io/en/latest/Parameters.html for more parameters.

            .. warning::

                \*\*kwargs is not supported in sklearn, it may cause unexpected issues.

        Note
        ----
        A custom objective function can be provided for the ``objective`` parameter.
        In this case, it should have the signature
        ``objective(y_true, y_pred) -> grad, hess`` or
        ``objective(y_true, y_pred, group) -> grad, hess``:

            y_true : array-like of shape = [n_samples]
                The target values.
            y_pred : array-like of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task)
                The predicted values.
            group : array-like
                Group/query data.
                Only used in the learning-to-rank task.
                sum(group) = n_samples.
                For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,
                where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.
            grad : array-like of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task)
                The value of the first order derivative (gradient) for each sample point.
            hess : array-like of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task)
                The value of the second order derivative (Hessian) for each sample point.

        For binary task, the y_pred is margin.
        For multi-class task, the y_pred is group by class_id first, then group by row_id.
        If you want to get i-th row y_pred in j-th class, the access way is y_pred[j * num_data + i]
        and you should group grad and hess in this way as well.
        """
        if not SKLEARN_INSTALLED:
            raise LightGBMError("scikit-learn is required for lightgbm.sklearn")

        self.boosting_type = boosting_type
        self.objective = objective
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample_for_bin = subsample_for_bin
        self.min_split_gain = min_split_gain
        self.min_child_weight = min_child_weight
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.subsample_freq = subsample_freq
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.silent = silent
        self.importance_type = importance_type
        self._Booster = None
        self._evals_result = None
        self._best_score = None
        self._best_iteration = None
        self._other_params = {}
        self._objective = objective
        self.class_weight = class_weight
        self._class_weight = None
        self._class_map = None
        self._n_features = None
        self._n_features_in = None
        self._classes = None
        self._n_classes = None
        self.set_params(**kwargs)

    def _more_tags(self):
        return {
            "allow_nan": True,
            "X_types": ["2darray", "sparse", "1dlabels"],
            "_xfail_checks": {
                "check_no_attributes_set_in_init": "scikit-learn incorrectly asserts that private attributes "
                "cannot be set in __init__: "
                "(see https://github.com/microsoft/LightGBM/issues/2628)"
            },
        }

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, optional (default=True)
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        params = super().get_params(deep=deep)
        params.update(self._other_params)
        return params

    def set_params(self, **params):
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params
            Parameter names with their new values.

        Returns
        -------
        self : object
            Returns self.
        """
        for key, value in params.items():
            setattr(self, key, value)
            if hasattr(self, "_" + key):
                setattr(self, "_" + key, value)
            self._other_params[key] = value
        return self

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        init_score=None,
        group=None,
        eval_set=None,
        eval_names=None,
        eval_sample_weight=None,
        eval_class_weight=None,
        eval_init_score=None,
        eval_group=None,
        eval_metric=None,
        early_stopping_rounds=None,
        verbose=True,
        feature_name="auto",
        categorical_feature="auto",
        callbacks=None,
        init_model=None,
    ):
        """Docstring is set after definition, using a template."""
        if self._objective is None:
            if isinstance(self, LGBMRegressor):
                self._objective = "regression"
            elif isinstance(self, LGBMClassifier):
                self._objective = "binary"
            elif isinstance(self, LGBMRanker):
                self._objective = "lambdarank"
            else:
                raise ValueError("Unknown LGBMModel type.")
        if callable(self._objective):
            self._fobj = _ObjectiveFunctionWrapper(self._objective)
        else:
            self._fobj = None
        evals_result = {}
        params = self.get_params()
        # user can set verbose with kwargs, it has higher priority
        if (
            not any(
                verbose_alias in params
                for verbose_alias in _ConfigAliases.get("verbosity")
            )
            and self.silent
        ):
            params["verbose"] = -1
        params.pop("silent", None)
        params.pop("importance_type", None)
        params.pop("n_estimators", None)
        params.pop("class_weight", None)
        if isinstance(params["random_state"], np.random.RandomState):
            params["random_state"] = params["random_state"].randint(
                np.iinfo(np.int32).max
            )
        for alias in _ConfigAliases.get("objective"):
            params.pop(alias, None)
        if self._n_classes is not None and self._n_classes > 2:
            for alias in _ConfigAliases.get("num_class"):
                params.pop(alias, None)
            params["num_class"] = self._n_classes
        if hasattr(self, "_eval_at"):
            for alias in _ConfigAliases.get("eval_at"):
                params.pop(alias, None)
            params["eval_at"] = self._eval_at
        params["objective"] = self._objective
        if self._fobj:
            params["objective"] = "None"  # objective = nullptr for unknown objective

        # Do not modify original args in fit function
        # Refer to https://github.com/microsoft/LightGBM/pull/2619
        eval_metric_list = copy.deepcopy(eval_metric)
        if not isinstance(eval_metric_list, list):
            eval_metric_list = [eval_metric_list]

        # Separate built-in from callable evaluation metrics
        eval_metrics_callable = [
            _EvalFunctionWrapper(f) for f in eval_metric_list if callable(f)
        ]
        eval_metrics_builtin = [m for m in eval_metric_list if isinstance(m, str)]

        # register default metric for consistency with callable eval_metric case
        original_metric = self._objective if isinstance(self._objective, str) else None
        if original_metric is None:
            # try to deduce from class instance
            if isinstance(self, LGBMRegressor):
                original_metric = "l2"
            elif isinstance(self, LGBMClassifier):
                original_metric = (
                    "multi_logloss" if self._n_classes > 2 else "binary_logloss"
                )
            elif isinstance(self, LGBMRanker):
                original_metric = "ndcg"

        # overwrite default metric by explicitly set metric
        params = _choose_param_value("metric", params, original_metric)

        # concatenate metric from params (or default if not provided in params) and eval_metric
        params["metric"] = (
            [params["metric"]]
            if isinstance(params["metric"], (str, type(None)))
            else params["metric"]
        )
        params["metric"] = [
            e for e in eval_metrics_builtin if e not in params["metric"]
        ] + params["metric"]
        params["metric"] = [metric for metric in params["metric"] if metric is not None]

        if not isinstance(X, (pd_DataFrame, dt_DataTable)):
            _X, _y = _LGBMCheckXY(
                X, y, accept_sparse=True, force_all_finite=False, ensure_min_samples=2
            )
            if sample_weight is not None:
                sample_weight = _LGBMCheckSampleWeight(sample_weight, _X)
        else:
            _X, _y = X, y

        if self._class_weight is None:
            self._class_weight = self.class_weight
        if self._class_weight is not None:
            class_sample_weight = _LGBMComputeSampleWeight(self._class_weight, y)
            if sample_weight is None or len(sample_weight) == 0:
                sample_weight = class_sample_weight
            else:
                sample_weight = np.multiply(sample_weight, class_sample_weight)

        self._n_features = _X.shape[1]
        # copy for consistency
        self._n_features_in = self._n_features

        def _construct_dataset(
            X,
            y,
            sample_weight,
            init_score,
            group,
            params,
            categorical_feature="auto",
        ):
            return Dataset(
                X,
                label=y,
                weight=sample_weight,
                group=group,
                init_score=init_score,
                params=params,
                categorical_feature=categorical_feature,
            )

        train_set = _construct_dataset(
            _X,
            _y,
            sample_weight,
            init_score,
            group,
            params,
            categorical_feature=categorical_feature,
        )

        valid_sets = []
        if eval_set is not None:

            def _get_meta_data(collection, name, i):
                if collection is None:
                    return None
                elif isinstance(collection, list):
                    return collection[i] if len(collection) > i else None
                elif isinstance(collection, dict):
                    return collection.get(i, None)
                else:
                    raise TypeError("{} should be dict or list".format(name))

            if isinstance(eval_set, tuple):
                eval_set = [eval_set]
            for i, valid_data in enumerate(eval_set):
                # reduce cost for prediction training data
                if valid_data[0] is X and valid_data[1] is y:
                    valid_set = train_set
                else:
                    valid_weight = _get_meta_data(
                        eval_sample_weight, "eval_sample_weight", i
                    )
                    valid_class_weight = _get_meta_data(
                        eval_class_weight, "eval_class_weight", i
                    )
                    if valid_class_weight is not None:
                        if (
                            isinstance(valid_class_weight, dict)
                            and self._class_map is not None
                        ):
                            valid_class_weight = {
                                self._class_map[k]: v
                                for k, v in valid_class_weight.items()
                            }
                        valid_class_sample_weight = _LGBMComputeSampleWeight(
                            valid_class_weight, valid_data[1]
                        )
                        if valid_weight is None or len(valid_weight) == 0:
                            valid_weight = valid_class_sample_weight
                        else:
                            valid_weight = np.multiply(
                                valid_weight, valid_class_sample_weight
                            )
                    valid_init_score = _get_meta_data(
                        eval_init_score, "eval_init_score", i
                    )
                    valid_group = _get_meta_data(eval_group, "eval_group", i)
                    valid_set = _construct_dataset(
                        valid_data[0],
                        valid_data[1],
                        valid_weight,
                        valid_init_score,
                        valid_group,
                        params,
                    )
                valid_sets.append(valid_set)

        if isinstance(init_model, LGBMModel):
            init_model = init_model.booster_

        self._Booster = train(
            params,
            train_set,
            self.n_estimators,
            valid_sets=valid_sets,
            valid_names=eval_names,
            early_stopping_rounds=early_stopping_rounds,
            evals_result=evals_result,
            fobj=self._fobj,
            feval=eval_metrics_callable,
            verbose_eval=verbose,
            feature_name=feature_name,
            callbacks=callbacks,
            init_model=init_model,
        )

        if evals_result:
            self._evals_result = evals_result

        if early_stopping_rounds is not None and early_stopping_rounds > 0:
            self._best_iteration = self._Booster.best_iteration

        self._best_score = self._Booster.best_score

        self.fitted_ = True

        # free dataset
        self._Booster.free_dataset()
        del train_set, valid_sets
        return self

    fit.__doc__ = (
        _lgbmmodel_doc_fit.format(
            X_shape="array-like or sparse matrix of shape = [n_samples, n_features]",
            y_shape="array-like of shape = [n_samples]",
            sample_weight_shape="array-like of shape = [n_samples] or None, optional (default=None)",
            group_shape="array-like or None, optional (default=None)",
        )
        + "\n\n"
        + _lgbmmodel_doc_custom_eval_note
    )

    def predict(
        self,
        X,
        raw_score=False,
        start_iteration=0,
        num_iteration=None,
        pred_leaf=False,
        pred_contrib=False,
        **kwargs
    ):
        """Docstring is set after definition, using a template."""
        if self._n_features is None:
            raise LGBMNotFittedError(
                "Estimator not fitted, call `fit` before exploiting the model."
            )
        if not isinstance(X, (pd_DataFrame, dt_DataTable)):
            X = _LGBMCheckArray(X, accept_sparse=True, force_all_finite=False)
        n_features = X.shape[1]
        if self._n_features != n_features:
            raise ValueError(
                "Number of features of the model must "
                "match the input. Model n_features_ is %s and "
                "input n_features is %s " % (self._n_features, n_features)
            )
        return self._Booster.predict(
            X,
            raw_score=raw_score,
            start_iteration=start_iteration,
            num_iteration=num_iteration,
            pred_leaf=pred_leaf,
            pred_contrib=pred_contrib,
            **kwargs
        )

    predict.__doc__ = _lgbmmodel_doc_predict.format(
        description="Return the predicted value for each sample.",
        X_shape="array-like or sparse matrix of shape = [n_samples, n_features]",
        output_name="predicted_result",
        predicted_result_shape="array-like of shape = [n_samples] or shape = [n_samples, n_classes]",
        X_leaves_shape="array-like of shape = [n_samples, n_trees] or shape = [n_samples, n_trees * n_classes]",
        X_SHAP_values_shape="array-like of shape = [n_samples, n_features + 1] or shape = [n_samples, (n_features + 1) * n_classes] or list with n_classes length of such objects",
    )

    @property
    def n_features_(self):
        """:obj:`int`: The number of features of fitted model."""
        if self._n_features is None:
            raise LGBMNotFittedError(
                "No n_features found. Need to call fit beforehand."
            )
        return self._n_features

    @property
    def n_features_in_(self):
        """:obj:`int`: The number of features of fitted model."""
        if self._n_features_in is None:
            raise LGBMNotFittedError(
                "No n_features_in found. Need to call fit beforehand."
            )
        return self._n_features_in

    @property
    def best_score_(self):
        """:obj:`dict` or :obj:`None`: The best score of fitted model."""
        if self._n_features is None:
            raise LGBMNotFittedError(
                "No best_score found. Need to call fit beforehand."
            )
        return self._best_score

    @property
    def best_iteration_(self):
        """:obj:`int` or :obj:`None`: The best iteration of fitted model if ``early_stopping_rounds`` has been specified."""
        if self._n_features is None:
            raise LGBMNotFittedError(
                "No best_iteration found. Need to call fit with early_stopping_rounds beforehand."
            )
        return self._best_iteration

    @property
    def objective_(self):
        """:obj:`string` or :obj:`callable`: The concrete objective used while fitting this model."""
        if self._n_features is None:
            raise LGBMNotFittedError("No objective found. Need to call fit beforehand.")
        return self._objective

    @property
    def booster_(self):
        """Booster: The underlying Booster of this model."""
        if self._Booster is None:
            raise LGBMNotFittedError("No booster found. Need to call fit beforehand.")
        return self._Booster

    @property
    def evals_result_(self):
        """:obj:`dict` or :obj:`None`: The evaluation results if ``early_stopping_rounds`` has been specified."""
        if self._n_features is None:
            raise LGBMNotFittedError(
                "No results found. Need to call fit with eval_set beforehand."
            )
        return self._evals_result

    @property
    def feature_importances_(self):
        """:obj:`array` of shape = [n_features]: The feature importances (the higher, the more important).

        .. note::

            ``importance_type`` attribute is passed to the function
            to configure the type of importance values to be extracted.
        """
        if self._n_features is None:
            raise LGBMNotFittedError(
                "No feature_importances found. Need to call fit beforehand."
            )
        return self._Booster.feature_importance(importance_type=self.importance_type)

    @property
    def feature_name_(self):
        """:obj:`array` of shape = [n_features]: The names of features."""
        if self._n_features is None:
            raise LGBMNotFittedError(
                "No feature_name found. Need to call fit beforehand."
            )
        return self._Booster.feature_name()
