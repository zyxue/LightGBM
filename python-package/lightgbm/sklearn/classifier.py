"""Scikit-learn wrapper interface for LGBMClassifier."""

import numpy as np

from ..basic import _log_warning
from ..compat import (
    LGBMNotFittedError,
    _LGBMAssertAllFinite,
    _LGBMCheckClassificationTargets,
    _LGBMClassifierBase,
    _LGBMLabelEncoder,
)
from .doc import _lgbmmodel_doc_predict
from .model import LGBMModel


class LGBMClassifier(LGBMModel, _LGBMClassifierBase):
    """LightGBM classifier."""

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        init_score=None,
        eval_set=None,
        eval_names=None,
        eval_sample_weight=None,
        eval_class_weight=None,
        eval_init_score=None,
        eval_metric=None,
        early_stopping_rounds=None,
        verbose=True,
        feature_name="auto",
        categorical_feature="auto",
        callbacks=None,
        init_model=None,
    ):
        """Docstring is inherited from the LGBMModel."""
        _LGBMAssertAllFinite(y)
        _LGBMCheckClassificationTargets(y)
        self._le = _LGBMLabelEncoder().fit(y)
        _y = self._le.transform(y)
        self._class_map = dict(
            zip(self._le.classes_, self._le.transform(self._le.classes_))
        )
        if isinstance(self.class_weight, dict):
            self._class_weight = {
                self._class_map[k]: v for k, v in self.class_weight.items()
            }

        self._classes = self._le.classes_
        self._n_classes = len(self._classes)

        if self._n_classes > 2:
            # Switch to using a multiclass objective in the underlying LGBM instance
            ova_aliases = {"multiclassova", "multiclass_ova", "ova", "ovr"}
            if self._objective not in ova_aliases and not callable(self._objective):
                self._objective = "multiclass"

        if not callable(eval_metric):
            if isinstance(eval_metric, (str, type(None))):
                eval_metric = [eval_metric]
            if self._n_classes > 2:
                for index, metric in enumerate(eval_metric):
                    if metric in {"logloss", "binary_logloss"}:
                        eval_metric[index] = "multi_logloss"
                    elif metric in {"error", "binary_error"}:
                        eval_metric[index] = "multi_error"
            else:
                for index, metric in enumerate(eval_metric):
                    if metric in {"logloss", "multi_logloss"}:
                        eval_metric[index] = "binary_logloss"
                    elif metric in {"error", "multi_error"}:
                        eval_metric[index] = "binary_error"

        # do not modify args, as it causes errors in model selection tools
        valid_sets = None
        if eval_set is not None:
            if isinstance(eval_set, tuple):
                eval_set = [eval_set]
            valid_sets = [None] * len(eval_set)
            for i, (valid_x, valid_y) in enumerate(eval_set):
                if valid_x is X and valid_y is y:
                    valid_sets[i] = (valid_x, _y)
                else:
                    valid_sets[i] = (valid_x, self._le.transform(valid_y))

        super().fit(
            X,
            _y,
            sample_weight=sample_weight,
            init_score=init_score,
            eval_set=valid_sets,
            eval_names=eval_names,
            eval_sample_weight=eval_sample_weight,
            eval_class_weight=eval_class_weight,
            eval_init_score=eval_init_score,
            eval_metric=eval_metric,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
            feature_name=feature_name,
            categorical_feature=categorical_feature,
            callbacks=callbacks,
            init_model=init_model,
        )
        return self

    _base_doc = LGBMModel.fit.__doc__
    _base_doc = (
        _base_doc[: _base_doc.find("group :")]
        + _base_doc[_base_doc.find("eval_set :") :]
    )
    fit.__doc__ = (
        _base_doc[: _base_doc.find("eval_group :")]
        + _base_doc[_base_doc.find("eval_metric :") :]
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
        """Docstring is inherited from the LGBMModel."""
        result = self.predict_proba(
            X,
            raw_score,
            start_iteration,
            num_iteration,
            pred_leaf,
            pred_contrib,
            **kwargs
        )
        if callable(self._objective) or raw_score or pred_leaf or pred_contrib:
            return result
        else:
            class_index = np.argmax(result, axis=1)
            return self._le.inverse_transform(class_index)

    predict.__doc__ = LGBMModel.predict.__doc__

    def predict_proba(
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
        result = super().predict(
            X,
            raw_score,
            start_iteration,
            num_iteration,
            pred_leaf,
            pred_contrib,
            **kwargs
        )
        if callable(self._objective) and not (raw_score or pred_leaf or pred_contrib):
            _log_warning(
                "Cannot compute class probabilities or labels "
                "due to the usage of customized objective function.\n"
                "Returning raw scores instead."
            )
            return result
        elif self._n_classes > 2 or raw_score or pred_leaf or pred_contrib:
            return result
        else:
            return np.vstack((1.0 - result, result)).transpose()

    predict_proba.__doc__ = _lgbmmodel_doc_predict.format(
        description="Return the predicted probability for each class for each sample.",
        X_shape="array-like or sparse matrix of shape = [n_samples, n_features]",
        output_name="predicted_probability",
        predicted_result_shape="array-like of shape = [n_samples] or shape = [n_samples, n_classes]",
        X_leaves_shape="array-like of shape = [n_samples, n_trees] or shape = [n_samples, n_trees * n_classes]",
        X_SHAP_values_shape="array-like of shape = [n_samples, n_features + 1] or shape = [n_samples, (n_features + 1) * n_classes] or list with n_classes length of such objects",
    )

    @property
    def classes_(self):
        """:obj:`array` of shape = [n_classes]: The class label array."""
        if self._classes is None:
            raise LGBMNotFittedError("No classes found. Need to call fit beforehand.")
        return self._classes

    @property
    def n_classes_(self):
        """:obj:`int`: The number of classes."""
        if self._n_classes is None:
            raise LGBMNotFittedError("No classes found. Need to call fit beforehand.")
        return self._n_classes
