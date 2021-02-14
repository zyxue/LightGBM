"""Scikit-learn wrapper interface for LGBMRegressor."""

from ..compat import _LGBMRegressorBase
from .model import LGBMModel


class LGBMRegressor(LGBMModel, _LGBMRegressorBase):
    """LightGBM regressor."""

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        init_score=None,
        eval_set=None,
        eval_names=None,
        eval_sample_weight=None,
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
        super().fit(
            X,
            y,
            sample_weight=sample_weight,
            init_score=init_score,
            eval_set=eval_set,
            eval_names=eval_names,
            eval_sample_weight=eval_sample_weight,
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
    _base_doc = (
        _base_doc[: _base_doc.find("eval_class_weight :")]
        + _base_doc[_base_doc.find("eval_init_score :") :]
    )
    fit.__doc__ = (
        _base_doc[: _base_doc.find("eval_group :")]
        + _base_doc[_base_doc.find("eval_metric :") :]
    )
