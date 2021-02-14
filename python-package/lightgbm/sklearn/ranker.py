"""Scikit-learn wrapper interface for LGBMRanker."""

from .model import LGBMModel


class LGBMRanker(LGBMModel):
    """LightGBM ranker."""

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
        eval_init_score=None,
        eval_group=None,
        eval_metric=None,
        eval_at=(1, 2, 3, 4, 5),
        early_stopping_rounds=None,
        verbose=True,
        feature_name="auto",
        categorical_feature="auto",
        callbacks=None,
        init_model=None,
    ):
        """Docstring is inherited from the LGBMModel."""
        # check group data
        if group is None:
            raise ValueError("Should set group for ranking task")

        if eval_set is not None:
            if eval_group is None:
                raise ValueError("Eval_group cannot be None when eval_set is not None")
            elif len(eval_group) != len(eval_set):
                raise ValueError("Length of eval_group should be equal to eval_set")
            elif (
                isinstance(eval_group, dict)
                and any(
                    i not in eval_group or eval_group[i] is None
                    for i in range(len(eval_group))
                )
                or isinstance(eval_group, list)
                and any(group is None for group in eval_group)
            ):
                raise ValueError(
                    "Should set group for all eval datasets for ranking task; "
                    "if you use dict, the index should start from 0"
                )

        self._eval_at = eval_at
        super().fit(
            X,
            y,
            sample_weight=sample_weight,
            init_score=init_score,
            group=group,
            eval_set=eval_set,
            eval_names=eval_names,
            eval_sample_weight=eval_sample_weight,
            eval_init_score=eval_init_score,
            eval_group=eval_group,
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
    fit.__doc__ = (
        _base_doc[: _base_doc.find("eval_class_weight :")]
        + _base_doc[_base_doc.find("eval_init_score :") :]
    )
    _base_doc = fit.__doc__
    _before_early_stop, _early_stop, _after_early_stop = _base_doc.partition(
        "early_stopping_rounds :"
    )
    fit.__doc__ = (
        _before_early_stop
        + "eval_at : iterable of int, optional (default=(1, 2, 3, 4, 5))\n"
        + " " * 12
        + "The evaluation positions of the specified metric.\n"
        + " " * 8
        + _early_stop
        + _after_early_stop
    )
