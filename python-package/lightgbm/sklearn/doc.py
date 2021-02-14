# documentation templates for LGBMModel methods are shared between the classes in
# this module and those in the ``dask`` module

_lgbmmodel_doc_fit = """
    Build a gradient boosting model from the training set (X, y).

    Parameters
    ----------
    X : {X_shape}
        Input feature matrix.
    y : {y_shape}
        The target values (class labels in classification, real numbers in regression).
    sample_weight : {sample_weight_shape}
        Weights of training data.
    init_score : array-like of shape = [n_samples] or None, optional (default=None)
        Init score of training data.
    group : {group_shape}
        Group/query data.
        Only used in the learning-to-rank task.
        sum(group) = n_samples.
        For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,
        where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.
    eval_set : list or None, optional (default=None)
        A list of (X, y) tuple pairs to use as validation sets.
    eval_names : list of strings or None, optional (default=None)
        Names of eval_set.
    eval_sample_weight : list of arrays or None, optional (default=None)
        Weights of eval data.
    eval_class_weight : list or None, optional (default=None)
        Class weights of eval data.
    eval_init_score : list of arrays or None, optional (default=None)
        Init score of eval data.
    eval_group : list of arrays or None, optional (default=None)
        Group data of eval data.
    eval_metric : string, callable, list or None, optional (default=None)
        If string, it should be a built-in evaluation metric to use.
        If callable, it should be a custom evaluation metric, see note below for more details.
        If list, it can be a list of built-in metrics, a list of custom evaluation metrics, or a mix of both.
        In either case, the ``metric`` from the model parameters will be evaluated and used as well.
        Default: 'l2' for LGBMRegressor, 'logloss' for LGBMClassifier, 'ndcg' for LGBMRanker.
    early_stopping_rounds : int or None, optional (default=None)
        Activates early stopping. The model will train until the validation score stops improving.
        Validation score needs to improve at least every ``early_stopping_rounds`` round(s)
        to continue training.
        Requires at least one validation data and one metric.
        If there's more than one, will check all of them. But the training data is ignored anyway.
        To check only the first metric, set the ``first_metric_only`` parameter to ``True``
        in additional parameters ``**kwargs`` of the model constructor.
    verbose : bool or int, optional (default=True)
        Requires at least one evaluation data.
        If True, the eval metric on the eval set is printed at each boosting stage.
        If int, the eval metric on the eval set is printed at every ``verbose`` boosting stage.
        The last boosting stage or the boosting stage found by using ``early_stopping_rounds`` is also printed.

        .. rubric:: Example

        With ``verbose`` = 4 and at least one item in ``eval_set``,
        an evaluation metric is printed every 4 (instead of 1) boosting stages.

    feature_name : list of strings or 'auto', optional (default='auto')
        Feature names.
        If 'auto' and data is pandas DataFrame, data columns names are used.
    categorical_feature : list of strings or int, or 'auto', optional (default='auto')
        Categorical features.
        If list of int, interpreted as indices.
        If list of strings, interpreted as feature names (need to specify ``feature_name`` as well).
        If 'auto' and data is pandas DataFrame, pandas unordered categorical columns are used.
        All values in categorical features should be less than int32 max value (2147483647).
        Large values could be memory consuming. Consider using consecutive integers starting from zero.
        All negative values in categorical features will be treated as missing values.
        The output cannot be monotonically constrained with respect to a categorical feature.
    callbacks : list of callback functions or None, optional (default=None)
        List of callback functions that are applied at each iteration.
        See Callbacks in Python API for more information.
    init_model : string, Booster, LGBMModel or None, optional (default=None)
        Filename of LightGBM model, Booster instance or LGBMModel instance used for continue training.

    Returns
    -------
    self : object
        Returns self.
    """

_lgbmmodel_doc_custom_eval_note = """
    Note
    ----
    Custom eval function expects a callable with following signatures:
    ``func(y_true, y_pred)``, ``func(y_true, y_pred, weight)`` or
    ``func(y_true, y_pred, weight, group)``
    and returns (eval_name, eval_result, is_higher_better) or
    list of (eval_name, eval_result, is_higher_better):

        y_true : array-like of shape = [n_samples]
            The target values.
        y_pred : array-like of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task)
            The predicted values.
        weight : array-like of shape = [n_samples]
            The weight of samples.
        group : array-like
            Group/query data.
            Only used in the learning-to-rank task.
            sum(group) = n_samples.
            For example, if you have a 100-document dataset with ``group = [10, 20, 40, 10, 10, 10]``, that means that you have 6 groups,
            where the first 10 records are in the first group, records 11-30 are in the second group, records 31-70 are in the third group, etc.
        eval_name : string
            The name of evaluation function (without whitespaces).
        eval_result : float
            The eval result.
        is_higher_better : bool
            Is eval result higher better, e.g. AUC is ``is_higher_better``.

    For binary task, the y_pred is probability of positive class (or margin in case of custom ``objective``).
    For multi-class task, the y_pred is group by class_id first, then group by row_id.
    If you want to get i-th row y_pred in j-th class, the access way is y_pred[j * num_data + i].
"""

_lgbmmodel_doc_predict = """
    {description}

    Parameters
    ----------
    X : {X_shape}
        Input features matrix.
    raw_score : bool, optional (default=False)
        Whether to predict raw scores.
    start_iteration : int, optional (default=0)
        Start index of the iteration to predict.
        If <= 0, starts from the first iteration.
    num_iteration : int or None, optional (default=None)
        Total number of iterations used in the prediction.
        If None, if the best iteration exists and start_iteration <= 0, the best iteration is used;
        otherwise, all iterations from ``start_iteration`` are used (no limits).
        If <= 0, all iterations from ``start_iteration`` are used (no limits).
    pred_leaf : bool, optional (default=False)
        Whether to predict leaf index.
    pred_contrib : bool, optional (default=False)
        Whether to predict feature contributions.

        .. note::

            If you want to get more explanations for your model's predictions using SHAP values,
            like SHAP interaction values,
            you can install the shap package (https://github.com/slundberg/shap).
            Note that unlike the shap package, with ``pred_contrib`` we return a matrix with an extra
            column, where the last column is the expected value.

    **kwargs
        Other parameters for the prediction.

    Returns
    -------
    {output_name} : {predicted_result_shape}
        The predicted values.
    X_leaves : {X_leaves_shape}
        If ``pred_leaf=True``, the predicted leaf of every tree for each sample.
    X_SHAP_values : {X_SHAP_values_shape}
        If ``pred_contrib=True``, the feature contributions for each sample.
    """
