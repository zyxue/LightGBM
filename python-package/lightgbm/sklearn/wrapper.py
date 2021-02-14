class _ObjectiveFunctionWrapper:
    """Proxy class for objective function."""

    def __init__(self, func):
        """Construct a proxy class.

        This class transforms objective function to match objective function with signature ``new_func(preds, dataset)``
        as expected by ``lightgbm.engine.train``.

        Parameters
        ----------
        func : callable
            Expects a callable with signature ``func(y_true, y_pred)`` or ``func(y_true, y_pred, group)
            and returns (grad, hess):

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

        .. note::

            For binary task, the y_pred is margin.
            For multi-class task, the y_pred is group by class_id first, then group by row_id.
            If you want to get i-th row y_pred in j-th class, the access way is y_pred[j * num_data + i]
            and you should group grad and hess in this way as well.
        """
        self.func = func

    def __call__(self, preds, dataset):
        """Call passed function with appropriate arguments.

        Parameters
        ----------
        preds : array-like of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task)
            The predicted values.
        dataset : Dataset
            The training dataset.

        Returns
        -------
        grad : array-like of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task)
            The value of the first order derivative (gradient) for each sample point.
        hess : array-like of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task)
            The value of the second order derivative (Hessian) for each sample point.
        """
        labels = dataset.get_label()
        argc = len(signature(self.func).parameters)
        if argc == 2:
            grad, hess = self.func(labels, preds)
        elif argc == 3:
            grad, hess = self.func(labels, preds, dataset.get_group())
        else:
            raise TypeError(
                "Self-defined objective function should have 2 or 3 arguments, got %d"
                % argc
            )
        """weighted for objective"""
        weight = dataset.get_weight()
        if weight is not None:
            """only one class"""
            if len(weight) == len(grad):
                grad = np.multiply(grad, weight)
                hess = np.multiply(hess, weight)
            else:
                num_data = len(weight)
                num_class = len(grad) // num_data
                if num_class * num_data != len(grad):
                    raise ValueError(
                        "Length of grad and hess should equal to num_class * num_data"
                    )
                for k in range(num_class):
                    for i in range(num_data):
                        idx = k * num_data + i
                        grad[idx] *= weight[i]
                        hess[idx] *= weight[i]
        return grad, hess


class _EvalFunctionWrapper:
    """Proxy class for evaluation function."""

    def __init__(self, func):
        """Construct a proxy class.

        This class transforms evaluation function to match evaluation function with signature ``new_func(preds, dataset)``
        as expected by ``lightgbm.engine.train``.

        Parameters
        ----------
        func : callable
            Expects a callable with following signatures:
            ``func(y_true, y_pred)``,
            ``func(y_true, y_pred, weight)``
            or ``func(y_true, y_pred, weight, group)``
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

        .. note::

            For binary task, the y_pred is probability of positive class (or margin in case of custom ``objective``).
            For multi-class task, the y_pred is group by class_id first, then group by row_id.
            If you want to get i-th row y_pred in j-th class, the access way is y_pred[j * num_data + i].
        """
        self.func = func

    def __call__(self, preds, dataset):
        """Call passed function with appropriate arguments.

        Parameters
        ----------
        preds : array-like of shape = [n_samples] or shape = [n_samples * n_classes] (for multi-class task)
            The predicted values.
        dataset : Dataset
            The training dataset.

        Returns
        -------
        eval_name : string
            The name of evaluation function (without whitespaces).
        eval_result : float
            The eval result.
        is_higher_better : bool
            Is eval result higher better, e.g. AUC is ``is_higher_better``.
        """
        labels = dataset.get_label()
        argc = len(signature(self.func).parameters)
        if argc == 2:
            return self.func(labels, preds)
        elif argc == 3:
            return self.func(labels, preds, dataset.get_weight())
        elif argc == 4:
            return self.func(labels, preds, dataset.get_weight(), dataset.get_group())
        else:
            raise TypeError(
                "Self-defined eval function should have 2, 3 or 4 arguments, got %d"
                % argc
            )
