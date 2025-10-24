import numpy as np

seed = 22102017
rng = np.random.RandomState(seed)


class L1Penalty(object):
    """L1 parameter penalty.

    Term to add to the objective function penalising parameters
    based on their L1 norm.
    """

    def __init__(self, coefficient):
        """Create a new L1 penalty object.

        Args:
            coefficient: Positive constant to scale penalty term by.
        """
        assert coefficient > 0.0, "Penalty coefficient must be positive."
        self.coefficient = coefficient

    def __call__(self, parameter):
        """Calculate L1 penalty value for a parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Value of penalty term.
        """
        penalty = self.coefficient * np.sum(np.abs(parameter))
        return penalty

    def grad(self, parameter):
        """Calculate the penalty gradient with respect to the parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Value of penalty gradient with respect to parameter. This
            should be an array of the same shape as the parameter.
        """
        grad = self.coefficient * np.sign(parameter)
        return grad

    def __repr__(self):
        return "L1Penalty({0})".format(self.coefficient)


class L2Penalty(object):
    """L1 parameter penalty.

    Term to add to the objective function penalising parameters
    based on their L2 norm.
    """

    def __init__(self, coefficient):
        """Create a new L2 penalty object.

        Args:
            coefficient: Positive constant to scale penalty term by.
        """
        assert coefficient > 0.0, "Penalty coefficient must be positive."
        self.coefficient = coefficient

    def __call__(self, parameter):
        """Calculate L2 penalty value for a parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Value of penalty term.
        """
        penalty = 0.5 * self.coefficient * np.sum(parameter**2)
        return penalty

    def grad(self, parameter):
        """Calculate the penalty gradient with respect to the parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Value of penalty gradient with respect to parameter. This
            should be an array of the same shape as the parameter.
        """
        grad = self.coefficient * parameter
        return grad

    def __repr__(self):
        return "L2Penalty({0})".format(self.coefficient)


class L1L2MixPenalty(object):
    """L1 & L2 mix penalty."""

    def __init__(self, l1_coefficient, l2_coefficient):
        """Create a new L1 & L2 mix penalty object.

        Args:
            coefficient: Positive constant to scale penalty term by.
        """
        self.l1_coefficient = l1_coefficient
        self.l2_coefficient = l2_coefficient

    def __call__(self, parameter):
        """Calculate L1 & L2 mix penalty value for a parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Value of penalty term.
        """
        l1_penalty = self.l1_coefficient * np.sum(np.abs(parameter))
        l2_penalty = 0.5 * self.l2_coefficient * np.sum(parameter**2)
        penalty = l1_penalty + l2_penalty
        return penalty

    def grad(self, parameter):
        """Calculate the penalty gradient with respect to the parameter.

        Args:
            parameter: Array corresponding to a model parameter.

        Returns:
            Value of penalty gradient with respect to parameter. This
            should be an array of the same shape as the parameter.
        """
        l1_grad = self.l1_coefficient * np.sign(parameter)
        l2_grad = self.l2_coefficient * parameter
        grad = l1_grad + l2_grad
        return grad

    def __repr__(self):
        return "L1L2MixPenalty(l1_coefficient={0}, l2_coefficient={1})".format(
            self.l1_coefficient, self.l2_coefficient
        )
