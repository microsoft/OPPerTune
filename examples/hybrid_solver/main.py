from typing import Any, Dict, List, Union

from oppertune import CategoricalValue, ContinuousValue, DiscreteValue, OPPerTune

# The types of parameters we will tune in this example.
ParameterTypes = Union[CategoricalValue, ContinuousValue, DiscreteValue]


class Application:
    """Dummy class representing the application of interest (whose configuration parameters we want to tune)"""

    def set_parameters(self, parameters):
        """Set the configuration parameters of the app which it will use in its next run"""
        ...

    def run(self):
        """Run the app and wait for it to complete"""
        ...


class Metrics:
    """Dummy class representing how a service stores its captured metrics"""

    ...


class MetricsMonitor:
    """Dummy class representing a service that captures metrics for the application"""

    def start(self):
        """Start capturing metrics in the background"""
        ...

    def stop(self) -> Metrics:
        """Stop the capture and return the captured metrics"""
        ...


def deploy_and_get_metrics(app: Application, parameters: Dict[str, Any]) -> Metrics:
    """Configure the app with the supplied parameters, run it and return the metrics observed"""
    monitor = MetricsMonitor()
    monitor.start()

    app.set_parameters(parameters)
    app.run()

    metrics = monitor.stop()
    return metrics


def calculate_reward(predicted_parameters: Dict[str, Any], parameters: List[ParameterTypes]) -> float:
    # Since this is a dummy example, we assume that we know the optimal parameters
    # for the application and calculate how far are we from these.
    optimal_parameters = {
        "max-total-unready-percentage": 0.825,
        "scan-interval": 60,
        "expander": "most-pods",
    }

    reward: float = 0.0
    for p in parameters:
        if isinstance(p, (ContinuousValue, DiscreteValue)):
            difference = optimal_parameters[p.name] - predicted_parameters[p.name]
            value_range = p.ub - p.lb  # To normalize the difference to [0, 1]
            reward += 1 - abs(difference) / value_range
        elif isinstance(p, CategoricalValue):
            reward += 1 if optimal_parameters[p.name] == predicted_parameters[p.name] else 0

    # Although not required, we normalize the reward to [0, 1].
    # A reward close to 0 means the predicted_parameters were quite bad for the app and resulted in poor performance.
    # Whereas a reward close to 1 means the parameters converged to the optimal values, providing optimal performance.
    reward /= len(parameters)
    return reward


def main():
    # The application we want to tune.
    # Assume we are tuning the parameters for Azure Kubernetes Service (AKS) Cluster Autoscaler and we want to
    # find the optimal configuration for our chosen subset of parameters to get the best scale-up time.
    app = Application()

    # We select the following subset of parameters to tune (parameters selected from
    # https://learn.microsoft.com/en-us/azure/aks/cluster-autoscaler?tabs=azure-cli#use-the-cluster-autoscaler-profile)
    parameters_to_tune = [
        ContinuousValue(
            name="max-total-unready-percentage",
            initial_value=0.45,
            lb=0.0,
            ub=1.0,
            step_size=None,
        ),
        DiscreteValue(
            name="scan-interval",
            initial_value=10,  # seconds
            lb=2,
            ub=300,
            step_size=2,
        ),
        CategoricalValue(
            name="expander",
            initial_value="random",
            categories=("random", "least-waste", "most-pods", "priority"),
        ),
    ]

    # Initialize an instance of OPPerTune which will be used to tune the parameters
    tuner = OPPerTune(
        parameters_to_tune,
        algorithm="hybrid_solver",  # Supports continuous, discrete and categorical parameters
        algorithm_args=dict(
            numerical_solver="bluefin",  # For the numerical (continuous and discrete) parameters
            numerical_solver_args=dict(
                feedback=2,
                eta=0.01,
                delta=0.1,
                random_seed=123,  # Just for reproducibility
            ),
            categorical_solver="exponential_weights_slates",  # For the categorical parameters
            categorical_solver_args=dict(
                random_seed=123,  # Just for reproducibility
            ),
        ),
    )

    iteration = 0
    while True:
        # Predict the next set of perturbed parameters
        predicted_parameter_values, _metadata = tuner.predict()
        print(f"[{iteration}] Prediction: {predicted_parameter_values}, ", end="")

        # Deploy the app with the new parameters and observe the metrics of interest
        metrics = deploy_and_get_metrics(app, predicted_parameter_values)  # noqa: F841

        # Calculate the reward based on the observed metrics (the better the metrics, the higher the reward).
        # Note: Since we are using a dummy application, we approximate the reward by calculating how far are
        # the predicted parameters from the optimal parameters. Here, we assume that we somehow know the
        # optimal parameters for the application. But, in a real-world scenario we, of course, would not
        # know the optimal parameters and will instead use the metrics alone to calculate the reward.

        # reward = calculate_reward(metrics)  # For an actual application
        reward = calculate_reward(predicted_parameter_values, parameters_to_tune)  # Our approximation
        print(f"Reward: {reward}")

        # Send the feedback to OPPerTune for the gradient update
        tuner.set_reward(reward, metadata=_metadata)

        iteration += 1

        # Stop when we find a good enough configuration
        if reward >= 0.95:
            break


if __name__ == "__main__":
    main()
