import random
from typing import Any, Dict, List, Mapping, NamedTuple, Union

from oppertune.algorithms.autoscope import AutoScope, AutoScopeFeature
from oppertune.core.types import Context
from oppertune.core.values import Integer, Real

# The types of parameters we will tune in this example.
ParameterTypes = Union[Integer, Real]


class Application:
    """Dummy class representing the application of interest (whose configuration parameters we want to tune)."""

    def set_parameters(self, parameters: Mapping[str, Any]):
        """Set the configuration parameters of the app which it will use in its next run."""
        ...

    def run(self):
        """Run the app and wait for it to complete."""
        ...


class Metrics:
    """Dummy class representing how a service stores its captured metrics."""


class MetricsMonitor:
    """Dummy class representing a service that captures metrics for the application."""

    def start(self):
        """Start capturing metrics in the background."""
        ...

    def stop(self) -> Metrics:
        """Stop the capture and return the captured metrics."""
        ...


def deploy_and_get_metrics(app: Application, parameters: Mapping[str, Any]) -> Metrics:
    """Configure the app with the supplied parameters, run it and return the metrics observed."""
    monitor = MetricsMonitor()
    monitor.start()

    app.set_parameters(parameters)
    app.run()

    metrics = monitor.stop()
    return metrics


def calculate_reward(predicted_parameters: Mapping[str, Any], parameters: List[ParameterTypes], jobtype: str) -> float:
    # Since this is a dummy example, we assume that we know the optimal parameters
    # for the application and calculate how far are we from these.
    optimal_parameters_per_jobtype = {
        "0": {
            "max-total-unready-percentage": 0.125,
            "scan-interval": 10,
        },
        "1": {
            "max-total-unready-percentage": 0.325,
            "scan-interval": 60,
        },
        "2": {
            "max-total-unready-percentage": 0.525,
            "scan-interval": 150,
        },
        "3": {
            "max-total-unready-percentage": 0.925,
            "scan-interval": 290,
        },
    }

    reward: float = 0.0
    for p in parameters:
        difference = optimal_parameters_per_jobtype[jobtype][p.name] - predicted_parameters[p.name]
        value_range = p.max - p.min  # To normalize the difference to [0, 1]
        reward += 1 - abs(difference) / value_range

    # Although not required, we normalize the reward to [0, 1].
    # A reward close to 0 means the predicted_parameters were quite bad for the app and resulted in poor performance.
    # Whereas a reward close to 1 means the parameters converged to the optimal values, providing optimal performance.
    reward /= len(parameters)
    return reward


class TuningConfig(NamedTuple):
    reward: float
    parameters: Union[Mapping[str, Any], None] = None


def main() -> None:
    # The application we want to tune.
    # Assume we are tuning the parameters for Azure Kubernetes Service (AKS) Cluster Autoscaler and we want to
    # find the optimal configuration for our chosen subset of parameters to get the best scale-up time.
    app = Application()

    # We select the following subset of parameters to tune (parameters selected from
    # https://learn.microsoft.com/en-us/azure/aks/cluster-autoscaler?tabs=azure-cli#use-the-cluster-autoscaler-profile)
    parameters_to_tune = [
        Real("max-total-unready-percentage", val=0.45, min=0.0, max=1.0),
        Integer("scan-interval", val=10, min=2, max=300, step=2),  # seconds
    ]

    # AutoScope can be used to perform scoping and configuration tuning. For example, if a application has multiple job
    # types and each job type has a different set of optimal parameters, then we can use AutoScope to find the optimal
    # parameter for each jobtype. AutoScope uses a feature vector (supplied during predict) to perform scoping.
    jobtype_values = ("0", "1", "2", "3")
    features = [
        AutoScopeFeature(name="jobtype", values=jobtype_values),
    ]

    internal_weights = None

    # If you are already aware of a good starting scope, you can set the initial internal weights of the AutoScope tree.
    # Otherwise just leave it as None. The internal weights represent the weights of each internal node in the tree.
    # AutoScope is an oblique tree. This means that each node sees all the features and can split on any feature.
    # A 1 at an index indicates that if that index is set, go to the left child, a -1 to the right child.
    # A 0 ignores that index.
    internal_weights = [
        [1, 1, -1, -1],  # Root
        [1, -1, 0, 0],  # Left child of root
        [0, 0, 1, -1],  # Right child of root
    ]

    random_seed = 4

    # Initialize an instance of OPPerTune which will be used to tune the parameters
    tuning_instance = AutoScope(
        parameters_to_tune,
        features=features,  # The features that are used for scoping
        height=2,  # The height of the tree, determines the number of leaves. Number of leaves = 2^height
        leaf_algorithm_args={
            "feedback": 2,  # Onepoint or twopoint feedback
            "eta": 0.01,  # eta for the leaf weights, higher learning rate increases rate of convergence of parameters
            "delta": 0.1,
            "optimizer": "rmsprop",
            "random_seed": random_seed,  # For reproducibility
        },
        eta=0.001,  # eta for the internal weights, higher learning rate increases the rate of scoping changes
        delta=0.1,
        optimizer="rmsprop",
        random_seed=random_seed,  # For reproducibility
        internal_weights=internal_weights,
        fix_internal_weights=True,  # Set to False if you want the scoping to be learnt
    )

    rng = random.Random(4)  # Seeded random number generator (for reproducibility)

    # Storing the best configurations for each jobtype
    best_config_for_jobtype: Dict[str, TuningConfig] = {}
    for jobtype in jobtype_values:
        best_config_for_jobtype[jobtype] = TuningConfig(reward=0.0, parameters=None)

    iteration = 0
    while True:
        # Get information about the current jobtype for the application
        cur_jobtype = rng.choice(jobtype_values)
        context = Context({"jobtype": cur_jobtype})

        # Predict the next set of perturbed parameters
        prediction, request_id = tuning_instance.predict(context)
        print(f"[{iteration}] Jobtype: {cur_jobtype}, Prediction: {prediction}, ", end="")

        # Deploy the app with the new parameters and observe the metrics of interest
        metrics = deploy_and_get_metrics(app, prediction)  # noqa: F841

        # Calculate the reward based on the observed metrics (the better the metrics, the higher the reward).
        # Note: Since we are using a dummy application, we approximate the reward by calculating how far are
        # the predicted parameters from the optimal parameters. Here, we assume that we somehow know the
        # optimal parameters for the application. But, in a real-world scenario we, of course, would not
        # know the optimal parameters and will instead use the metrics alone to calculate the reward.

        # reward = calculate_reward(metrics)  # For an actual application
        reward = calculate_reward(prediction, parameters_to_tune, cur_jobtype)  # Our approximation for this example
        print(f"Reward: {reward}")

        # Store the reward for the request ID (useful for aggregating multiple rewards before calling set_reward)
        tuning_instance.store_reward(request_id, reward)

        # Send the feedback to OPPerTune for the gradient update
        tuning_instance.set_reward(reward, context_id=context.id)

        iteration += 1

        if reward > best_config_for_jobtype[cur_jobtype].reward:
            best_config_for_jobtype[cur_jobtype] = TuningConfig(reward, prediction)

        # Stop when we find a good enough configuration for each jobtype
        if all(best_config_for_jobtype[jobtype].reward >= 0.95 for jobtype in jobtype_values):
            break

    print("\nBest configurations for each jobtype:")
    for jobtype in jobtype_values:
        reward, parameters = best_config_for_jobtype[jobtype]
        print(f"  Jobtype: {jobtype}, Reward: {reward}, Parameters: {parameters}")


if __name__ == "__main__":
    main()
