import random
from typing import Any, Dict, List, NamedTuple, Union

from oppertune import ContinuousValue, DiscreteValue, OPPerTune

# The types of parameters we will tune in this example.
ParameterTypes = Union[ContinuousValue, DiscreteValue]


class Application:
    """Dummy class representing the application of interest (whose configuration parameters we want to tune)"""

    def set_parameters(self, parameters: Dict[str, Any]):
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


def calculate_reward(predicted_parameters: Dict[str, Any], parameters: List[ParameterTypes], jobtype: int) -> float:
    # Since this is a dummy example, we assume that we know the optimal parameters
    # for the application and calculate how far are we from these.
    optimal_parameters_per_jobtype = {
        0: {
            "max-total-unready-percentage": 0.125,
            "scan-interval": 10,
        },
        1: {
            "max-total-unready-percentage": 0.325,
            "scan-interval": 60,
        },
        2: {
            "max-total-unready-percentage": 0.525,
            "scan-interval": 150,
        },
        3: {
            "max-total-unready-percentage": 0.925,
            "scan-interval": 290,
        },
    }

    reward: float = 0.0
    for p in parameters:
        difference = optimal_parameters_per_jobtype[jobtype][p.name] - predicted_parameters[p.name]
        value_range = p.ub - p.lb  # To normalize the difference to [0, 1]
        reward += 1 - abs(difference) / value_range

    # Although not required, we normalize the reward to [0, 1].
    # A reward close to 0 means the predicted_parameters were quite bad for the app and resulted in poor performance.
    # Whereas a reward close to 1 means the parameters converged to the optimal values, providing optimal performance.
    reward /= len(parameters)
    return reward


class TuningConfig(NamedTuple):
    reward: float
    parameters: Union[Dict[str, Any], None] = None


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
    ]

    # DGT can be used to perform scoping and configuration tuning. For example, if a application has multiple job
    # types and each job type has a different set of optimal parameters, then we can use DGT to find the optimal
    # parameter for each jobtype. DGT uses a feature vector (supplied during predict) to perform scoping.
    jobtype_values = (0, 1, 2, 3)
    features = [
        {
            "name": "jobtype",
            "values": jobtype_values,
        },
    ]

    internal_weights = None

    # If you are already aware of a good starting scope, you can set the initial internal weights of the DGT tree.
    # Otherwise just leave it as None. The internal weights represent the weights of each internal node in the tree.
    # DGT is an oblique tree. This means that each node sees all the features and can split on any feature. A 1 at an
    # index indicates that if that index is set, go to the left child, a -1 to the right child. A 0 ignores that index.
    internal_weights = [
        [1, 1, -1, -1],  # Root
        [1, -1, 0, 0],  # Left child of root
        [0, 0, 1, -1],  # Right child of root
    ]

    # Initialize an instance of OPPerTune which will be used to tune the parameters
    tuner = OPPerTune(
        parameters_to_tune,
        algorithm="dgt",  # Supports continuous and discrete parameters
        algorithm_args=dict(
            features=features,  # The features that are used for scoping
            height=2,  # The height of the tree, determines the number of leaves. Number of leaves = 2^height
            feedback=2,  # Onepoint or twopoint feedback
            eta1=0.001,  # eta for the internal weights, higher learning rate increases the rate of scoping changes
            eta2=0.01,  # eta for the leaf weights, higher learning rate increases rate of convergence of parameters
            delta=0.1,
            random_seed=4,  # Just for reproducibility
            optimizer="rmsprop",
            internal_weights=internal_weights,
            fix_internal_weights=True,  # Set to False if you want the scoping to be learnt
            leaf_weights=None,  # Set to a list of leaf weights if you want to start with a specific set of initial parameters for each scope
        ),
    )

    rng = random.Random(4)  # Seeded random number generator (for reproducibility)

    # Storing the best configurations for each jobtype
    best_config_for_jobtype: Dict[int, TuningConfig] = {}
    for jobtype in jobtype_values:
        best_config_for_jobtype[jobtype] = TuningConfig(reward=0.0, parameters=None)

    iteration = 0
    while True:
        # Get information about the current jobtype for the application
        cur_jobtype = rng.choice(jobtype_values)
        cur_features = {"jobtype": cur_jobtype}

        # Predict the next set of perturbed parameters
        predicted_parameter_values, _metadata = tuner.predict(features=cur_features)
        print(f"[{iteration}] Jobtype: {cur_jobtype}, Prediction: {predicted_parameter_values}, ", end="")

        # Deploy the app with the new parameters and observe the metrics of interest
        metrics = deploy_and_get_metrics(app, predicted_parameter_values)  # noqa: F841

        # Calculate the reward based on the observed metrics (the better the metrics, the higher the reward).
        # Note: Since we are using a dummy application, we approximate the reward by calculating how far are
        # the predicted parameters from the optimal parameters. Here, we assume that we somehow know the
        # optimal parameters for the application. But, in a real-world scenario we, of course, would not
        # know the optimal parameters and will instead use the metrics alone to calculate the reward.

        # reward = calculate_reward(metrics)  # For an actual application
        reward = calculate_reward(predicted_parameter_values, parameters_to_tune, cur_jobtype)  # Our approximation
        print(f"Reward: {reward}")

        # Send the feedback to OPPerTune for the gradient update
        tuner.set_reward(reward, metadata=_metadata)

        iteration += 1

        if reward > best_config_for_jobtype[cur_jobtype].reward:
            best_config_for_jobtype[cur_jobtype] = TuningConfig(reward, predicted_parameter_values)

        # Stop when we find a good enough configuration for each jobtype
        if all(best_config_for_jobtype[jobtype].reward >= 0.95 for jobtype in jobtype_values):
            break

    print("\nBest configurations for each jobtype:")
    for jobtype in jobtype_values:
        reward, parameters = best_config_for_jobtype[jobtype]
        print(f"  Jobtype: {jobtype}, Reward: {reward}, Parameters: {parameters}")


if __name__ == "__main__":
    main()
