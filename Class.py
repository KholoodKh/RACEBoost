from typing import Tuple, Union
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score
from typing import Union
import numpy as np
from Discretization import *
from enum import Enum
import gc
import networkx as nx

class Algorithm(Enum):
    RACER = "RACER"
    RACERS = "RACERS"

    RRA = "RRa"
    RRD = "RRd"
    RRA_S = "RRa,s"
    RRD_S = "RRd,s"

    ROPAC_M = "ROPAC-M"
    ROPAC_L = "ROPAC-L"

class ROPACPreprocesser:
    def __init__(self, framework="numpy"):
        assert framework in [
            "numpy",
            "torch",
        ], "framework must be either 'numpy' or 'torch'"
        self.framework = framework

    def fit_transform(
        self, data: Union[pd.DataFrame, np.ndarray], dataTypes, classIndex
    ):
        """Preprocesses the dataset by replacing nominal values with dummy variables.
        Converts to torch bool tensors and returns the dataset. All numerical values are discretized
        into equal-sized bins using a quantile-based method (pd.qcut).
        Args:
            X (pandas.DataFrame or np.ndarray): features vector
            y (pandas.DataFrame or np.ndarray): targets vector
        Returns:
            X (torch.Tensor or np.ndarray): features vector
            y (torch.Tensor or np.ndarray): targets vector
        """

        y = data[:, classIndex]
        if classIndex == 0:
            X = data[:, 1:]
        else:
            X = data[:, :-1]
        #numerics = X.select_dtypes(include=[np.number]).columns.tolist()

        numAttrIndex = [i for i, t in enumerate(dataTypes[:classIndex]) if t == 'numeric' or t == 'integer' or t == 'real']

        if numAttrIndex:
            #X[numerics] = pd.qcut(X[numerics], bins=self.bins)
            for num in numAttrIndex:
                newData = Discretization.basedOnEntropy(data[:,[num,classIndex]])
                x_index = num if num < classIndex else num - 1
                X[:, x_index] = newData

        X = pd.get_dummies(pd.DataFrame(X)).to_numpy()
        y = pd.get_dummies(y).to_numpy()
        if self.framework == "torch":
            import torch

            X, y = torch.tensor(X, dtype=torch.bool), torch.tensor(y, dtype=torch.bool)
        return X, y

    def matrix_to_list(adj_matrix):
            edges = []
            rows, cols = np.where(adj_matrix > 0)
    
            for i in range(len(rows)):
                weight = adj_matrix[rows[i]][cols[i]]
                edges.append((rows[i], cols[i], {'weight': weight} ))
            gc.collect()
            return edges


from numpy import (
    logical_and as AND,
    logical_not as NOT,
    logical_or as OR,
    logical_xor as XOR,
)


def XNOR(input: np.ndarray, other: np.ndarray) -> np.ndarray:
    """Computes the XNOR gate. (semantically the same as `input == other`)
    Args:
        input (np.ndarray): Input tensor
        other (np.ndarray): Other input tensor
    Returns:
        np.ndarray: Output tensor
    """
    return NOT(XOR(input, other)).astype(bool)


class ROPAC:
    def __init__(
        self,
        alpha=0.99,
        fitness_fn="weighted_average",
        gamma=0.6,
        suppress_warnings=False,
        benchmark=False,
    ):
        """
        Initialize the ROPAC class.

        Args:
            alpha (float, optional): Value of alpha according to the ROPAC paper.
            fitness_fn (str): Fitness function:
                - "weighted_average"
                - "weighted"
                - "f-beta"
            gamma (float, optional): Weight given to coverage score during closest_matching.
            suppress_warnings (bool, optional): Whether to suppress prediction warnings.
            benchmark (bool, optional): Whether to benchmark fit time.
        """

        self._alpha = alpha
        self._beta = 1 - alpha
        self._gamma = gamma

        fitness_functions = {
            "weighted_average": self._fitness_weighted_avg,
            "weighted": self._fitness_weighted,
            "f-beta": self._fitness_f_beta,
        }

        if fitness_fn not in fitness_functions:
            raise ValueError(
                f"fitness_fn must be one of {list(fitness_functions.keys())}"
            )

        self._fitness_fn = fitness_functions[fitness_fn]

        self._fitness_fn_name = {
            "weighted_average": "Weighted Average",
            "weighted": "Weighted",
            "f-beta": "F-Beta",
        }[fitness_fn]

        self._suppress_warnings = suppress_warnings
        self._benchmark = benchmark

        self._has_fit = False

        self.cardinality = []
        self.weights = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        algorithm: Algorithm = Algorithm.RACER,
    ) -> None:
        """
        Fit the selected rule-based classification algorithm.

        Algorithms:
            RACER    : Original RACER.
            RACERs   : RACER with early stopping after a successful merge.
            RRa      : RACE-RANK with ascending rule ranking.
            RRd      : RACE-RANK with descending rule ranking.
            RRa,s     : RACE-RANK ascending + early stopping.
            RRd,s     : RACE-RANK descending + early stopping.
            ROPAC-M   : ROPAC using MFZB.
            ROPAC-L   : ROPAC using LFZB.

        Args:
            X: Feature matrix.
            y: Target vector.
            algorithm: Algorithm configuration to use.
        """

        if self._benchmark:
            from time import perf_counter
            tic = perf_counter()

        # =========================================================
        # Store training data
        # =========================================================
        self._X = X
        self._y = y

        self._cardinality, self._rule_len = X.shape
        self._classes = np.unique(y, axis=0)

        self._class_indices = {
            self._label_to_int(cls): np.where(
                np.min(XNOR(y, cls), axis=-1)
            )[0]
            for cls in self._classes
        }

        # =========================================================
        # Create initial rules
        # =========================================================
        self._create_init_rules()

        # =========================================================
        # ROPAC initial search
        # =========================================================
        if algorithm == Algorithm.ROPAC_M:
            self.MFZB(
                self._extants_if,
                self._extants_then
            )

        elif algorithm == Algorithm.ROPAC_L:
            self.LFZB(
                self._extants_if,
                self._extants_then
            )

        # =========================================================
        # Generalize rules
        # =========================================================
        self._generalize_extants()

        # =========================================================
        # Process rules class by class
        # =========================================================
        for cls in self._class_indices.keys():

            indices = self._class_indices[cls]

            # Determine rule processing order
            rule_iterator = self._get_rule_order(
                indices,
                algorithm
            )

            # Early stopping is used only by:
            # RACERs, RRa,s, RRd,s
            stop_merging = algorithm in {
                Algorithm.RACERS,
                Algorithm.RRA_S,
                Algorithm.RRD_S,
            }

            # -----------------------------------------------------
            # Pairwise rule processing
            # -----------------------------------------------------
            for i in range(len(rule_iterator)):

                for j in range(i + 1, len(rule_iterator)):

                    composed = self._process_rules(
                        rule_iterator[i],
                        rule_iterator[j]
                    )

                    if stop_merging and composed:
                        break

        # =========================================================
        # Keep only independent rules
        # =========================================================
        independent_indices = ~self._extants_covered

        self._extants_if = (
            self._extants_if[independent_indices]
        )

        self._extants_then = (
            self._extants_then[independent_indices]
        )

        self._fitnesses = (
            self._fitnesses[independent_indices]
        )

        # =========================================================
        # Final generalization
        # =========================================================
        self._generalize_extants()

        # =========================================================
        # Sort final rules by fitness
        # =========================================================
        args = np.argsort(self._fitnesses)[::-1]

        self._final_rules_if = self._extants_if[args]
        self._final_rules_then = self._extants_then[args]
        self._fitnesses = self._fitnesses[args]

        # =========================================================
        # Mark model as fitted
        # =========================================================
        self._has_fit = True

        if self._benchmark:
            self._bench_time = perf_counter() - tic

    def _get_rule_order(
        self,
        indices: np.ndarray,
        algorithm: Algorithm,
    ) -> np.ndarray:
        """
        Determine the order in which rules are processed.

        RACER and RACERs:
            Original rule order.

        RRa / RRa,s:
            RACE-RANK ascending PageRank order.

        RRd / RRd,s:
            RACE-RANK descending PageRank order.

        ROPAC-M / ROPAC-L:
            Original order (their ordering is handled by MFZB/LFZB).
        """

        # =========================================================
        # Algorithms without additional rule ordering
        # =========================================================
        if algorithm in {
            Algorithm.RACER,
            Algorithm.RACERS,
            Algorithm.ROPAC_M,
            Algorithm.ROPAC_L,
        }:
            return indices

        # =========================================================
        # RACE-RANK
        # =========================================================
        if algorithm in {
            Algorithm.RRA,
            Algorithm.RRD,
            Algorithm.RRA_S,
            Algorithm.RRD_S,
        }:

            # No ranking needed for one or zero rules
            if len(indices) <= 1:
                return indices

            current_list = self._X[indices]
            num_rules = len(indices)

            # -----------------------------------------------------
            # Pairwise intersection matrix
            # -----------------------------------------------------
            intersection_matrix = np.zeros(
                (num_rules, num_rules)
            )

            for i in range(num_rules):
                for j in range(i + 1, num_rules):

                    intersection = np.sum(
                        current_list[i] & current_list[j]
                    )

                    intersection_matrix[i, j] = intersection
                    intersection_matrix[j, i] = intersection

            # -----------------------------------------------------
            # Convert intersection matrix to graph
            # -----------------------------------------------------
            edges = ROPACPreprocesser.matrix_to_list(
                intersection_matrix
            )

            G = nx.DiGraph()
            G.add_edges_from(edges)

            # -----------------------------------------------------
            # Calculate PageRank
            # -----------------------------------------------------
            network_values = nx.pagerank(
                G,
                weight="weight"
            )

            # -----------------------------------------------------
            # Ascending / descending ranking
            #
            # RRa / RRa,s -> ascending
            # RRd / RRd,s -> descending
            # -----------------------------------------------------
            descending = algorithm in {
                Algorithm.RRD,
                Algorithm.RRD_S,
            }

            sorted_nodes = sorted(
                network_values.items(),
                key=lambda item: item[1],
                reverse=descending
            )

            # Convert graph node positions back to dataset indices
            rule_iterator = np.array([
                indices[node]
                for node, _ in sorted_nodes
            ])

            return rule_iterator

        raise ValueError(
            f"Unsupported algorithm: {algorithm}"
        )

    #This is the implementation of the rule optimization function in ROPAC-M
    def MFZB(self, base_array, base_result):
        # Count the number of 1s in each bit position across all rules
        bit_counts = [sum(1 for rule in base_array if rule[i]) for i in range(len(base_array[0]))]

        for i in range(len(base_array)):
            temp_result = base_array[i]
            temp_fitness = self._fitnesses[i]
            false_indices = [idx for idx, value in enumerate(base_array[i]) if not value]

            if not false_indices:
                continue

            # Sort the false indices based on the number of 1s in the corresponding bit position
            sorted_indices = sorted(false_indices, key=lambda idx: bit_counts[idx], reverse=True)

            for index in sorted_indices:
                new_array = base_array[i].copy()
                new_array[index] = True
                fitness = self._fitness_fn(new_array, base_result[i])

                if fitness > temp_fitness:
                    temp_fitness = fitness
                    temp_result = new_array
                    break  # Stop iterating if a better fitness is found

            self._extants_if[i] = temp_result
            self._fitnesses[i] = temp_fitness

    #This is the implementation of the rule optimization function in ROPAC-L
    def LFZB(self, base_array, base_result):
        # Count the number of 1s in each bit position across all rules
        bit_counts = [sum(1 for rule in base_array if rule[i]) for i in range(len(base_array[0]))]

        for i in range(len(base_array)):
            temp_result = base_array[i]
            temp_fitness = self._fitnesses[i]
            false_indices = [idx for idx, value in enumerate(base_array[i]) if not value]

            if not false_indices:
                continue

            # Sort the false indices based on the number of 1s in the corresponding bit position
            sorted_indices = sorted(false_indices, key=lambda idx: bit_counts[idx])

            for index in sorted_indices:
                new_array = base_array[i].copy()
                new_array[index] = True
                fitness = self._fitness_fn(new_array, base_result[i])

                if fitness > temp_fitness:
                    temp_fitness = fitness
                    temp_result = new_array
                    break  # Stop iterating if a better fitness is found

            self._extants_if[i] = temp_result
            self._fitnesses[i] = temp_fitness

    def predict(self, X: np.ndarray, convert_dummies=True) -> np.ndarray:
        """Given input X, predict label using ROPAC
        Args:
            X (np.ndarray): input features vector
            convert_dummies (bool): whether to convert the output to a one-dimensional array
        Returns:
            np.ndarray: label as predicted by ROPAC
        """
        assert self._has_fit, "ROPAC has not been fit yet."
        labels = np.zeros((len(X), self._final_rules_then.shape[1]), dtype=bool)
        found = np.zeros(len(X), dtype=bool)
        for i in range(len(self._final_rules_if)):
            covered = self._covered(X, self._final_rules_if[i])
            labels[AND(covered, NOT(found))] = self._final_rules_then[i]
            found[covered] = True
            if found.sum() == len(X):  # -> every instance was matched to a rule
                break

        all_found = found.sum() == len(X)
        if not all_found:
            print(
                f"Warning: ROPAC was unable to find a perfect match for {len(X) - found.sum()} instances out of {len(X)}."
            )
            print(
                "Labels for these instances will be determined by a closest match algorithm."
            )
            leftover_indices = np.where(~found)[0]
            for idx in leftover_indices:
                labels[idx] = self._closest_match(X[idx])

        if convert_dummies:
            labels = np.argmax(labels, axis=-1)

        return labels

    def display_rules(self):
        """Print out the final rules"""
        assert self._has_fit, "ROPAC has not been fit yet."
        print("Algorithm Parameters:")
        print(f"\t- Fitness Function: {self._fitness_fn_name}")
        print(f"\t- Alpha: {self._alpha}")
        print(f"\t- Gamma: {self._gamma}")
        if self._benchmark:
            print(f"\t- Compute Device: CPU")
            print(f"\t- Time to fit: {self._bench_time}s")
        print(
            f"\nFinal Rules ({len(self._final_rules_if)} total): (if --> then (label) | fitness)"
        )
        for i in range(len(self._final_rules_if)):
            print(
                f"\t{np.array2string(self._final_rules_if[i].astype(int), separator='')} -->"
                f" {np.array2string(self._final_rules_then[i].astype(int), separator='')}"
                f" ({np.argmax(self._final_rules_then[i].astype(int))})"
                f" | {self._fitnesses[i]}"
            )

    def _closest_match(self, X: np.ndarray) -> np.ndarray:
        """Find the closest matching rule to `X`
        Args:
            X (np.ndarray): input `X`
        Returns:
            np.ndarray: matched rule
        """
        int_X = X.astype(int)  # <- cast boolean array to integer array
        overlap = OR(NOT(int_X), AND(self._final_rules_if, int_X)).sum(axis=-1)
        overlap = overlap / self._rule_len  # -> normalize by rule length
        scores = np.multiply(self._gamma * overlap, (1 - self._gamma) * self._fitnesses)
        argmax = np.argmax(scores)
        return self._final_rules_then[argmax]

    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Returns accuracy on the provided test data
        Args:
            X_test (np.ndarray): test features vector
            y_test (np.ndarray): test targets vector
        Returns:
            float: accuracy score
        """
        assert self._has_fit, "ROPAC has not been fit yet."
        X_test, y_test = X_test.astype(np.float32), y_test.astype(np.float32)
        if y_test.ndim != 1 and y_test.shape[1] != 1:
            y_test = np.argmax(y_test, axis=-1)
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)

    def _fitness_weighted_avg(
        self, rule_if: np.ndarray, rule_then: np.ndarray
    ) -> float:
        """Returns fitness for a given rule according to the ROPAC paper
        Args:
            rule_if (np.ndarray): if part of a rule (x)
            rule_then (np.ndarray): then part of a rule (y)
        Returns:
            float: fitness score for the rule as defined in the ROPAC paper
        """

        n_covered, n_correct = self._confusion(rule_if, rule_then)
        if n_covered != 0 :
            accuracy = n_correct / n_covered
            coverage = n_covered / self._cardinality
        else :
            accuracy = 0
            coverage = 0

        return self._alpha * accuracy + self._beta * coverage

    def _fitness_weighted(self, rule_if: np.ndarray, rule_then: np.ndarray) -> np.ndarray:
        """Returns fitness for a given rule according to the ROPAC paper
        Args:
            rule_if (np.ndarray): if part of a rule (x)
            rule_then (np.ndarray): then part of a rule (y)
        Returns:
            float: fitness score for the rule
        """
        n_covered, n_correct = self._confusion_weighted(rule_if, rule_then)
        if n_covered != 0 :
            accuracy = n_correct / n_covered
            coverage = n_covered / self._weights.sum()
        else :
            accuracy = 0
            coverage = 0

        return self._alpha * accuracy + self._beta * coverage

    def _fitness_f_beta(self, rule_if: np.ndarray, rule_then: np.ndarray) -> np.ndarray:
        """Returns f-beta-score fitness for a given rule
        Args:
            rule_if (np.ndarray): if part of a rule (x)
            rule_then (np.ndarray): then part of a rule (y)
        Returns:
            np.ndarray: f-beta-score fitness for the rule
        """
        # `beta` in f-beta-score is chosen such that recall is considered `beta` times as important as precision
        # https://en.wikipedia.org/wiki/F-score
        beta = self._beta / self._alpha
        n_covered, n_correct = self._confusion(rule_if, rule_then)
        accuracy = n_correct / n_covered
        coverage = n_covered / self._cardinality

        return (
            (1 + beta**2) * (accuracy * coverage) / (beta**2 * accuracy + coverage)
        )

    def _covered(self, X: np.ndarray, rule_if: np.ndarray) -> np.ndarray:
        """Returns indices of instances in `X` that are covered by `rule_if`.
        Note that rule covers instance if EITHER of the following holds in a bitwise manner:
        1. instance[i] == 0
        2. instance[1] == 1 AND rule[i] == 1
        Args:
            X (np.ndarray): instances
            rule_if (np.ndarray): if part of rule (x)
        Returns:
            np.ndarray: An array containing indices of instances in `X_same_class` that are covered by `rule_if`
        """
        covered = OR(NOT(X[:self._cardinality,:]), AND(rule_if, X[:self._cardinality,:])).min(axis=-1)
        return covered

    def _confusion_weighted(
        self, rule_if: np.ndarray, rule_then: np.ndarray
    ) -> Tuple[float, float]:
        """Returns weighted n_covered and n_correct for instances classified by a rule.
        Args:
            rule_if (np.ndarray): if part of rule (x)
            rule_then (np.ndarray): then part of rule (y)
        Returns:
            Tuple[float, float]: (n_covered, n_correct)
        """

        # which instances are covered by rule_if
        covered = self._covered(self._X, rule_if)
        covered_mask = np.zeros(self._cardinality, dtype=bool)
        covered_mask[:self._cardinality] = covered

        # which covered instances are correctly classified by rule_then
        y_covered = self._y[covered_mask]
        correct = XNOR(y_covered, rule_then).min(axis=-1)

        correct_mask = np.zeros(self._cardinality, dtype=bool)
        correct_mask[covered_mask] = correct

        # weighted sums
        n_covered = self._weights[covered_mask].sum()
        n_correct = self._weights[correct_mask].sum()

        return n_covered, n_correct

    def _confusion(
        self, rule_if: np.ndarray, rule_then: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns n_correct and n_covered for instances classified by a rule.
        Args:
            rule_if (np.ndarray): if part of rule (x)
            rule_then (np.ndarray): then part of rule (y)
        Returns:
            Tuple[np.ndarray, np.ndarray]: (n_covered, n_correct)
        """
        covered = self._covered(self._X, rule_if)
        covered_mask = np.zeros(self._cardinality, dtype=bool)
        covered_mask[:self._cardinality] = covered
        n_covered = covered.sum()
        y_covered = self._y[covered_mask]
        n_correct = XNOR(y_covered, rule_then).min(axis=-1).sum()
        return n_covered, n_correct

    def _create_init_rules(self):
        """Creates an initial set of rules from the input feature vectors"""
        self._extants_if = self._X.copy()
        self._extants_then = self._y.copy()
        self._extants_covered = np.zeros(len(self._X), dtype=bool)
        self._fitnesses = np.array(
            [
                self._fitness_fn(rule_if, rule_then)
                for rule_if, rule_then in zip(self._X, self._y)
            ]
        )

    def _composable(self, idx1: int, idx2: int) -> bool:
        """Returns true if two rules indicated by their indices are composable
        Args:
            idx1 (int): index of the first rule
            idx2 (int): index of the second rule
        Returns:
            bool: True if labels match and neither of the rules are covered. False otherwise.
        """
        labels_match = XNOR(self._extants_then[idx1], self._extants_then[idx2]).min() # This line is not nessecary since we already check this
        return (
            labels_match
            and not self._extants_covered[idx1].all()
            and not self._extants_covered[idx2].all()
        )

    def _process_rules(self, idx1: int, idx2: int):
        """Process two rules indicated by their indices
        Args:
            idx1 (int): index of the first rule
            idx2 (int): index of the second rule
        """
        if self._composable(idx1, idx2):
            composition = self._compose(self._extants_if[idx1], self._extants_if[idx2])
            composition_fitness = self._fitness_fn(
                composition, self._extants_then[idx1]
            )

            if composition_fitness > np.maximum(
                self._fitnesses[idx1], self._fitnesses[idx2]
            ):
                self._update_extants(
                    idx1, composition, self._extants_then[idx1], composition_fitness
                )

    def _compose(self, rule1: np.ndarray, rule2: np.ndarray) -> np.ndarray:
        """Composes rule1 with rule2
        Args:
            rule1 (np.ndarray): the first rule
            rule2 (np.ndarray): the second rule
        Returns:
            np.ndarray: The composed rule which is simply the bitwise OR of the two rules
        """
        return OR(rule1, rule2)

    def _update_extants(
        self,
        index: int,
        new_rule_if: np.ndarray,
        new_rule_then: np.ndarray,
        new_rule_fitness: np.ndarray,
    ):
        """Remove all rules from current extants that are covered by `new_rule`.
        Then append new rule to extants.
        Args:
            index (int): index of `new_rule`
            new_rule_if (np.ndarray): if part of `new_rule` (x)
            new_rule_then (np.ndarray): then part of `new_rule` (y)
            new_rule_fitness (np.ndarray): fitness of the `new_rule`
        """
        same_class_indices = self._class_indices[self._label_to_int(new_rule_then)]
        covered = self._covered(self._extants_if[same_class_indices], new_rule_if)
        covered_mask = np.zeros(len(same_class_indices), dtype=bool)
        covered_mask[:self._cardinality] = covered
        self._extants_covered[same_class_indices[covered_mask]] = True
        self._extants_covered[index] = False  # -> except new rule from covered
        self._extants_if[index], self._extants_then[index], self._fitnesses[index] = (
            new_rule_if,
            new_rule_then,
            new_rule_fitness,
        )

    def _label_to_int(self, label: np.ndarray) -> int:
        """Converts dummy label to int
        Args:
            label (np.ndarray): label to convert
        Returns:
            int: converted label
        """
        return int(np.argmax(label))

    def _generalize_extants(self):
        """Generalize the extants by flipping every 0 to a 1 and checking if the fitness improves."""
        new_extants_if = np.zeros_like(self._extants_if, dtype=bool)
        for i in range(len(self._extants_if)):
            for j in range(len(self._extants_if[i])):
                if not self._extants_if[i][j]:
                    self._extants_if[i][j] = True
                    fitness = self._fitness_fn(
                        self._extants_if[i], self._extants_then[i]
                    )
                    if fitness > self._fitnesses[i]:
                        self._fitnesses[i] = fitness
                    else:
                        self._extants_if[i][j] = False
            new_extants_if[i] = self._extants_if[i]
        self._extants_if = new_extants_if

    def getNumOfRules(self):
        return len(self._final_rules_if)

    def reduceRules(self):
        assert self._has_fit, "ROPAC has not been fit yet."
        i = 0
        j = 0
        while(i < len(self._final_rules_if)-1):
            j = i+1
            coveredIndex = []
            while(j < len(self._final_rules_if)):
                covered = True
                for t in range(len(self._final_rules_if[j])):
                    if self._final_rules_if[j][t] and not self._final_rules_if[i][t]:
                        covered = False
                        break
                if covered :
                    coveredIndex.append(j)
                j += 1
            self._final_rules_if = np.delete(self._final_rules_if, coveredIndex, axis = 0)
            self._final_rules_then = np.delete(self._final_rules_then, coveredIndex, axis = 0)
            self._fitnesses = np.delete(self._fitnesses, coveredIndex, axis = 0)
            i += 1
