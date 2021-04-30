from typing import Tuple
import numpy as np
import math


class Viterbi:
    calculated_values = None
    calculated_max_values = None
    decisions = None
    sequence = None

    def __init__(self, S: list, T: np.array, pi: np.array, Sigma: list, E: np.array):
        self.S = S
        self.T = T
        self.pi = pi
        self.Sigma = Sigma
        self.E = E

    def comprehend_sequence(self, observation_sequence: list, start_values: list) -> None:
        if len(start_values) != len(self.S):
            print("Exception: start_values list has incorrect length")
            return

        # initialise the values we calculate
        self.calculated_values = np.zeros((len(self.S), len(observation_sequence)), dtype=np.double)
        self.calculated_max_values = np.zeros((len(self.S), len(observation_sequence)), dtype=np.double)
        self.decisions = np.zeros((len(self.S), len(observation_sequence)-1), dtype=int)
        for i in range(len(start_values)):
            self.calculated_values[i][0] = start_values[i]
        self.sequence = np.chararray((1, len(observation_sequence)), unicode=True)

        # go through sequence ignoring the start char
        for i in range(1, len(observation_sequence) + 1):
            for s in range(len(self.S)):
                max_list = np.zeros(len(self.S), dtype=np.double)
                # max_list_str = ""
                # form the list that we use in the max
                # for each state
                for state in range(len(self.S)):
                    max_list[state] = self.calculated_values[state][i - 1] + self.T[state][s]
                    """
                    max_list_str += "(P_" + self.S[state] + "(" + self.Sigma[observation_sequence[i]] + ", " + str(i-1) + "){" + str(self.calculated_values[state][i - 1]) + "} + P_" + self.S[state] + self.S[s] + "{" + str(self.T[state][s]) + "}):{" + str(max_list[state]) + "}"
                    if i < len(observation_sequence) - 1:
                        max_list_str += ", "
                    """

                # once we have the max list choose a max and note down what state we chose
                max_state, state_index = self.__get_state_from_max_list(max_list)

                if not i >= len(observation_sequence):
                    self.calculated_values[s][i] = np.double(self.E[s][observation_sequence[i]]) + max_state
                    self.decisions[s][i-1] = state_index

                self.calculated_max_values[s][i-1] = max_state

                # print()
                # print("P_" + self.S[s] + "(" + self.Sigma[observation_sequence[i]] + ", " + str(i) + ") = e_" + self.S[s] + "(" + self.Sigma[observation_sequence[i]] + "):{" + str(self.E[s][observation_sequence[i]]) + "} + max(" + max_list_str + "):{" + str(max_state) + "}")
                # print()

        self.compute_sequence()

    def compute_sequence(self) -> None:
        # check all decisions
        for sequence_index in range(len(self.calculated_max_values[0]) - 1, -1, -1):
            # check for the smallest value
            current_max_value = -math.inf
            max_value_index = 0
            for state_index in range(len(self.S)):
                value = self.calculated_max_values[state_index][sequence_index]
                if value >= current_max_value:
                    max_value_index = state_index
                    current_max_value = value
            self.sequence[0][sequence_index] = self.S[max_value_index]


    def __get_state_from_max_list(self, max_list: np.array) -> Tuple[float, int]:
        # print(max_list)
        max_state = np.max(max_list)
        state = 0
        for i in range(len(max_list)):
            if max_list[i] == max_state:
                state = i
                break
        return max_state, state
