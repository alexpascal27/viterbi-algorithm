from typing import Tuple

import numpy
import numpy as np


class Viterbi:
    calculated_values = None
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
        self.calculated_values = np.zeros((len(self.S), len(observation_sequence)))
        for i in range(len(start_values)):
            self.calculated_values[i][0] = start_values[i]
        self.sequence = numpy.chararray((1, len(observation_sequence)))
        self.sequence[0][0] = self.S[self.__get_state_from_max_list(start_values)[1]]

        # go through sequence ignoring the start char
        for i in range(1, len(observation_sequence)):
            for s in range(len(self.S)):
                max_list = np.zeros(len(self.S), dtype=float)
                max_list_str = ""
                # form the list that we use in the max
                # for each state
                for state in range(len(self.S)):
                    max_list[state] = self.calculated_values[state][i - 1] + self.T[state][s]
                    max_list_str += "(P_" + self.S[state] + "(" + self.Sigma[observation_sequence[i]] + ", " + str(i-1) + "){" + str(self.calculated_values[state][i - 1]) + "} + P_" + self.S[state] + self.S[s] + "{" + str(self.T[state][s]) + "}):{" + str(max_list[state]) + "}"
                    if i < len(observation_sequence) - 1:
                        max_list_str += ", "

                # once we have the max list choose a max and note down what state we chose
                max_state, state = self.__get_state_from_max_list(max_list)
                self.sequence[0][i] = self.S[state]

                self.calculated_values[s][i] = self.E[s][observation_sequence[i]] + max_state
                print()
                print("P_" + self.S[s] + "(" + self.Sigma[observation_sequence[i]] + ", " + str(i) + ") = e_" + self.S[s] + "(" + self.Sigma[observation_sequence[i]] + "):{" + str(self.E[s][observation_sequence[i]]) + "} + max(" + max_list_str + "):{" + str(max_state) + "}")
                print()

    def __get_state_from_max_list(self, max_list: np.array) -> Tuple[float, int]:
        #print(max_list)
        max_state = np.max(max_list)
        state = 0
        for i in range(len(max_list)):
            if max_list[i] == max_state:
                state = i
                break
        return max_state, state
