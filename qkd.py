import numpy as np
from scipy import linalg, sparse
import random

NUM_QUBITS = 100

Z_BASE = 0
X_BASE = 1
X_GATE = np.array([[0, 1], [1, 0]])
Z_GATE = np.array([[1, 0], [0, -1]])
H_GATE = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
COMP_BASE_STATES = [np.array([1, 0]), np.array([0, 1])]


def compact_complex_repr(num: complex) -> str:
    if np.isreal(num):
        if num.real == 1:
            return ""
        if num.real == -1:
            return "-"
        return format(num.real, "g")
    if num.imag == 1:
        return "j"
    if num.imag == -1:
        return "-j"
    return num


def main():
    qubit = Qubit()
    print(qubit.measure(H_GATE))
    print(qubit)


class Qubit:
    def __init__(self) -> None:
        self._state = np.array([1, 0], dtype=complex)  # |0> state

    def __str__(self) -> str:
        ket_repr = ""
        if abs(self._state[0]) != 0:
            coef_0 = compact_complex_repr(self._state[0])
            ket_repr += f"{coef_0}|0>"
        if abs(self._state[1]) != 0:
            coef_1 = compact_complex_repr(self._state[1])
            ket_repr += " + " if ket_repr != "" else ""
            ket_repr += f"{coef_1}|1>"
        return ket_repr

    def apply_gate(self, gate: np.ndarray):
        self._state = gate @ self._state

    def measure(self, base: np.ndarray = None) -> int:ß
        if base is not None:
            self._state = base.conj().T @ self._state
        prob = np.square(np.abs(self._state))
        res = random.choices([0, 1], weights=prob)[0]
        self._state = COMP_BASE_STATES[res]
        if base is not None:
            self._state = base @ self._state
        return res


class Alice:
    def __init__(self) -> None:
        self._rnd = np.random.default_rng()

    def generate_key(self):
        self._key = self._rnd.integers(2, size=NUM_QUBITS)
        return self._key

    def generate_quantum_state_for_bob(self) -> np.ndarray:
        self._encoding_base = self._rnd.integers(2, size=NUM_QUBITS)
        quantum_state = []
        for i in range(NUM_QUBITS):
            base = self._encoding_base[i]
            key_bit = self._key[i]
        return quantum_state


class Bob:
    def __init__(self) -> None:
        self._rnd = np.random.default_rng()


def qkd():
    pass


def cascade(noisy_key, qber):
    pass


if __name__ == "__main__":
    main()
