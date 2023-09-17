from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory
from math import ceil, pi
import random
import time
import numpy as np
from scipy import linalg, sparse
from scipy.stats import unitary_group

NUM_QUBITS = 10

Z_BASE = 0
X_BASE = 1
I_GATE = np.array([[1, 0], [0, 1]])
X_GATE = np.array([[0, 1], [1, 0]])
Z_GATE = np.array([[1, 0], [0, -1]])
H_GATE = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
COMP_BASE_STATES = [np.array([1, 0]), np.array([0, 1])]
X_BASE_STATES = [np.array([1, 1]) / np.sqrt(2), np.array([1, -1]) / np.sqrt(2)]


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
    qkd()


class Qubit:
    def __init__(self) -> None:
        self._state = np.array([1, 0], dtype=complex)  # |0> state

    def __str__(self) -> str:
        ket_repr = ""
        if (self._state == X_BASE_STATES[0]).all():
            return "|+>"
        if (self._state == X_BASE_STATES[1]).all():
            return "|->"
        if abs(self._state[0]) != 0:
            coef_0 = compact_complex_repr(self._state[0])
            ket_repr += f"{coef_0}|0>"
        if abs(self._state[1]) != 0:
            coef_1 = compact_complex_repr(self._state[1])
            if ket_repr != "":
                ket_repr += " + " if self._state[1] > 0 else " "
            ket_repr += f"{coef_1}|1>"
        return ket_repr

    def __repr__(self) -> str:
        return str(self)

    def apply_gate(self, gate: np.ndarray):
        self._state = gate @ self._state

    def measure(self, base: np.ndarray = None) -> int:
        if base is not None:
            self._state = base.conj().T @ self._state
        prob = np.square(np.abs(self._state))
        res = random.choices([0, 1], weights=prob)[0]
        self._state = COMP_BASE_STATES[res]
        if base is not None:
            self._state = base @ self._state
        return res


class Player:
    def __init__(self, seed=None):
        self._rnd = np.random.default_rng(seed)
        self._key = None
        self._agreed_key = None
        self._encoding_base = None
        self._quantum_state = None

    @property
    def key(self):
        return self._key

    @property
    def agreed_key(self):
        return self._agreed_key

    @agreed_key.setter
    def agreed_key(self, value: list):
        self._agreed_key = value

    @property
    def encoding_bases(self):
        return self._encoding_base

    @property
    def quantum_state(self):
        return self._quantum_state


class Alice(Player):
    def __init__(self, seed=None):
        Player.__init__(self, seed)
        self._quantum_state = [Qubit() for i in range(NUM_QUBITS)]

    def generate_key(self) -> np.ndarray:
        self._key = self._rnd.integers(2, size=NUM_QUBITS)
        return self._key

    def generate_quantum_state_for_bob(self) -> list[Qubit]:
        self._encoding_base = self._rnd.integers(2, size=NUM_QUBITS)

        st = time.time()
        for i in range(NUM_QUBITS):
            base = self._encoding_base[i]
            key_bit = self._key[i]
            qubit = self._quantum_state[i]
            if key_bit == 1:
                qubit.apply_gate(X_GATE)
            if base == X_BASE:
                qubit.apply_gate(H_GATE)
        et = time.time()
        print(et - st)

        return self._quantum_state

    def send_state_to_bob(self, mode="noise") -> list[Qubit]:
        if mode == "noise":
            noise = self._rnd.choice([0, 1], size=NUM_QUBITS, p=[0.875, 0.125])

            st = time.time()
            for i in range(NUM_QUBITS):
                if noise[i] == 1:
                    u_noise = unitary_group.rvs(2)
                    self._quantum_state[i].apply_gate(u_noise)

            et = time.time()
            print(et - st)

        return self._quantum_state


class Bob(Player):
    def __init__(self, state: list[Qubit], seed=None) -> None:
        Player.__init__(self, seed)
        self._quantum_state = state

    def measure_all_qubits(self) -> np.ndarray:
        self._encoding_base = self._rnd.integers(2, size=NUM_QUBITS)
        self._key = np.zeros_like(self._encoding_base)

        st = time.time()
        for i in range(NUM_QUBITS):
            base = self._encoding_base[i]
            qubit = self._quantum_state[i]
            if base == X_BASE:
                res = qubit.measure(H_GATE)
            else:
                res = qubit.measure()
            self._key[i] = res
        et = time.time()
        print(et - st)

        return self._key


def qkd_print_summary(alice: Alice, bob: Bob):
    print(f"\nBB84 summary. Number of Qubits: {NUM_QUBITS}")
    print("**************** keys ****************")
    print(f"Alice's key: {alice.key}")
    print(f"Bob's key: {bob.key}")
    print("**************** bases ****************")
    print(f"Alice's encoding bases: {alice.encoding_bases}")
    print(f"Bob's measuring bases: {bob.encoding_bases}")
    print("**************** agreed keys ****************")
    print(f"Alice's agreed key: {alice.agreed_key}")
    print(f"Bob's agreed key: {bob.agreed_key}")


def agreed_key(alice: Alice, bob: Bob):
    alice_key = alice.key
    alice_bases = alice.encoding_bases
    bob_key = bob.key
    bob_bases = bob.encoding_bases
    alice_agreed_key = []
    bob_agreed_key = []
    for i in range(NUM_QUBITS):
        if alice_bases[i] == bob_bases[i]:
            alice_agreed_key.append(alice_key[i])
            bob_agreed_key.append(bob_key[i])
    alice.agreed_key = alice_agreed_key
    bob.agreed_key = bob_agreed_key
    return alice_agreed_key, bob_agreed_key


def qkd():
    alice = Alice()
    alice.generate_key()
    print(alice.generate_quantum_state_for_bob())
    bob = Bob(alice.send_state_to_bob())
    print(bob.quantum_state)
    bob.measure_all_qubits()
    agreed_key(alice, bob)
    qkd_print_summary(alice, bob)


def cascade(noisy_key, qber):
    pass


if __name__ == "__main__":
    main()
