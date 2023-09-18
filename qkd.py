from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory
from math import ceil, pi, floor
import random
import time
import numpy as np
from scipy import linalg, sparse
from scipy.stats import unitary_group

# TODO: consider bitarray instead of int

NUM_QUBITS = 500000

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
        self._key_after_qber = None
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
    def key_after_qber(self):
        return self._agreed_key

    @key_after_qber.setter
    def key_after_qber(self, value: list):
        self._key_after_qber = value

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
        print(f"Alice generate state time: {et - st}")

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
        print(f"Alice send time: {et - st}")

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
        print(f"Bob measure time: {et - st}")

        return self._key


def print_quantum_phase_summary(
    alice: Alice, bob: Bob, estimated_qber: float, true_qber: float
):
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
    print("**************** quantum bit error rate ****************")
    print(f"estimated error: {estimated_qber}")
    print(f"true error: {true_qber}")
    print("**************** key after qber calculation ****************")
    print(f"Alice's agreed key: {alice.key_after_qber}")
    print(f"Bob's agreed key: {bob.key_after_qber}")


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


def calculate_qber(
    alice: Alice, bob: Bob, sacrifice_fraction: float
) -> tuple[float, float]:
    key_length = len(alice.agreed_key)
    index_perm = np.random.permutation(key_length)
    random_sacrifice_bits, bits_left = (
        index_perm[: ceil(sacrifice_fraction * key_length)],
        index_perm[ceil(sacrifice_fraction * key_length) :],
    )
    # random_sacrifice_bits = random.sample(range(key_length), k=floor(sacrifice_fraction * key_length))

    alice_key, bob_key = np.array(alice.agreed_key), np.array(bob.agreed_key)
    estimated_qber = np.average(
        alice_key[random_sacrifice_bits] ^ bob_key[random_sacrifice_bits]
    )
    true_qber = np.average(alice_key ^ bob_key)
    alice.key_after_qber, bob.key_after_qber = alice_key[bits_left], bob_key[bits_left]
    return estimated_qber, true_qber


def bb84_quantum_phase(
    sacrifice_fraction: float, print_summary: bool
) -> tuple[Alice, Bob, float, float]:
    alice = Alice()
    alice.generate_key()
    alice.generate_quantum_state_for_bob()
    bob = Bob(alice.send_state_to_bob())
    # print(bob.quantum_state)
    bob.measure_all_qubits()
    print("start agree key phase")
    agreed_key(alice, bob)
    print("start qber calculation phase")
    estimated_qber, true_qber = calculate_qber(alice, bob, sacrifice_fraction)
    print("end qber calculation phase")
    if print_summary:
        print_quantum_phase_summary(alice, bob, estimated_qber, true_qber)
    return alice, bob, estimated_qber, true_qber


class Block:
    def __init__(
        self,
        start_index: int,
        end_index: int,
        iter_num: int,
    ) -> None:
        self._start_index = start_index
        self._end_index = end_index
        self._iter = iter_num
        self._parity = None

    def get_sub_blocks(self):
        mid_index = ceil((self._start_index + self._end_index) / 2)
        left_sub_block = Block(
            start_index=self._start_index, end_index=mid_index, iter_num=self._iter
        )
        right_sub_block = Block(
            start_index=mid_index, end_index=self._end_index, iter_num=self._iter
        )
        return left_sub_block, right_sub_block

    @property
    def parity(self):
        return self._parity

    @parity.setter
    def parity(self, value: int):
        self._parity = value

    @property
    def start_index(self):
        return self._start_index

    @property
    def end_index(self):
        return self._end_index

    @property
    def iter(self):
        return self._iter

    @property
    def size(self):
        return self._end_index - self._start_index


class Cascade:
    def __init__(
        self,
        true_key: list[int],
        noisy_key: list[int],
        qber: float,
        iterations: int = 4,
    ) -> None:
        self._key_len = len(true_key)
        self._true_key = true_key.copy()
        self._noisy_key = noisy_key.copy()
        self._qber = max(qber, 1 / self._key_len)
        self._iterations = iterations
        self._shuffles = []
        self._index_to_blocks = [[] for i in range(self._key_len)]
        self._odd_block_queue = []
        self._reveled_bits = []

    def calculate_block_size(self, iter_num: int) -> int:
        return ceil(0.73 / self._qber) * 2**iter_num

    def bit_flip(self, index: int):
        self._noisy_key[index] ^= 1
        for block in self._index_to_blocks[index]:
            if block.parity is not None:
                block.parity ^= 1
                if block.parity == 1:
                    self._odd_block_queue.append(block)

    def binary_algorithm(self, block: Block):
        if block.parity == 0:
            return
        if block.size == 1:
            index = self._shuffles[block.iter][block.start_index]
            self._reveled_bits.append(index)
            self.bit_flip(index)
        else:
            left_sub_block, right_sub_block = block.get_sub_blocks()
            if self.calculate_block_parity(left_sub_block) == 1:
                right_sub_block.parity = 0
                self.binary_algorithm(left_sub_block)
            else:
                right_sub_block.parity = 1
                self.binary_algorithm(right_sub_block)

    def calculate_block_parity(self, block: Block):
        if block.parity is None:
            noisy_block_parity, true_block_parity = 0, 0
            shuffle = self._shuffles[block.iter]
            for index in range(block.start_index, block.end_index):
                non_shuffled_index = shuffle[index]
                noisy_block_parity ^= self._noisy_key[non_shuffled_index]  # Bob
                true_block_parity ^= self._true_key[non_shuffled_index]  # Alice
                self._index_to_blocks[non_shuffled_index].append(block)
            block.parity = noisy_block_parity ^ true_block_parity
        return block.parity

    def cascade(self):
        shuffle = range(self._key_len)
        for iter_num in range(self._iterations):
            et = time.time()
            print(f"start iter {iter_num} time: {et}")
            # print(f"noisy key iter {iter_num}: {self._noisy_key}")
            if iter_num != 0:
                shuffle = np.random.permutation(self._key_len)
            self._shuffles.append(shuffle)
            block_size = self.calculate_block_size(iter_num)
            for start_block in range(0, self._key_len, block_size):
                end_block = min(self._key_len, start_block + block_size)
                block = Block(start_block, end_block, iter_num)
                if self.calculate_block_parity(block) == 1:
                    self._odd_block_queue.append(block)
            while len(self._odd_block_queue) != 0:
                block = self._odd_block_queue.pop(0)
                self.binary_algorithm(block)
            st = time.time()
            print(f"end iter {iter_num} time: {st}. Took {st-et}")


def qkd_bb83_algorithm(sacrifice_fraction: float = 0.5, print_summary: bool = True):
    print("start quantum phase")
    alice, bob, estimated_qber, true_qber = bb84_quantum_phase(
        sacrifice_fraction, print_summary
    )
    print("start reconciliation phase")
    reconciliation = Cascade(alice.key_after_qber, bob.key_after_qber, estimated_qber)
    reconciliation.cascade()
    print(
        f"\nBB84 summary. Number of Qubits: {NUM_QUBITS}. Final key length: {reconciliation._key_len}"
    )
    print("**************** quantum bit error rate ****************")
    print(f"estimated error: {estimated_qber}")
    print(f"true error: {true_qber}")
    print(
        f"reveled bits rate {len(reconciliation._reveled_bits)/ reconciliation._key_len}"
    )
    print(np.sum(np.array(bob.key_after_qber) ^ np.array(alice.key_after_qber)))
    print(np.sum(np.array(reconciliation._noisy_key) ^ np.array(alice.key_after_qber)))


def main():
    qkd_bb83_algorithm(print_summary=False)


if __name__ == "__main__":
    main()
