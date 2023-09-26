from concurrent.futures import ProcessPoolExecutor
from enum import Enum
import argparse
from math import ceil
import random
import time
import numpy as np
from scipy.stats import unitary_group

# TODO: consider bitarray instead of int

################################ Constants ################################

NUM_QUBITS = 100
PARALLELIZATION_THRESHOLD = 500000
NOISE_PROB = 0.125  # 1/8

# 1-qubit Gates
I_GATE = np.array([[1, 0], [0, 1]])
X_GATE = np.array([[0, 1], [1, 0]])
Z_GATE = np.array([[1, 0], [0, -1]])
H_GATE = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

# Base states
COMP_BASE_STATES = [np.array([1, 0]), np.array([0, 1])]  # Z base
X_BASE_STATES = [np.array([1, 1]) / np.sqrt(2), np.array([1, -1]) / np.sqrt(2)]  # X base


# Enums
class Base(Enum):
    """Enum for a 1-qubit hilbert space base"""

    Z = 0  # The computational base
    X = 1  # |+>,|-> base, often called Hadamard base


class Output(Enum):
    """Printing mode for bb84 algorithm"""

    NONE = 0
    GRAPHS = 1
    SHORT = 2
    EXTEND = 3

    def __str__(self):
        return self.name.lower()


class ChannelMode(Enum):
    """bb84 algorithm quantum channel mode"""

    NOISY = "noisy"  # noise on the channel
    EAVESDROPPING = "eve"  # eavesdropping on the channel

    def __str__(self):
        return self.value


def main():
    args = cli()
    qkd_bb83_algorithm(print_mode=False)


def cli() -> argparse.Namespace:
    # create a cli parser object
    parser = argparse.ArgumentParser(description="QKD BB84 algorithm simulator. Made by Ofek Zohar.")

    # Number of qubits argument
    parser.add_argument(
        "-n",
        "--num-qubits",
        metavar="num",
        type=int,
        help="Number of qubits to be used in the BB84 quantum phase.",
    )

    # Quantum channel mode argument
    # determines the post quantum process (info reconciliation/privacy amplification)
    parser.add_argument(
        "-m",
        "--mode",
        metavar="channel_mode",
        type=ChannelMode,
        choices=list(ChannelMode),
        default=ChannelMode.NOISY,
        help="Quantum channel mode %(choices)a. (default: %(default)s)",
    )

    # Algorithm results printing mode
    parser.add_argument(
        "-o",
        "--output",
        metavar="output_mode",
        type=cli_arg_to_output_enum,
        choices=list(Output),
        default=Output.NONE.name.lower(),
        help="BB84 printing mode %(choices)a. (default: %(default)s)",
    )

    # qber sacrificed bit fraction argument
    parser.add_argument(
        "-f",
        "--fraction",
        type=fraction_float,
        default=0.5,
        help="Fraction of bits to be used ('sacrificed') to calculate quantum bit estimated error (AKA qber). (default: %(default)s)",
    )

    # parse the arguments from standard input
    return parser.parse_args()


def cli_arg_to_output_enum(astring: str) -> Output:
    try:
        return Output[astring.upper()]
    except KeyError as exc:
        msg = f"Output mode: use one of {[str(mode) for mode in (Output)]}"
        raise argparse.ArgumentTypeError(msg) from exc


def fraction_float(x) -> float:
    try:
        x = float(x)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{x!a} not a floating-point literal") from exc

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError(f"{x} not in range [0.0, 1.0]")
    return x


def print_output(required_mode: Output, current_mode: Output, msg: str):
    if current_mode is required_mode or (required_mode is Output.SHORT and current_mode is Output.EXTEND):
        print(msg)


def qkd_bb83_algorithm(
    qber_sacrifice_fraction: float = 0.5,
    print_mode: Output = Output.NONE,
    ch_mode: ChannelMode = ChannelMode.NOISY,
):
    print_output(Output.EXTEND, print_mode, "start quantum phase")
    alice, bob, estimated_qber, true_qber = bb84_quantum_phase(qber_sacrifice_fraction, print_mode)

    if ch_mode is ChannelMode.NOISY:
        print_output(Output.EXTEND, print_mode, "start reconciliation phase (cascade algorithm)")
        reconciliation = Cascade(alice.key_after_qber, bob.key_after_qber, estimated_qber)
        reconciliation.cascade()
    elif ch_mode is ChannelMode.EAVESDROPPING:
        pass

    print(f"\nBB84 summary. Number of Qubits: {NUM_QUBITS}. Final key length: {reconciliation._key_len}")
    print("**************** quantum bit error rate ****************")
    print(f"estimated error: {estimated_qber}")
    print(f"true error: {true_qber}")
    print(f"reveled bits rate {len(reconciliation._reveled_bits)/ reconciliation._key_len}")
    print(np.sum(np.array(bob.key_after_qber) ^ np.array(alice.key_after_qber)))
    print(np.sum(np.array(reconciliation._noisy_key) ^ np.array(alice.key_after_qber)))


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


######################## Quantum Phase - classes and handlers ########################


class Qubit:
    """The Qubit class represent a 1-qubit state. Initialize as the zero state |0>.

    Attributes:
        state (np.ndarray): The qubit state
    """

    def __init__(self) -> None:
        self._state = np.array([1, 0], dtype=complex)  # |0> state

    def __str__(self) -> str:
        ket_repr = ""
        if (self._state == X_BASE_STATES[0]).all():
            return "|+>"
        if (self._state == X_BASE_STATES[1]).all():
            return "|->"
        if abs(self._state[0]) != 0:  # coefficient of |0> isn't zero
            coef_0 = compact_complex_repr(self._state[0])
            ket_repr += f"{coef_0}|0>"
        if abs(self._state[1]) != 0:  # coefficient of |1> isn't zero
            coef_1 = compact_complex_repr(self._state[1])
            if ket_repr != "":  # coefficient of |0> isn't zero, add plus sign if needed
                ket_repr += " + " if self._state[1] > 0 else " "
            ket_repr += f"{coef_1}|1>"
        return ket_repr

    def __repr__(self) -> str:
        return str(self)

    def apply_gate(self, gate: np.ndarray):
        """Apply a 1-quibt quantum gate on the current state.

        Args:
            gate (np.ndarray): The gate to apply - (2,2) unitary matrix
        """
        self._state = gate @ self._state

    def measure(self, base: np.ndarray = None) -> int:
        """Measure the current state according to the provided base.
        Measure in a non-computational base (U) can be done by:
            1. Rotating to the computational base (apply U*)
            2. Measure the qubit state
            3. Rotate back to the provided base (from the computational base, apply U)


        Args:
            base (np.ndarray, optional): The base to measure by. Defaults to None (measure in the computational base).

        Returns:
            int: The measurement result bit (the first base state always represented by the 'zero' result)
        """
        if base is not None:
            self._state = base.conj().T @ self._state
        prob = np.square(np.abs(self._state))  # Calc the weighted probabilities
        res = random.choices([0, 1], weights=prob)[0]  # The weighted probability measurement
        self._state = COMP_BASE_STATES[res]  # The collapsed state
        if base is not None:
            self._state = base @ self._state
        return res


class Player:
    """Player class represent an abstract party in the BB84 algorithm."""

    def __init__(self, rnd=None):
        # random generator, if not provided use default_rng()
        self._rnd = rnd if not None else np.random.default_rng()

        self._key = None  # ndarray: The full original key (before agreement phase)
        self._agreed_key = None  # ndarray: The agreed key (after agreement phase)
        self._key_after_qber = None
        # ndarray: The agreed key after discarding random bits (qber calculation phase)

        self._encoding_base = None  # ndarray: The randomized encoding base
        self._quantum_state = None  # list[Qubit]: The BB84 quantum state

    @property
    def key(self) -> np.ndarray:
        """ndarray: The full original key (before agreement phase)."""
        return self._key

    @property
    def agreed_key(self) -> np.ndarray:
        """ndarray: The agreed key (after agreement phase)."""
        return self._agreed_key

    @agreed_key.setter
    def agreed_key(self, value: list):
        self._agreed_key = value

    @property
    def key_after_qber(self) -> np.ndarray:
        """ndarray: The agreed key after discarding random bits (qber calculation phase)"""
        return self._agreed_key

    @key_after_qber.setter
    def key_after_qber(self, value: np.ndarray):
        self._key_after_qber = value

    @property
    def encoding_bases(self) -> np.ndarray:
        """ndarray: The randomized encoding base"""
        return self._encoding_base

    @property
    def quantum_state(self) -> list[Qubit]:
        """list[Qubit]: The BB84 quantum state"""
        return self._quantum_state


class Alice(Player):
    """Alice class represent alice's part of the BB84 algorithm"""

    def __init__(self, rnd=None):
        Player.__init__(self, rnd)
        self._quantum_state = [Qubit() for _ in range(NUM_QUBITS)]  # Init the |00...0> state

    def generate_key(self) -> np.ndarray:
        """Randomly generate key (bits). Key length is NUM_QUBITS.

        Returns:
            np.ndarray: The generated key
        """
        self._key = self._rnd.integers(2, size=NUM_QUBITS)
        return self._key

    def generate_quantum_state_for_bob(self) -> list[Qubit]:
        """Generate quantum state according to alice's key in randomized encoding base.

        If key length is over the PARALLELIZATION THRESHOLD, break the encoding process to multiprocessors task.

        encoding states:
            |0> - encoding the zero bit in Z base.

            |1> - encoding the one bit in Z base.

            |+> - encoding the zero bit in X base.

            |-> - encoding the one bit in X base.

        Returns:
            list[Qubit]: The generated quantum state.
        """
        self._encoding_base = self._rnd.integers(2, size=NUM_QUBITS)  # Choose encoding base randomly

        def task(index: int):  # Encoding one key bit into 1-qubit state
            base = self._encoding_base[index]
            key_bit = self._key[index]
            qubit = self._quantum_state[index]
            if key_bit == 1:
                qubit.apply_gate(X_GATE)
            if base is Base.X:
                qubit.apply_gate(H_GATE)

        if NUM_QUBITS < PARALLELIZATION_THRESHOLD:
            # Small key, multiprocessing overhead is too expensive
            # Iterate sequently over all bits
            for i in range(NUM_QUBITS):
                task(i)
        else:  # multiprocessing the task to simulate quantum parallelism
            with ProcessPoolExecutor() as pool:
                pool.map(task, range(NUM_QUBITS), chunksize=NUM_QUBITS)

        return self._quantum_state

    def send_state_to_bob(self, mode: ChannelMode = ChannelMode.NOISY) -> list[Qubit]:
        """Send state to Bob over the quantum channel.

        Noisy channel -> There is NOISE_PROB chance for every qubit to get noisy (arbitrary unitary rotation)
        Eavesdropping channel -> Eve act as MITM (measures the sent qubits).

        If key length is over the PARALLELIZATION THRESHOLD, break the process to multiprocessors task.

        Args:
            mode (ChannelMode, optional): The Quantum channel mode. Defaults to ChannelMode.NOISY.

        Returns:
            list[Qubit]: _description_
        """
        if mode is ChannelMode.NOISY:  # Noise on the quantum channel
            self._noisy_channel()
        elif mode is ChannelMode.EAVESDROPPING:
            self._eve_channel()

        return self._quantum_state

    def _noisy_channel(self):
        """Noisy channel handler"""

        # There is NOISE_PROB chance for every qubit to get noisy
        noise = self._rnd.choice([0, 1], size=NUM_QUBITS, p=[1 - NOISE_PROB, NOISE_PROB])
        noise_indexes = np.where(noise == 1)[0]

        def task(index: int):  # Apply arbitrary unitary rotation
            u_noise = unitary_group.rvs(2)
            self._quantum_state[index].apply_gate(u_noise)

        if NUM_QUBITS < PARALLELIZATION_THRESHOLD:
            # Small key, multiprocessing overhead is too expensive
            # Iterate sequently over all bits
            for i in noise_indexes:
                task(i)
        else:  # multiprocessing the task to simulate quantum parallelism
            with ProcessPoolExecutor() as pool:
                pool.map(task, noise_indexes, chunksize=NUM_QUBITS)

    def _eve_channel(self):
        pass


class Bob(Player):
    def __init__(self, state: list[Qubit], rnd=None):
        Player.__init__(self, rnd)
        self._quantum_state = state

    def measure_all_qubits(self) -> np.ndarray:
        self._encoding_base = self._rnd.integers(2, size=NUM_QUBITS)
        self._key = np.zeros_like(self._encoding_base)

        st = time.time()
        for i in range(NUM_QUBITS):
            base = self._encoding_base[i]
            qubit = self._quantum_state[i]
            if base is Base.X:
                res = qubit.measure(H_GATE)
            else:
                res = qubit.measure()
            self._key[i] = res
        et = time.time()
        print(f"Bob measure time: {et - st}")

        return self._key


def print_quantum_phase_summary(alice: Alice, bob: Bob, estimated_qber: float, true_qber: float):
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


def calculate_qber(alice: Alice, bob: Bob, sacrifice_fraction: float) -> tuple[float, float]:
    key_length = len(alice.agreed_key)
    index_perm = np.random.permutation(key_length)
    random_sacrifice_bits, bits_left = (
        index_perm[: ceil(sacrifice_fraction * key_length)],
        index_perm[ceil(sacrifice_fraction * key_length) :],
    )
    # random_sacrifice_bits = random.sample(range(key_length), k=floor(sacrifice_fraction * key_length))

    alice_key, bob_key = np.array(alice.agreed_key), np.array(bob.agreed_key)
    estimated_qber = np.average(alice_key[random_sacrifice_bits] ^ bob_key[random_sacrifice_bits])
    true_qber = np.average(alice_key ^ bob_key)
    alice.key_after_qber, bob.key_after_qber = alice_key[bits_left], bob_key[bits_left]
    return estimated_qber, true_qber


def bb84_quantum_phase(sacrifice_fraction: float, print_summary: bool) -> tuple[Alice, Bob, float, float]:
    alice = Alice()
    alice.generate_key()
    alice.generate_quantum_state_for_bob2()
    bob = Bob(alice.send_state_to_bob2())
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
    def __init__(self, start_index: int, end_index: int, iter_num: int) -> None:
        self._start_index = start_index
        self._end_index = end_index
        self._iter = iter_num
        self._parity = None

    def get_sub_blocks(self):
        mid_index = ceil((self._start_index + self._end_index) / 2)
        left_sub_block = Block(start_index=self._start_index, end_index=mid_index, iter_num=self._iter)
        right_sub_block = Block(start_index=mid_index, end_index=self._end_index, iter_num=self._iter)
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
    def __init__(self, true_key: list[int], noisy_key: list[int], qber: float, iterations: int = 4):
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


if __name__ == "__main__":
    main()
