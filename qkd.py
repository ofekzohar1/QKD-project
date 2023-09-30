from enum import Enum
import argparse
from math import ceil, pi, floor
import random
import time
import numpy as np
from scipy.stats import unitary_group

################################ Constants ################################

NUM_QUBITS = 100000
NOISE_PROB = 0.125  # 1/8

# 1-qubit Gates
I_GATE = np.array([[1, 0], [0, 1]])
X_GATE = np.array([[0, 1], [1, 0]])
Z_GATE = np.array([[1, 0], [0, -1]])
H_GATE = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

# Base states
COMP_BASE_STATES = [np.array([1, 0]), np.array([0, 1])]  # Z base
X_BASE_STATES = [np.array([1, 1]) / np.sqrt(2), np.array([1, -1]) / np.sqrt(2)]  # X base
BREIDBART_BASE = [np.array([np.cos(pi / 8), np.sin(pi / 8)]), np.array([-np.sin(pi / 8), np.cos(pi / 8)])]


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
            base_gate = np.column_stack(base)  # The unitary gate induced by the base
            self._state = base_gate.conj().T @ self._state
        prob = np.square(np.abs(self._state))  # Calc the weighted probabilities
        res = random.choices([0, 1], weights=prob)[0]  # The weighted probability measurement
        self._state = COMP_BASE_STATES[res]  # The collapsed state
        if base is not None:
            self._state = base_gate @ self._state
        return res


class Player:
    """Player class represent an abstract party in the BB84 algorithm."""

    def __init__(self, rnd=None):
        # random generator, if not provided use default_rng()
        self._rnd = rnd if rnd is not None else np.random.default_rng()

        self.key = None  # ndarray: The full original key (before agreement phase)
        self.agreed_key = None  # ndarray: The agreed key (after agreement phase)
        self.key_after_qber = None
        # ndarray: The agreed key after discarding random bits (qber calculation phase)
        self.key_reconciliation = None  # ndarray: The agreed key after reconciliation
        self.key_privacy_amplification = None  # ndarray: The agreed key after privacy amplification

        self.encoding_base = None  # ndarray: The randomized encoding base
        self.quantum_state = None  # list[Qubit]: The BB84 quantum state


class Alice(Player):
    """Alice class represent Alice's part of the BB84 algorithm

    Args:
        rnd (any, optional): Random generator object.
    """

    def __init__(self, rnd=None):
        Player.__init__(self, rnd)
        self.quantum_state = [Qubit() for _ in range(NUM_QUBITS)]  # Init the |00...0> state

    def generate_key(self) -> np.ndarray:
        """Randomly generate key (bits). Key length is NUM_QUBITS.

        Returns:
            np.ndarray: The generated key
        """
        self.key = self._rnd.integers(2, size=NUM_QUBITS)
        return self.key

    def generate_quantum_state_for_bob(self) -> list[Qubit]:
        """Generate quantum state according to alice's key in randomized encoding base.

        encoding states:
            |0> - encoding the zero bit in Z base.

            |1> - encoding the one bit in Z base.

            |+> - encoding the zero bit in X base.

            |-> - encoding the one bit in X base.

        Returns:
            list[Qubit]: The generated quantum state.
        """
        self.encoding_base = self._rnd.integers(2, size=NUM_QUBITS)  # Choose encoding base randomly

        for i in range(NUM_QUBITS):
            base = self.encoding_base[i]
            key_bit = self.key[i]
            qubit = self.quantum_state[i]
            if key_bit == 1:  # Rotation |0> to |1>
                qubit.apply_gate(X_GATE)
            if base == Base.X.value:  # Encode in the X base -> apply Hadamard gate
                qubit.apply_gate(H_GATE)

        return self.quantum_state

    def send_state_to_bob(self, mode: ChannelMode = ChannelMode.NOISY) -> list[Qubit]:
        """Send state to Bob over the quantum channel.

        Noisy channel -> There is NOISE_PROB chance for every qubit to get noisy (arbitrary unitary rotation)

        Args:
            mode (ChannelMode, optional): The Quantum channel mode. Defaults to ChannelMode.NOISY.

        Returns:
            list[Qubit]: _description_
        """
        if mode is ChannelMode.NOISY:  # Noise on the quantum channel
            # There is NOISE_PROB chance for every qubit to get noisy
            noise = self._rnd.choice([0, 1], size=NUM_QUBITS, p=[1 - NOISE_PROB, NOISE_PROB])
            noise_indexes = np.where(noise == 1)[0]

            for index in noise_indexes:  # Apply arbitrary unitary rotation
                u_noise = unitary_group.rvs(2)
                self.quantum_state[index].apply_gate(u_noise)

        return self.quantum_state


class Bob(Player):
    """Bob class represent Bob's part of the BB84 algorithm

    Args:
        state (list[Qubit]): The quantum state sent to bob.
        rnd (any, optional): Random generator object.
    """

    def __init__(self, state: list[Qubit], rnd=None):
        Player.__init__(self, rnd)
        self.quantum_state = state

    def measure_all_qubits(self) -> np.ndarray:
        """Measure the received quantum state according to guessed encoding bases (randomized).

        Returns:
            np.ndarray: The measured key (bits)
        """
        self.encoding_base = self._rnd.integers(2, size=NUM_QUBITS)  # Guess Alice's encoding base randomly
        self.key = np.zeros_like(self.encoding_base)  # Create an "empty" key with the correct length

        for i in range(NUM_QUBITS):
            # Measure 1-qubit state in the guessed base
            base = self.encoding_base[i]
            qubit = self.quantum_state[i]
            if base == Base.X.value:
                res = qubit.measure(X_BASE_STATES)
            else:
                res = qubit.measure()
            self.key[i] = res  # Save the measurement result in Bob's key

        return self.key


class Eve(Player):
    """Eve class represent an eavesdropper on the quantum channel of the BB84 algorithm."""

    def measure_eavesdropped_qubits(self, state: list[Qubit]) -> tuple[list[Qubit], np.ndarray]:
        """Measure the eavesdropped quantum state according to the BREIDBART base.

        Args:
            state (list[Qubit]): The quantum state sent to bob.

        Returns:
            tuple[list[Qubit], np.ndarray]: The measured state, Eve's measured key
        """
        self.quantum_state = state
        self.key = np.zeros(NUM_QUBITS, dtype=int)  # Create an "empty" key with the correct length

        # There is NOISE_PROB chance for every qubit to get measured by eve
        eavesdropped = self._rnd.choice([0, 1], size=NUM_QUBITS, p=[1 - NOISE_PROB, NOISE_PROB])

        for i in range(NUM_QUBITS):
            if eavesdropped[i] == 1:
                qubit = self.quantum_state[i]
                self.key[i] = qubit.measure(BREIDBART_BASE)
                # Save the measurement (BREIDBART base) result in Eve's key
            else:
                # Eve can't measure this qubit - flip a toss
                self.key[i] = self._rnd.integers(2)

        return self.quantum_state, self.key

    def study_key_from_reconciliation_leak(self, revealed_bits: np.ndarray, bob_key: np.ndarray):
        """The bits eve learned from the bits leaked during the reconciliation process (unsecure channel)

        Args:
            revealed_bits (np.ndarray): _description_
            bob_key (np.ndarray): _description_
        """
        self.key_reconciliation = self.key_after_qber.copy()
        for i in revealed_bits:
            self.key_reconciliation[i] = bob_key[i]


def print_quantum_phase_summary(alice: Alice, bob: Bob, estimated_qber: float, true_qber: float):
    print(f"\nBB84 summary. Number of Qubits: {NUM_QUBITS}")
    print("**************** keys ****************")
    print(f"Alice's key: {alice.key}")
    print(f"Bob's key: {bob.key}")
    print("**************** bases ****************")
    print(f"Alice's encoding bases: {alice.encoding_base}")
    print(f"Bob's measuring bases: {bob.encoding_base}")
    print("**************** agreed keys ****************")
    print(f"Alice's agreed key: {alice.agreed_key}")
    print(f"Bob's agreed key: {bob.agreed_key}")
    print("**************** quantum bit error rate ****************")
    print(f"estimated error: {estimated_qber}")
    print(f"true error: {true_qber}")
    print("**************** key after qber calculation ****************")
    print(f"Alice's agreed key: {alice.key_after_qber}")
    print(f"Bob's agreed key: {bob.key_after_qber}")


def agreed_key(alice: Alice, bob: Bob, eve: Eve = None) -> np.ndarray:
    same_base_indexes = np.where(alice.encoding_base == bob.encoding_base)[0]

    alice.agreed_key = alice.key[same_base_indexes].copy()
    bob.agreed_key = bob.key[same_base_indexes].copy()
    if eve is not None:
        eve.agreed_key = eve.key[same_base_indexes].copy()
    return same_base_indexes


def calculate_qber(
    alice: Alice, bob: Bob, eve: Eve = None, sacrifice_fraction: float = 0.5
) -> tuple[float, float]:
    key_length = len(alice.agreed_key)
    index_perm = np.random.permutation(key_length)
    random_sacrifice_bits, remain_bits = (
        index_perm[: ceil(sacrifice_fraction * key_length)],
        index_perm[ceil(sacrifice_fraction * key_length) :],
    )

    alice_key, bob_key = alice.agreed_key, bob.agreed_key
    estimated_qber = np.average(alice_key[random_sacrifice_bits] ^ bob_key[random_sacrifice_bits])
    alice.key_after_qber, bob.key_after_qber = alice_key[remain_bits], bob_key[remain_bits]
    true_qber = np.average(alice.key_after_qber ^ bob.key_after_qber)
    if eve is not None:
        eve.key_after_qber = eve.agreed_key[remain_bits].copy()
    return estimated_qber, true_qber


######################## Reconciliation Phase - classes and handlers ########################


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
        return self._noisy_key, self._reveled_bits


####


def privacy_amplification(alice: Alice, bob: Bob, eve: Eve, allowed_key_leak: int):
    new_key_length = floor(-np.log2(allowed_key_leak))  # The allowed leak is 2^-m when m is the new length
    hash_random_binary = np.random.default_rng().integers(2, size=len(alice.key_reconciliation))
    # The hash random value in binary repr
    hash_random_value = fast_binary_to_decimal_modulo(hash_random_binary, 2**new_key_length)
    for player in [alice, bob, eve]:
        player.key_privacy_amplification = universal_hash_extractor(
            player.key_reconciliation, hash_random_value, new_key_length
        )
    eve_advantage_before_amp = 1 - np.average(alice.key_reconciliation ^ eve.key_reconciliation)
    eve_advantage_after_amp = 1 - np.average(alice.key_privacy_amplification ^ eve.key_privacy_amplification)
    return eve_advantage_before_amp, eve_advantage_after_amp


def universal_hash_extractor(key: np.ndarray, hash_random: int, hash_length: int):
    hash_mod = 2**hash_length
    new_key_value = fast_binary_to_decimal_modulo(key, hash_mod) * hash_random
    new_key_value %= hash_mod
    return np.array([int(i) for i in bin(new_key_value)[2:].zfill(hash_length)])


def fast_binary_to_decimal_modulo(binary: np.ndarray, mod: int) -> int:
    res = 0
    for bit in reversed(binary):
        res *= 2
        res += bit
        res %= mod
    return int(res)


#####


def bb84_quantum_phase(
    qber_sacrifice_fraction: float, print_mode: Output = Output.NONE, ch_mode: ChannelMode = ChannelMode.NOISY
) -> tuple[Alice, Bob, Eve, float, float]:
    eve = Eve() if ch_mode is ChannelMode.EAVESDROPPING else None
    alice = Alice()
    alice.generate_key()
    alice.generate_quantum_state_for_bob()

    sent_state = alice.send_state_to_bob()
    if ch_mode is ChannelMode.EAVESDROPPING:
        sent_state, _ = eve.measure_eavesdropped_qubits(sent_state)
    bob = Bob(sent_state)

    bob.measure_all_qubits()

    print("start agree key phase")
    agreed_key(alice, bob, eve)
    print("start qber calculation phase")
    estimated_qber, true_qber = calculate_qber(alice, bob, eve, qber_sacrifice_fraction)
    print("end qber calculation phase", estimated_qber, true_qber)
    if print_mode is Output.SHORT:
        print_quantum_phase_summary(alice, bob, estimated_qber, true_qber)
    return alice, bob, eve, estimated_qber, true_qber


def qkd_bb84_algorithm(
    qber_sacrifice_fraction: float = 0.5,
    print_mode: Output = Output.NONE,
    ch_mode: ChannelMode = ChannelMode.NOISY,
    allowed_key_leak: float = 2**-32,
):
    print_output(Output.EXTEND, print_mode, "start quantum phase")
    alice, bob, eve, estimated_qber, true_qber = bb84_quantum_phase(
        qber_sacrifice_fraction, print_mode, ch_mode
    )

    print_output(Output.EXTEND, print_mode, "start reconciliation phase (cascade algorithm)")
    reconciliation = Cascade(alice.key_after_qber, bob.key_after_qber, estimated_qber)
    bob.key_reconciliation, reveled_bits = reconciliation.cascade()
    alice.key_reconciliation = alice.key_after_qber

    if ch_mode is ChannelMode.EAVESDROPPING:
        eve.study_key_from_reconciliation_leak(reveled_bits, bob.key_reconciliation)
        pa_key_error, pa_key_leak = privacy_amplification(alice, bob, eve, allowed_key_leak)
        print(pa_key_error, pa_key_leak)

    print(f"\nBB84 summary. Number of Qubits: {NUM_QUBITS}. Final key length: {len(bob.key_reconciliation)}")
    print("**************** quantum bit error rate ****************")
    print(f"estimated error: {estimated_qber}")
    print(f"true error: {true_qber}")
    print(f"reveled bits rate {len(reveled_bits) / len(bob.key_reconciliation)}")
    print(np.sum(bob.key_after_qber ^ alice.key_after_qber))
    print(np.sum(bob.key_reconciliation ^ alice.key_after_qber))


def main():
    args = cli()
    qkd_bb84_algorithm(ch_mode=ChannelMode.EAVESDROPPING)


if __name__ == "__main__":
    main()
