"""
BB84 QKD Algorithm - Workshop in Quantum Computing & Cryptography.
Made by Ofek Zohar (312490402).

The implementation based on the above sources:
    1. "Quantum cryptography: BB84 quantum key distribution by Nilanjana Datta & DAMTP Cambridge"
    2. "Cascade - Post Quantum Reconciliation" (https://cascade-python.readthedocs.io/en/latest/protocol.html)
    3. "Privacy Amplification - Post Quantum Reconciliation" Notes from the
    
Important general notes about the program & the implementation:
    1. The program has a command line interface (cli) - managing the algorithm parameters
        For more details run "python qky.py -h"
    3. The implementation doesn't take communication delays and protocols into consideration! (communication is immediate)
    2. The BB84 QKD algorithm is embarrassingly parallel - only 1-qubit gates, no entanglement.
        a. I chose to represent the state as a list of separate qubits (like block diagonal state).
           The perfect way to implement this kind of quantum circuit is using multi-CPU threads or GPU HW
                unfortunately Python doesn't support real CPU intensive multi-threading (I tried... Python GIL is a problem).
        b. So for small NUM_QUBIT numbers (under 1e9) it's more efficient to do sequential calculation.
    3. Cascade implementation - I chose to use an odd-parity-queue to gain the most of the Cascade effect
    4. Privacy Amplification - The eavesdropper (Eve) capability is to measure fraction of the qubits
        at the "communication step" (MITM attack)
        This a simple version of Amplification uses a hash function for extracting shorthand safe key.
"""

from enum import Enum
import argparse
from math import ceil, pi
import random
import time
import numpy as np
from scipy.stats import unitary_group

################################ Constants & Globals ################################

NUM_QUBITS = 100000  # BB84 number of qubits
NOISE_PROB = 0.125  # 1/8
MAX_QUBITS_ALLOWED = 1000000  # The number of maximum qubits allowed on BB84 (a feasible run)
MAX_QUBITS_ALLWOED_AMPLIFICATION = 1000  # The number of maximum qubits allowed on BB84 with amplification

print_mode = None  # The algorithm output mode

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
    SHORT = 1
    EXTEND = 2

    def __str__(self):
        return self.name.lower()


class ChannelMode(Enum):
    """bb84 algorithm quantum channel mode"""

    NOISY = "noisy"  # noise on the channel
    EAVESDROPPING = "eve"  # eavesdropping on the channel

    def __str__(self):
        return self.value


################################ Handlers ################################


def cli_arg_to_output_enum(astring: str) -> Output:
    """Convert cli output argument to Output enum value. Handler for argparse parsing.

    Args:
        astring (str): cli argument to convert to Output enum

    Raises:
        argparse.ArgumentTypeError: If the provided argument can't cast to Output type

    Returns:
        Output: The corresponded Output enum value
    """
    try:
        return Output[astring.upper()]
    except KeyError as exc:
        msg = f"Output mode: use one of {[str(mode) for mode in (Output)]}"
        raise argparse.ArgumentTypeError(msg) from exc


def fraction_float(x: str) -> float:
    """Float fraction converter (in range [0.0,1.0]). Handler for argparse parsing.

    Args:
        x (str): The number to convert

    Raises:
        argparse.ArgumentTypeError: If x not represent a floating point or not in the range

    Returns:
        float: The converted fraction float
    """
    try:
        x = float(x)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{x!a} not a floating-point literal") from exc

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError(f"{x} not in range [0.0, 1.0]")
    return x


def print_output(required_mode: Output, msg: str):
    """Print algorithm results according to the output mode

    Args:
        required_mode (Output): The required output mode for printing
        msg (str): The message to be printed
    """
    if print_mode is required_mode or (required_mode is Output.SHORT and print_mode is Output.EXTEND):
        print(msg)


def compact_complex_repr(num: complex) -> str:
    """Return compact string representation of complex numbers

    Args:
        num (complex): A complex number

    Returns:
        str: compact representation of num
    """
    if np.isreal(num):  # real number, exclude j
        if num.real == 1:
            return ""
        if num.real == -1:
            return "-"
        return format(num.real, "g")

    # Special cases for imaginary numbers
    if num.imag == 1:
        return "j"
    if num.imag == -1:
        return "-j"

    return num


######################## Quantum Phase - classes ########################


class Qubit:
    """The Qubit class represent a 1-qubit state. Initialize as the zero state |0>.

    Attributes:
        state (ndarray): The qubit state
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
    """Player class represent an abstract party in the BB84 algorithm.

    Attributes:
        key (:obj:`ndarray`): The full original key (before agreement phase)
        agreed_key (:obj:`ndarray`): The agreed key (after agreement phase)
        key_after_qber (:obj:`ndarray`): The agreed key after discarding random bits (qber calculation phase)
        key_reconciliation (:obj:`ndarray`): The agreed key after reconciliation
        key_privacy_amplification (:obj:`ndarray`): The agreed key after privacy amplification
        encoding_base (:obj:`ndarray`): The randomized encoding base
        quantum_state (:obj:`list` of :obj:`Qubit`): The BB84 quantum state

    Args:
        rnd (any, optional): Random generator object.
    """

    def __init__(self, rnd=None):
        # random generator, if not provided use default_rng()
        self._rnd = rnd if rnd is not None else np.random.default_rng()

        self.key = np.array([], dtype=int)
        self.agreed_key = np.array([], dtype=int)
        self.key_after_qber = np.array([], dtype=int)
        self.key_reconciliation = np.array([], dtype=int)
        self.key_privacy_amplification = np.array([], dtype=int)

        self.encoding_base = np.array([], dtype=int)
        self.quantum_state = []


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
            state (list[Qubit]): The quantum state sent to bob

        Returns:
            tuple[list[Qubit], np.ndarray]: The measured state, Eve's measured key
        """
        self.quantum_state = state
        self.key = np.zeros(NUM_QUBITS, dtype=int)  # Create an "empty" key with the correct length

        # There is NOISE_PROB chance for every qubit to get measured by eve
        eavesdropped = self._rnd.choice([0, 1], size=NUM_QUBITS, p=[1 - NOISE_PROB, NOISE_PROB])

        for i in range(NUM_QUBITS):
            if eavesdropped[i] == 1:
                # Measure the qubit in the BREIDBART base
                qubit = self.quantum_state[i]
                self.key[i] = qubit.measure(BREIDBART_BASE)
            else:
                # Eve can't measure this qubit - flip a toss
                self.key[i] = self._rnd.integers(2)

        return self.quantum_state, self.key

    def study_key_from_reconciliation_leak(self, revealed_bits: np.ndarray, bob_key: np.ndarray):
        """The bits eve learned from the leakage during the reconciliation process (unsecure channel)

        Args:
            revealed_bits (np.ndarray): The bits index reveled during the reconciliation process
            bob_key (np.ndarray): Bob's key after reconciliation
        """
        self.key_reconciliation = self.key_after_qber.copy()
        for i in revealed_bits:  # copy all leaked bits from bob's key
            self.key_reconciliation[i] = bob_key[i]


######################## Reconciliation Phase - classes & implementation ########################


class Block:
    """Block class represent a block (sequential) of key bits for the use of the Cascade and Binary algorithms.

    Attributes:
        start_index (:obj:`int`): The start index of the block
        end_index (:obj:`int`): The ebd index of the block (end index is excluded like in Range)
        iter (ndarray): The iteration number (of the Cascade algorithm) the block belongs to.
            Used for considering the right key permutation.
        parity (int): The one's parity of the block

    Args:
        start_index (:obj:`int`): The provided start index of the block
        end_index (:obj:`int`): The provided end index of the block
        iter_num (:obj:`int`): The provided iteration number
    """

    def __init__(self, start_index: int, end_index: int, iter_num: int):
        self.start_index = start_index
        self.end_index = end_index
        self.iter = iter_num
        self.parity = None

    def get_sub_blocks(self) -> tuple:
        """Split the block in the middle (if odd length the left sub block is bigger by one)

        Returns:
            tuple[Block, Block]: The left and right sub blocks
        """
        mid_index = ceil((self.start_index + self.end_index) / 2)
        left_sub_block = Block(start_index=self.start_index, end_index=mid_index, iter_num=self.iter)
        right_sub_block = Block(start_index=mid_index, end_index=self.end_index, iter_num=self.iter)
        return left_sub_block, right_sub_block

    @property
    def size(self):
        """int: The block size (length)"""
        return self.end_index - self.start_index


class Cascade:
    """Cascade class is the main class for the Cascade Algorithm implementation.

    The implementation details taken from: https://cascade-python.readthedocs.io/en/latest/protocol.html

    Args:
        true_key (:obj:`ndarray`): The provided correct key
        noisy_key (:obj:`ndarray`): The provided noisy key bits to be reconciliated
        qber (:obj:`float`): The estimated noisy key qber
        iterations (:obj:`int`): The number of Cascade iterations. Default to 4.
    """

    def __init__(self, true_key: np.ndarray, noisy_key: np.ndarray, qber: float, iterations: int = 4):
        self._key_len = len(true_key)  # int: The key length
        self._true_key = true_key.copy()  # ndarray: The correct key bits
        self._noisy_key = noisy_key.copy()  # ndarray: The noisy key bits to be reconciliated
        self._qber = (
            max(qber, 1 / self._key_len) if self._key_len != 0 else qber
        )  # int: The noisy key qber, set minimum to 1/key_len
        self._iterations = iterations  # int: Number of Cascade iterations

        self._shuffles = []  # list[ndarray]: list of index permutations for each iteration
        self._index_to_blocks = [[] for _ in range(self._key_len)]  # Linked blocks to key indexes
        self._odd_block_queue = []  # The odd parity queue - mark blocks to be reconciliated
        self._reveled_bits = []  # bits revealed (leaked) by the algorithm (unsecure channel)

    def _calculate_block_size(self, iter_num: int) -> int:
        """Calculate the Cascade top-level block size according to the given iteration

        Args:
            iter_num (int): The iteration number

        Returns:
            int: The Cascade top-level block size
        """
        return ceil(0.73 / self._qber) * 2**iter_num

    def _bit_flip(self, index: int):
        """Flipping a bit key value in the given index.

        All the index's linked blocks are also flipped (if already calculated).
        The new odd parity blocks are marked for "Cascading" (queued up).

        Args:
            index (int): The index of the bit to be flipped
        """
        self._noisy_key[index] ^= 1
        for block in self._index_to_blocks[index]:  # The blocks registered with the current index
            if block.parity is not None:  # parity already calculated
                block.parity ^= 1
                if block.parity == 1:  # If odd parity block - queue up!
                    self._odd_block_queue.append(block)

    def binary_algorithm(self, block: Block):
        """The binary algorithm implementation (recovering one odd parity block error).

        Args:
            block (Block): The Cascade top-level odd parity block
        """
        if block.parity == 0:
            # Make sure it's an odd parity block (can change since queued because of bit flipping)
            return

        if block.size == 1:  # Recursion stop condition, recover the error 0 bit flip
            index = self._shuffles[block.iter][block.start_index]  # find the original key index
            self._reveled_bits.append(index)  # Reveled bit index, add to list
            self._bit_flip(index)  # correct the bit value
        else:  # recursion step
            left_sub_block, right_sub_block = block.get_sub_blocks()
            if self._calculate_block_parity(left_sub_block) == 1:  # Left sub block is odd parity
                right_sub_block.parity = 0  # Right sub block must be even parity
                self.binary_algorithm(left_sub_block)
            else:  # Left sub block is even parity
                right_sub_block.parity = 1  # Right sub block must be odd parity
                self.binary_algorithm(right_sub_block)

    def _calculate_block_parity(self, block: Block) -> int:
        """Calculate the given block parity (if not calculated before). If already calculated return block.parity immediately

        Args:
            block (Block): _description_

        Returns:
            int: The block parity
        """
        if block.parity is None:  # Not calculated yet
            noisy_block_parity, true_block_parity = 0, 0
            shuffle = self._shuffles[block.iter]
            for index in range(block.start_index, block.end_index):  # iterate over the block indexes
                non_shuffled_index = shuffle[index]
                noisy_block_parity ^= self._noisy_key[non_shuffled_index]  # Bob
                true_block_parity ^= self._true_key[non_shuffled_index]  # Alice

                # register the block to be linked with non_shuffled_index
                self._index_to_blocks[non_shuffled_index].append(block)

            block.parity = noisy_block_parity ^ true_block_parity  # Parity between corrected and noisy key

        return block.parity

    def cascade(self) -> tuple[np.ndarray, list[int]]:
        """The main Cascade algorithm implementation.

        Using a blocks queue to gain the "Cascade" effect.

        Returns:
            np.ndarray: The reconciliated key
            list[int]: The reveled bits indexes
        """
        if self._key_len == 0:  # No process
            return self._noisy_key, []
        shuffle = range(self._key_len)  # 1st shuffle == the identity permutation

        for iter_num in range(self._iterations):  # Main Cascade loop iteration
            start_iter_time = time.time()
            print_output(Output.SHORT, f"start iter {iter_num}")

            if iter_num != 0:  # If not the 1st iteration, shuffle the key
                shuffle = np.random.permutation(self._key_len)
            self._shuffles.append(shuffle)  # Save the shuffle permutation for later use

            block_size = self._calculate_block_size(iter_num)  # Calculate top-level block size

            for start_block in range(0, self._key_len, block_size):  # Divide the shuffled key to blocks
                end_block = min(self._key_len, start_block + block_size)
                block = Block(start_block, end_block, iter_num)
                if self._calculate_block_parity(block) == 1:
                    # Mark as odd parity block (add to queue)
                    self._odd_block_queue.append(block)

            while len(self._odd_block_queue) != 0:  # Run binary algorithm on odd parity blocks
                block = self._odd_block_queue.pop(0)
                self.binary_algorithm(block)

            end_iter_time = time.time()
            print_output(
                Output.SHORT,
                f"end iter {iter_num}: Took {end_iter_time-start_iter_time}",
            )

        return self._noisy_key, self._reveled_bits


######################## Privacy Amplification Phase - implementation ########################


def privacy_amplification(alice: Alice, bob: Bob, eve: Eve):
    """Post quantum privacy amplification based on using an almost 2-universal hash function extractor (multiple shift bits - MSB)

    The extractor multiply the key with a randomized number (up to 2^NUM_QUBITS) and keep the first new_key_length bits
    Because of integer representation - no more than 64 bit length result allowed

    Args:
        alice (Alice): Alice player object
        bob (Bob): Bob player object
        eve (Eve): Eve player object
    """
    new_key_length = min(64, ceil(0.25 * len(alice.key_reconciliation)))  # The shorthand key length

    # Choose the random hash_value (salt) for the MSB hash function
    hash_random_binary = np.random.default_rng().integers(2, size=len(alice.key_reconciliation))
    # The hash random value in binary repr
    hash_random_value = fast_binary_to_decimal_modulo(hash_random_binary, 2**new_key_length)

    for player in [alice, bob, eve]:  # Calculate the new extracted amplified key for every party
        player.key_privacy_amplification = universal_hash_extractor(
            player.key_reconciliation, hash_random_value, new_key_length
        )


def universal_hash_extractor(key: np.ndarray, hash_random: int, hash_length: int) -> np.ndarray:
    """Applying hash function - multiple shift bits (almost 2-universal hash function)

    new_key_value = (key_value * hash_random) mod hash_length (Keep only the first hash_length of bits)

    Args:
        key (np.ndarray): key to be hashed
        hash_random (int): A random value determine the hash operation (Salt)
        hash_length (int): The number of bits to be kept

    Returns:
        ndarray: The bit representation of the calculated hash value
    """
    hash_mod = 2**hash_length
    new_key_value = fast_binary_to_decimal_modulo(key, hash_mod) * hash_random
    new_key_value %= hash_mod
    return np.array([int(i) for i in bin(new_key_value)[2:].zfill(hash_length)])


def fast_binary_to_decimal_modulo(binary: np.ndarray, mod: int) -> int:
    """Fast bit-array numpy to decimal int under modulo.

    Args:
        binary (np.ndarray): The bit array
        mod (int): The modulo

    Returns:
        int: The calculated number
    """
    res = 0
    for bit in reversed(binary):
        res *= 2
        res += bit
        res %= mod
    return int(res)


######################## BB84 Main Algorithm - classes & implementation ########################


class BB84:
    """BB84 class is the main class for the BB84 Algorithm implementation.

    Args:
        qber_sacrifice_fraction (:obj:`float`): The bit fraction used to estimate the qber. Defaults to 0.5.
        ch_mode (:obj:`ChannelMode`): The quantum channel mode. Defaults to ChannelMode.NOISY.
    """

    def __init__(
        self,
        qber_sacrifice_fraction: float = 0.5,
        ch_mode: ChannelMode = ChannelMode.NOISY,
    ):
        self._ch_mode = ch_mode  # The quantum channel mode. Defaults to ChannelMode.NOISY.

        self._qber_sacrifice_fraction = qber_sacrifice_fraction
        # float: The bit fraction used to estimate the qber. Defaults to 0.5.

        self._alice = None  # Alice: Alice player object
        self._bob = None  # Bob: Bob player object
        self._eve = None  # Eve: Eve player object

    def bb84(self):
        """The bb84 main algorithm"""

        print_output(Output.SHORT, "----------------------------- Run BB84 -----------------------------")

        # Quantum phase
        start_time = time.time()
        print_output(Output.SHORT, "##### start quantum phase ####")
        estimated_qber, true_qber = self.quantum_phase()
        alice, bob, eve = self._alice, self._bob, self._eve
        print_output(Output.SHORT, f"##### end quantum phase ({time.time()-start_time}) #####\n")

        # Reconciliation phase
        start_time = time.time()
        print_output(Output.SHORT, "##### start reconciliation phase (cascade algorithm) #####")
        reveled_bits = []
        reconciliation = Cascade(alice.key_after_qber, bob.key_after_qber, estimated_qber)
        bob.key_reconciliation, reveled_bits = reconciliation.cascade()
        alice.key_reconciliation = alice.key_after_qber  # Alice's key didn't change during the reconciliation
        print_output(Output.SHORT, f"##### end reconciliation phase ({time.time()-start_time}) #####\n")

        # Amplification phase
        if (
            self._ch_mode is ChannelMode.EAVESDROPPING
            and alice.key is not None
            and len(alice.key_reconciliation) != 0
        ):
            start_time = time.time()
            print_output(Output.SHORT, "##### start privacy amplification phase (hashing) #####")
            eve.study_key_from_reconciliation_leak(reveled_bits, bob.key_reconciliation)
            privacy_amplification(alice, bob, eve)
            print_output(
                Output.SHORT, f"##### end privacy amplification phase ({time.time()-start_time}) #####\n"
            )
        print_output(Output.SHORT, "----------------------------- End of BB84 -----------------------------")

        self._print_summary(estimated_qber, true_qber, len(reveled_bits))

    def quantum_phase(self) -> tuple[float, float]:
        """The quantum Phase of the BB84 algorithm (step 1-3)

        Returns:
            tuple[float, float]: The estimated qber (from the sacrificed bits), The true qber (from the remain bits)
        """

        print_output(Output.SHORT, "generate key step")
        self._eve = Eve() if self._ch_mode is ChannelMode.EAVESDROPPING else None
        self._alice = Alice()
        self._alice.generate_key()
        self._alice.generate_quantum_state_for_bob()

        print_output(Output.SHORT, "quantum communication step")
        sent_state = self._alice.send_state_to_bob()  # Alice send the state through quantum channel
        if self._ch_mode is ChannelMode.EAVESDROPPING:
            # Eve measure the qubits sent to Bob
            sent_state, _ = self._eve.measure_eavesdropped_qubits(sent_state)
        self._bob = Bob(sent_state)  # Bob get the sent state

        self._bob.measure_all_qubits()

        print_output(Output.SHORT, "key agree step")
        self._agreed_key()

        print_output(Output.SHORT, "qber calculation step")
        estimated_qber, true_qber = self._calculate_qber()

        return estimated_qber, true_qber

    def _agreed_key(self) -> np.ndarray:
        """The key agreement step. Keep every bit measured (Bob) by the same base were encoded (Alice).

        Returns:
            np.ndarray: The indexes Alice & Bob agrees on (same encoding base)
        """
        same_base_indexes = np.where(self._alice.encoding_base == self._bob.encoding_base)[0]

        self._alice.agreed_key = self._alice.key[same_base_indexes].copy()
        self._bob.agreed_key = self._bob.key[same_base_indexes].copy()
        if self._eve is not None:
            self._eve.agreed_key = self._eve.key[same_base_indexes].copy()
        return same_base_indexes

    def _calculate_qber(self) -> tuple[float, float]:
        """Calculating the Bob's key quantum bit error rate.
        Alice and Bob randomly chose sacrifice_fraction of the agreed key bits to estimate the error rate.

        Returns:
            tuple[float, float]: The estimated qber (from the sacrificed bits), The true qber (from the remain bits)

            The true qber made only for compassion and isn't accessible to the Players!
        """
        alice, bob, eve = self._alice, self._bob, self._eve
        sacrifice_fraction = self._qber_sacrifice_fraction

        key_length = len(alice.agreed_key)
        index_perm = np.random.permutation(key_length)  # Choose random permutation on the key's indexes
        random_sacrifice_bits, remain_bits = (
            # The first sacrifice_fraction bits selected for revealing
            index_perm[: ceil(sacrifice_fraction * key_length)],
            index_perm[ceil(sacrifice_fraction * key_length) :],  # The remain indexes are the unraveled bits
        )

        alice_key, bob_key = alice.agreed_key, bob.agreed_key
        # Average over the different bits
        estimated_qber = np.average(alice_key[random_sacrifice_bits] ^ bob_key[random_sacrifice_bits])

        # Discard all reveled bits from the key
        alice.key_after_qber, bob.key_after_qber = alice_key[remain_bits], bob_key[remain_bits]

        # Calculate the qber over the remain bits (inaccessible to the players)
        true_qber = np.average(alice.key_after_qber ^ bob.key_after_qber)
        if eve is not None:  # If Eve on the channel, she also discard the reveled bits
            eve.key_after_qber = eve.agreed_key[remain_bits].copy()
        return estimated_qber, true_qber

    def _print_summary(self, estimated_qber: float, true_qber: float, num_reveled_bits: int):
        """Print the BB84 algorithm summary according to the right output mode.

        Args:
            estimated_qber (float): The qber estimated after key agreement step
            true_qber (float): The true qber after key agreement step
            num_reveled_bits (int): The number of bits reveled by the reconciliation phase
        """
        alice, bob, eve = self._alice, self._bob, self._eve
        key_length = (
            len(bob.key_privacy_amplification)
            if self._ch_mode is ChannelMode.EAVESDROPPING
            else len(bob.key_reconciliation)
        )

        print_output(
            Output.SHORT,
            f"""
----------------------------- BB84 Summary -----------------------------
Number of Qubits: {NUM_QUBITS}
Final key length: {key_length}

############# Quantum Phase Summary ##############""",
        )

        print_output(
            Output.EXTEND,
            f"""**************** original keys ****************
Key length: {len(alice.key)}
Alice's key: {alice.key}
Bob's key: {bob.key}
**************** encoding bases ****************
Alice's encoding bases: {alice.encoding_base}
Bob's measuring bases: {bob.encoding_base}
**************** agreed keys ****************")
Key length: {len(alice.agreed_key)}
Alice's agreed key: {alice.agreed_key}
Bob's agreed key: {bob.agreed_key}""",
        )
        if self._ch_mode is ChannelMode.EAVESDROPPING:
            print_output(Output.EXTEND, f"Eve's agreed key: {eve.agreed_key}")

        print_output(
            Output.SHORT,
            f"""**************** quantum bit error rate ****************
estimated error: {estimated_qber}
true error: {true_qber}""",
        )

        print_output(Output.SHORT, "\n############# Reconciliation Summary ##############")
        print_output(
            Output.EXTEND,
            f"""**************** reconciliated keys ****************
Alice's key: {alice.key_reconciliation}
Bob's key: {bob.key_reconciliation}""",
        )
        print_output(
            Output.SHORT,
            f"""**************** rates ****************
reveled bits rate: {num_reveled_bits / len(bob.key_reconciliation) if len(bob.key_reconciliation) != 0 else 0}
qber after reconciliation: {np.average(bob.key_reconciliation ^ alice.key_reconciliation)}""",
        )

        if self._ch_mode is ChannelMode.EAVESDROPPING:
            # Calculate the average diff of eve from the right key before/after amplification
            eve_advantage_before_amp = 0.5 - np.average(alice.key_reconciliation ^ eve.key_reconciliation)
            eve_advantage_after_amp = 0.5 - np.average(
                alice.key_privacy_amplification ^ eve.key_privacy_amplification
            )

            print_output(Output.SHORT, "\n############# Privacy Amplification Summary ##############")
            print_output(
                Output.EXTEND,
                f"""**************** Privacy Amplification keys ****************
Key length: {len(alice.key_privacy_amplification) if alice.key_privacy_amplification is not None else 0}
Alice's key: {alice.key_privacy_amplification}
Bob's key: {bob.key_privacy_amplification}
Eve's key: {eve.key_privacy_amplification}""",
            )
            print_output(
                Output.SHORT,
                f"""**************** Eve advantage compare to pure random choice (0.5) ****************
Eve advantage before amplification: {eve_advantage_before_amp}
Eve advantage after amplification: {eve_advantage_after_amp}""",
            )


######################## Main Program & command line interface ########################


def cli() -> argparse.Namespace:
    """Generates a cli for the BB84 program with optional flags.

    Returns:
        argparse.Namespace: cli argument parser
    """
    # create a cli parser object
    parser = argparse.ArgumentParser(description="QKD BB84 algorithm simulator. Made by Ofek Zohar.")

    # Number of qubits argument
    parser.add_argument(
        "-n",
        "--num-qubits",
        metavar="num",
        type=int,
        default=NUM_QUBITS,
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


def main():
    """The main program"""
    global NUM_QUBITS, print_mode

    args = cli()  # Create the command line interface
    NUM_QUBITS = args.num_qubits  # setting the number of qubits for the BB84 algorithm
    if 1 > NUM_QUBITS or NUM_QUBITS > MAX_QUBITS_ALLOWED:
        print(f"Number of qubits should be in range [1, {MAX_QUBITS_ALLOWED}]")
        return
    print_mode = args.output  # Setting the algorithm's printing mode

    if args.mode is ChannelMode.EAVESDROPPING and NUM_QUBITS > MAX_QUBITS_ALLWOED_AMPLIFICATION:
        print(f"Amplification phase is limited to {MAX_QUBITS_ALLWOED_AMPLIFICATION} qubits!")
        return

    # Run the BB84 algorithm
    qkd_bb84_algorithm = BB84(ch_mode=args.mode, qber_sacrifice_fraction=args.fraction)
    qkd_bb84_algorithm.bb84()


if __name__ == "__main__":
    main()
