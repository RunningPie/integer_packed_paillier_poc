import numpy as np
import phe.paillier as paillier
import tenseal as ts

# Constants for Packed Paillier
SLOT_WIDTH = 32
MULTIPLIER = 10000  # scale by 10,000 to retain 4 decimal places
SIGN_EXT_MASK = 0xFFFFFF  # 24-bit sign-extended representation
SIGN_BIT_MASK = 0x800000  # 24th bit acts as sign indicator (2^23)
BIAS = 65536  # 2^16 bias for original biased packing

def generate_paillier_keys():
    """Generate a 2048-bit Paillier public/private keypair."""
    public_key, private_key = paillier.generate_paillier_keypair(n_length=2048)
    return public_key, private_key

def encrypt_elementwise_paillier(public_key, weights):
    """
    Encrypt each model parameter individually using element-wise Paillier.
    weights: 1D numpy array of floats.
    Returns: list of phe.paillier.EncryptedNumber.
    """
    return [public_key.encrypt(float(w)) for w in weights]

def decrypt_elementwise_paillier(private_key, encrypted_weights):
    """
    Decrypt each model parameter individually.
    encrypted_weights: list of phe.paillier.EncryptedNumber.
    Returns: 1D numpy array of floats.
    """
    return np.array([private_key.decrypt(w) for w in encrypted_weights], dtype=float)

def pack_parameters(weights, packing_method='twos_complement', multiplier=MULTIPLIER):
    """
    Pack a 1D float array of parameters into a single large python integer.
    packing_method: 'twos_complement' or 'biased'
    """
    packed_int = 0
    for i, w in enumerate(weights):
        if packing_method == 'twos_complement':
            q = int(round(w * multiplier))
            if q < -32768 or q > 32767:
                raise ValueError(f"Quantized weight {q} at index {i} overflows signed 16-bit payload range.")
            extended = q & SIGN_EXT_MASK
            packed_int |= (extended & 0xFFFFFFFF) << (SLOT_WIDTH * i)
        elif packing_method == 'biased':
            q = int(round(w * multiplier)) + BIAS
            if q < 0 or q >= (1 << SLOT_WIDTH):
                raise ValueError(f"Quantized weight {q} at index {i} overflows 32-bit slot width.")
            packed_int |= (q & 0xFFFFFFFF) << (SLOT_WIDTH * i)
        else:
            raise ValueError(f"Unknown packing method: {packing_method}")
    return packed_int

def unpack_parameters(packed_int, num_clients, num_params, packing_method='twos_complement', multiplier=MULTIPLIER):
    """
    Unpack a large python integer back into a 1D float array of weights.
    packing_method: 'twos_complement' or 'biased'
    """
    weights = []
    for i in range(num_params):
        slot = (packed_int >> (SLOT_WIDTH * i)) & 0xFFFFFFFF
        if packing_method == 'twos_complement':
            sum_q = slot & SIGN_EXT_MASK
            if sum_q >= SIGN_BIT_MASK:
                sum_q -= (SIGN_EXT_MASK + 1)
            avg_w = sum_q / (num_clients * multiplier)
        elif packing_method == 'biased':
            sum_q = slot - num_clients * BIAS
            avg_w = sum_q / (num_clients * multiplier)
        else:
            raise ValueError(f"Unknown packing method: {packing_method}")
        weights.append(avg_w)
    return np.array(weights, dtype=float)

def encrypt_packed_paillier(public_key, weights, packing_method='twos_complement'):
    """
    Pack parameters and encrypt the resulting single large integer.
    """
    packed_int = pack_parameters(weights, packing_method=packing_method)
    return public_key.encrypt(packed_int)

def decrypt_and_unpack_packed_paillier(private_key, encrypted_packed, num_clients, num_params, packing_method='twos_complement'):
    """
    Decrypt the single packed Paillier ciphertext and unpack it to return the averaged weights.
    """
    packed_int = private_key.decrypt(encrypted_packed)
    return unpack_parameters(packed_int, num_clients, num_params, packing_method=packing_method)

def create_ckks_context():
    """
    Initialize a TenSEAL context for addition-only CKKS.
    Poly modulus degree 8192, coeff_mod_bit_sizes=[60, 40, 40, 60] (standard depth-0).
    """
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2**40
    return context

def encrypt_ckks(context, weights):
    """
    Encrypt a 1D float array of weights into a single CKKS vector.
    """
    return ts.ckks_vector(context, weights)

def decrypt_ckks(private_context, encrypted_vector):
    """
    Decrypt a CKKS vector using the private context (which contains the secret key).
    Returns: 1D numpy array of floats.
    """
    decrypted = encrypted_vector.decrypt()
    return np.array(decrypted, dtype=float)
