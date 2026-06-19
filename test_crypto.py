import numpy as np
from crypto_utils import (
    generate_paillier_keys,
    encrypt_elementwise_paillier,
    decrypt_elementwise_paillier,
    pack_parameters,
    unpack_parameters,
    encrypt_packed_paillier,
    decrypt_and_unpack_packed_paillier,
    create_ckks_context,
    encrypt_ckks,
    decrypt_ckks
)
from server import Server

def test_crypto_correctness():
    print("=== Testing Cryptographic Correctness ===")
    np.random.seed(42)
    
    # 1. Generate keys
    print("Generating Paillier keys...")
    paillier_pub, paillier_priv = generate_paillier_keys()
    print("Generating CKKS context...")
    ckks_context = create_ckks_context()
    
    # 2. Setup dummy parameters (length 55)
    num_params = 55
    original_weights = np.random.uniform(-1.5, 1.5, num_params)
    
    # 3. Element-wise Paillier Test
    print("Testing Element-wise Paillier...")
    enc_ew = encrypt_elementwise_paillier(paillier_pub, original_weights)
    dec_ew = decrypt_elementwise_paillier(paillier_priv, enc_ew)
    ew_error = np.max(np.abs(original_weights - dec_ew))
    print(f"  Element-wise Paillier max error: {ew_error:.6f}")
    assert np.allclose(original_weights, dec_ew, atol=1e-5), "Element-wise Paillier failed!"

    # 4. Packed Paillier Test (1 client) - Two's Complement
    print("Testing Packed Paillier (1 client, twos_complement)...")
    enc_packed = encrypt_packed_paillier(paillier_pub, original_weights, packing_method='twos_complement')
    dec_packed = decrypt_and_unpack_packed_paillier(paillier_priv, enc_packed, num_clients=1, num_params=num_params, packing_method='twos_complement')
    packed_error = np.max(np.abs(original_weights - dec_packed))
    print(f"  twos_complement max error: {packed_error:.6f}")
    assert np.allclose(original_weights, dec_packed, atol=1e-4), f"Packed Paillier twos_complement failed (max error: {packed_error:.6f})!"

    # 5. Packed Paillier Test (1 client) - Biased
    print("Testing Packed Paillier (1 client, biased)...")
    enc_packed_b = encrypt_packed_paillier(paillier_pub, original_weights, packing_method='biased')
    dec_packed_b = decrypt_and_unpack_packed_paillier(paillier_priv, enc_packed_b, num_clients=1, num_params=num_params, packing_method='biased')
    packed_error_b = np.max(np.abs(original_weights - dec_packed_b))
    print(f"  biased max error: {packed_error_b:.6f}")
    assert np.allclose(original_weights, dec_packed_b, atol=1e-4), f"Packed Paillier biased failed (max error: {packed_error_b:.6f})!"

    # 6. CKKS Test
    print("Testing CKKS...")
    enc_ckks = encrypt_ckks(ckks_context, original_weights)
    dec_ckks = decrypt_ckks(ckks_context, enc_ckks)
    ckks_error = np.max(np.abs(original_weights - dec_ckks))
    print(f"  CKKS max error: {ckks_error:.6f}")
    assert np.allclose(original_weights, dec_ckks, atol=1e-4), "CKKS failed!"
    
    print("All single-client tests passed!\n")

def test_aggregation():
    print("=== Testing Homomorphic Aggregation ===")
    np.random.seed(100)
    num_clients = 3
    num_params = 55
    
    # Generate keys
    paillier_pub, paillier_priv = generate_paillier_keys()
    ckks_context = create_ckks_context()
    
    # Generate 3 sets of client updates
    client_weights = [np.random.uniform(-1.0, 1.0, num_params) for _ in range(num_clients)]
    plaintext_sum = np.sum(client_weights, axis=0)
    plaintext_avg = plaintext_sum / num_clients
    
    server = Server()
    
    # --- Plaintext ---
    pt_sum = server.aggregate('plaintext', client_weights)
    pt_avg = pt_sum / num_clients
    assert np.allclose(plaintext_avg, pt_avg, atol=1e-7)
    print("  Plaintext aggregation: OK")

    # --- Element-wise Paillier ---
    print("  Running Element-wise Paillier aggregation...")
    ew_encrypted = [encrypt_elementwise_paillier(paillier_pub, w) for w in client_weights]
    ew_sum_enc = server.aggregate('elementwise_paillier', ew_encrypted)
    ew_sum_dec = decrypt_elementwise_paillier(paillier_priv, ew_sum_enc)
    ew_avg = ew_sum_dec / num_clients
    ew_error = np.max(np.abs(plaintext_avg - ew_avg))
    print(f"    Element-wise Paillier aggregation max error: {ew_error:.6f}")
    assert np.allclose(plaintext_avg, ew_avg, atol=1e-5)

    # --- Packed Paillier (twos_complement) ---
    print("  Running Packed Paillier (twos_complement) aggregation...")
    packed_encrypted = [encrypt_packed_paillier(paillier_pub, w, packing_method='twos_complement') for w in client_weights]
    packed_sum_enc = server.aggregate('packed_paillier', packed_encrypted)
    packed_avg = decrypt_and_unpack_packed_paillier(paillier_priv, packed_sum_enc, num_clients=num_clients, num_params=num_params, packing_method='twos_complement')
    packed_error = np.max(np.abs(plaintext_avg - packed_avg))
    print(f"    twos_complement aggregation max error: {packed_error:.6f}")
    assert np.allclose(plaintext_avg, packed_avg, atol=1e-4)

    # --- Packed Paillier (biased) ---
    print("  Running Packed Paillier (biased) aggregation...")
    packed_encrypted_b = [encrypt_packed_paillier(paillier_pub, w, packing_method='biased') for w in client_weights]
    packed_sum_enc_b = server.aggregate('packed_paillier', packed_encrypted_b)
    packed_avg_b = decrypt_and_unpack_packed_paillier(paillier_priv, packed_sum_enc_b, num_clients=num_clients, num_params=num_params, packing_method='biased')
    packed_error_b = np.max(np.abs(plaintext_avg - packed_avg_b))
    print(f"    biased aggregation max error: {packed_error_b:.6f}")
    assert np.allclose(plaintext_avg, packed_avg_b, atol=1e-4)

    # --- CKKS ---
    print("  Running CKKS aggregation...")
    ckks_encrypted = [encrypt_ckks(ckks_context, w) for w in client_weights]
    ckks_sum_enc = server.aggregate('ckks', ckks_encrypted)
    ckks_sum_dec = decrypt_ckks(ckks_context, ckks_sum_enc)
    ckks_avg = ckks_sum_dec / num_clients
    ckks_error = np.max(np.abs(plaintext_avg - ckks_avg))
    print(f"    CKKS aggregation max error: {ckks_error:.6f}")
    assert np.allclose(plaintext_avg, ckks_avg, atol=1e-4)

    print("All homomorphic aggregation tests passed!\n")

if __name__ == '__main__':
    test_crypto_correctness()
    test_aggregation()
    print("All tests successfully completed!")
