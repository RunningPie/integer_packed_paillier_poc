import os
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder

# Import custom modules
from crypto_utils import (
    generate_paillier_keys,
    decrypt_elementwise_paillier,
    decrypt_and_unpack_packed_paillier,
    create_ckks_context,
    decrypt_ckks
)
from client import Client, SimpleMLP
from server import Server

def get_fixed_initial_params(seed=42):
    """Generate a fixed set of initial parameters so all simulations start identically."""
    np.random.seed(seed)
    # MLP structure: 4 -> 4 -> 4 -> 3
    # Total parameter count = 55
    W1 = np.random.randn(4, 4) * np.sqrt(2.0 / 4)
    b1 = np.zeros(4)
    W2 = np.random.randn(4, 4) * np.sqrt(2.0 / 4)
    b2 = np.zeros(4)
    W3 = np.random.randn(4, 3) * np.sqrt(2.0 / 4)
    b3 = np.zeros(3)
    
    return np.concatenate([
        W1.ravel(), b1.ravel(),
        W2.ravel(), b2.ravel(),
        W3.ravel(), b3.ravel()
    ])

def load_and_partition_data(seed=42):
    """
    Load Iris dataset, shuffle, take exactly half (75 samples),
    and split into 3 clients (25 samples each: 15 train, 5 val, 5 test).
    """
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Shuffle dataset
    np.random.seed(seed)
    indices = np.random.permutation(len(y))
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    # Take exactly half (75 samples)
    X_half = X_shuffled[:75]
    y_half = y_shuffled[:75]
    
    # One-hot encode targets
    encoder = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(y_half.reshape(-1, 1))
    
    # Standardize input features
    mean = np.mean(X_half, axis=0)
    std = np.std(X_half, axis=0)
    X_half = (X_half - mean) / (std + 1e-15)
    
    num_clients = 3
    samples_per_client = 25
    
    client_train_x = []
    client_train_y = []
    client_val_x = []
    client_val_y = []
    client_test_x = []
    client_test_y = []
    
    for c in range(num_clients):
        start = c * samples_per_client
        # 15 train
        client_train_x.append(X_half[start : start + 15])
        client_train_y.append(y_onehot[start : start + 15])
        # 5 validation
        client_val_x.append(X_half[start + 15 : start + 20])
        client_val_y.append(y_onehot[start + 15 : start + 20])
        # 5 testing
        client_test_x.append(X_half[start + 20 : start + 25])
        client_test_y.append(y_onehot[start + 20 : start + 25])
        
    # Aggregate sets for global evaluation
    X_train_all = np.vstack(client_train_x)
    y_train_all = np.vstack(client_train_y)
    X_val_all = np.vstack(client_val_x)
    y_val_all = np.vstack(client_val_y)
    X_test_all = np.vstack(client_test_x)
    y_test_all = np.vstack(client_test_y)
    
    return (client_train_x, client_train_y, 
            X_train_all, y_train_all, 
            X_val_all, y_val_all, 
            X_test_all, y_test_all)

def run_simulation(scheme, num_rounds, data, initial_params, control_weights_history=None, packing_method='twos_complement'):
    """Run the federated learning simulation for a specific scheme."""
    (client_train_x, client_train_y, X_train_all, y_train_all, X_val_all, y_val_all, X_test_all, y_test_all) = data
    num_clients = len(client_train_x)
    num_params = 55
    
    # Initialize global model
    global_model = SimpleMLP()
    global_model.set_parameters(initial_params)
    
    # Initialize virtual clients
    clients = []
    for c in range(num_clients):
        client = Client(client_id=c, X_train=client_train_x[c], y_train=client_train_y[c], seed=42+c)
        clients.append(client)
        
    # Setup cryptographic context
    pub_key, priv_key = None, None
    ckks_context = None
    
    if scheme in ('elementwise_paillier', 'packed_paillier'):
        print(f"[{scheme.upper()}] Generating 2048-bit Paillier keys...")
        pub_key, priv_key = generate_paillier_keys()
        crypto_key = pub_key
    elif scheme == 'ckks':
        print(f"[{scheme.upper()}] Setting up TenSEAL CKKS context...")
        ckks_context = create_ckks_context()
        crypto_key = ckks_context
        # The client needs the context with secret key for decryption,
        # but to simulate a decoupled server, we copy context and strip secret key for aggregation
        server_context = ckks_context.copy()
        server_context.make_context_public()
    else:
        crypto_key = None
        
    server = Server()
    results = []
    weights_history = []
    
    print(f"[{scheme.upper()}] Starting simulation for {num_rounds} rounds (packing: {packing_method if scheme == 'packed_paillier' else 'N/A'})...")
    
    for r in range(1, num_rounds + 1):
        global_params = global_model.get_parameters()
        
        # 1. Share global parameters with clients
        for client in clients:
            client.set_model_parameters(global_params)
            
        # 2. Local plaintext updates (1 epoch, batch size 5)
        for client in clients:
            client.train_local(epochs=1, batch_size=5, lr=0.05)
            
        # 3. Encryption Masking (client-side)
        t_enc_start = time.perf_counter()
        client_updates = []
        for client in clients:
            enc_update = client.encrypt_parameters(scheme, crypto_key, packing_method=packing_method)
            client_updates.append(enc_update)
        t_enc = time.perf_counter() - t_enc_start
        
        # 4. Central Homomorphic Aggregation (server-side)
        t_agg_start = time.perf_counter()
        aggregated_update = server.aggregate(scheme, client_updates)
        t_agg = time.perf_counter() - t_agg_start
        
        # 5. Global Delivery and Decryption (client-side)
        t_dec_start = time.perf_counter()
        if scheme == 'plaintext':
            avg_params = aggregated_update / num_clients
        elif scheme == 'elementwise_paillier':
            sum_params = decrypt_elementwise_paillier(priv_key, aggregated_update)
            avg_params = sum_params / num_clients
        elif scheme == 'packed_paillier':
            avg_params = decrypt_and_unpack_packed_paillier(priv_key, aggregated_update, num_clients, num_params, packing_method=packing_method)
        elif scheme == 'ckks':
            sum_params = decrypt_ckks(ckks_context, aggregated_update)
            avg_params = sum_params / num_clients
        t_dec = time.perf_counter() - t_dec_start
        
        # Update global model
        global_model.set_parameters(avg_params)
        weights_history.append(avg_params.copy())
        
        # Evaluate model performance
        # Global Training Loss
        train_preds = global_model.forward(X_train_all)
        train_loss = -np.mean(np.sum(y_train_all * np.log(train_preds + 1e-15), axis=-1))
        
        # Validation Loss and Accuracy
        val_preds = global_model.forward(X_val_all)
        val_loss = -np.mean(np.sum(y_val_all * np.log(val_preds + 1e-15), axis=-1))
        val_acc = np.mean(np.argmax(val_preds, axis=-1) == np.argmax(y_val_all, axis=-1))
        
        # Test Accuracy
        test_preds = global_model.forward(X_test_all)
        test_acc = np.mean(np.argmax(test_preds, axis=-1) == np.argmax(y_test_all, axis=-1))
        
        # Compute Noise MSE compared to Plaintext Control Group
        weight_mse = 0.0
        if control_weights_history is not None and r <= len(control_weights_history):
            control_w = control_weights_history[r - 1]
            weight_mse = np.mean((avg_params - control_w) ** 2)
            
        results.append({
            'round': r,
            'scheme': scheme,
            'packing_method': packing_method if scheme == 'packed_paillier' else 'N/A',
            'enc_time': t_enc,
            'agg_time': t_agg,
            'dec_time': t_dec,
            'total_crypto_time': t_enc + t_agg + t_dec,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'weight_mse': weight_mse
        })
        
        if r % max(1, num_rounds // 5) == 0 or r == num_rounds:
            print(f"  Round {r}/{num_rounds} | Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Crypto Time: {t_enc+t_agg+t_dec:.4f}s")
            
    return results, weights_history

def generate_plots(df, num_rounds, timestamp):
    """Generate and save publication-quality performance graphs."""
    print("Generating performance charts...")
    os.makedirs('graphs', exist_ok=True)
    
    plt.rcParams.update({'font.size': 11})
    colors = {
        'plaintext': '#2C3E50',
        'elementwise_paillier': '#E74C3C',
        'packed_paillier': '#2ECC71',
        'ckks': '#9B59B6'
    }
    labels = {
        'plaintext': 'Plaintext (Control)',
        'elementwise_paillier': 'Element-wise Paillier',
        'packed_paillier': 'Integer-Packed Paillier (Proposed)',
        'ckks': 'CKKS (TenSEAL)'
    }
    
    # 1. Validation Accuracy & Loss
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy Plot
    for scheme in df['scheme'].unique():
        sub_df = df[df['scheme'] == scheme]
        axes[0].plot(sub_df['round'], sub_df['val_acc'], label=labels[scheme], color=colors[scheme], linewidth=2)
    axes[0].set_title('Validation Accuracy over Communication Rounds')
    axes[0].set_xlabel('Communication Round')
    axes[0].set_ylabel('Accuracy')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend()
    
    # Loss Plot
    for scheme in df['scheme'].unique():
        sub_df = df[df['scheme'] == scheme]
        axes[1].plot(sub_df['round'], sub_df['val_loss'], label=labels[scheme], color=colors[scheme], linewidth=2)
    axes[1].set_title('Validation Loss over Communication Rounds')
    axes[1].set_xlabel('Communication Round')
    axes[1].set_ylabel('Cross-Entropy Loss')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(f'graphs/{timestamp}_accuracy_loss_comparison_{num_rounds}rounds.png', dpi=300)
    plt.close()
    
    # 2. Weight Noise (MSE)
    fig, ax = plt.subplots(figsize=(8, 5))
    for scheme in ['elementwise_paillier', 'packed_paillier', 'ckks']:
        if scheme in df['scheme'].unique():
            sub_df = df[df['scheme'] == scheme]
            ax.plot(sub_df['round'], sub_df['weight_mse'], label=labels[scheme], color=colors[scheme], linewidth=2)
    ax.set_title('Cryptographic Noise Infiltration (Weight MSE)')
    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Mean Squared Error (vs Plaintext)')
    ax.set_yscale('log')
    ax.grid(True, which="both", linestyle='--', alpha=0.6)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'graphs/{timestamp}_weight_noise_comparison_{num_rounds}rounds.png', dpi=300)
    plt.close()
    
    # 3. Computational Complexity (Time Comparison)
    avg_times = []
    for scheme in df['scheme'].unique():
        sub_df = df[df['scheme'] == scheme]
        avg_times.append({
            'scheme': labels[scheme],
            'Encryption': sub_df['enc_time'].mean(),
            'Aggregation': sub_df['agg_time'].mean(),
            'Decryption': sub_df['dec_time'].mean(),
            'color': colors[scheme]
        })
    time_df = pd.DataFrame(avg_times)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(time_df))
    width = 0.25
    
    rects1 = ax.bar(x - width, time_df['Encryption'], width, label='Encryption', color='#34495E')
    rects2 = ax.bar(x, time_df['Aggregation'], width, label='Aggregation', color='#BDC3C7')
    rects3 = ax.bar(x + width, time_df['Decryption'], width, label='Decryption', color='#7F8C8D')
    
    ax.set_ylabel('Time (seconds, log scale)')
    ax.set_title('Average Computation Overhead Per Round')
    ax.set_xticks(x)
    ax.set_xticklabels(time_df['scheme'], rotation=15)
    ax.set_yscale('log')
    ax.grid(True, which="both", linestyle='--', alpha=0.4)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'graphs/{timestamp}_computation_time_comparison_{num_rounds}rounds.png', dpi=300)
    plt.close()
    print(f"Performance charts saved successfully in 'graphs/' with timestamp prefix {timestamp} and suffix _{num_rounds}rounds.png")

def main():
    parser = argparse.ArgumentParser(description="Integer-Packed Paillier Federated Learning PoC")
    parser.add_argument('--rounds', type=int, default=250, help='Total training rounds for Plaintext, Packed Paillier, and CKKS (default: 250)')
    parser.add_argument('--ew-rounds', type=int, default=50, help='Total training rounds for slow Element-wise Paillier (default: 50)')
    parser.add_argument('--packing-method', type=str, choices=['twos_complement', 'biased'], default='twos_complement',
                        help='Packing method for Integer-Packed Paillier: twos_complement or biased (default: twos_complement)')
    args = parser.parse_args()
    
    # Load and split dataset
    print("Loading and partitioning Iris dataset (75 samples)...")
    data = load_and_partition_data()
    
    # Get standard initial weights
    initial_params = get_fixed_initial_params()
    
    all_results = []
    
    # 1. Run Plaintext (Control Group)
    print("\n--- Running Plaintext (Control Group) Simulation ---")
    pt_results, pt_weights_history = run_simulation('plaintext', args.rounds, data, initial_params)
    all_results.extend(pt_results)
    
    # 2. Run CKKS (TenSEAL)
    print("\n--- Running CKKS (TenSEAL) Simulation ---")
    ckks_results, _ = run_simulation('ckks', args.rounds, data, initial_params, pt_weights_history)
    all_results.extend(ckks_results)
    
    # 3. Run Packed Paillier (Proposed)
    print("\n--- Running Integer-Packed Paillier Simulation ---")
    packed_results, _ = run_simulation('packed_paillier', args.rounds, data, initial_params, pt_weights_history, packing_method=args.packing_method)
    all_results.extend(packed_results)
    
    # 4. Run Element-wise Paillier (Baseline)
    print(f"\n--- Running Element-wise Paillier Simulation ({args.ew_rounds} rounds) ---")
    print("WARNING: This baseline will be slow because it encrypts and decrypts each of the 55 weights individually.")
    ew_results, _ = run_simulation('elementwise_paillier', args.ew_rounds, data, initial_params, pt_weights_history[:args.ew_rounds])
    all_results.extend(ew_results)
    
    # Save results to CSV for analysis
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save results to CSV for analysis
    csv_file = f"{timestamp}_simulation_results_{args.rounds}rounds.csv"
    print(f"\nWriting results to {csv_file} for pandas analysis...")
    df = pd.DataFrame(all_results)
    df.to_csv(csv_file, index=False)
    print(f"Results successfully saved. Size: {len(df)} rows.")
    
    # Generate visual comparison charts
    generate_plots(df, args.rounds, timestamp)
    
    print(f"\nSimulation complete! You can load '{csv_file}' directly into a pandas DataFrame.")

if __name__ == '__main__':
    main()
