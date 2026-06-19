class Server:
    def __init__(self):
        pass

    def aggregate(self, scheme, client_updates):
        """
        Aggregate client model updates homomorphically.
        The server does not possess the decryption key and only performs addition.
        
        scheme: 'plaintext', 'elementwise_paillier', 'packed_paillier', or 'ckks'
        client_updates: list of encrypted/plaintext updates from clients.
          - For 'plaintext': list of 1D numpy arrays.
          - For 'elementwise_paillier': list of lists of phe.paillier.EncryptedNumber.
          - For 'packed_paillier': list of phe.paillier.EncryptedNumber (each represents all 55 parameters).
          - For 'ckks': list of tenseal.CKKSVector.
        """
        if not client_updates:
            return None

        if scheme == 'plaintext':
            # Sum of arrays element-wise
            sum_params = client_updates[0].copy()
            for c in range(1, len(client_updates)):
                sum_params += client_updates[c]
            return sum_params

        elif scheme == 'elementwise_paillier':
            # Sum of parameters across clients element-wise.
            # In phe, adding two EncryptedNumber objects performs homomorphic addition.
            num_params = len(client_updates[0])
            aggregated = []
            for i in range(num_params):
                val = client_updates[0][i]
                for c in range(1, len(client_updates)):
                    val = val + client_updates[c][i]
                aggregated.append(val)
            return aggregated

        elif scheme == 'packed_paillier':
            # Sum of the single packed ciphertext across clients.
            # In phe, adding two EncryptedNumber objects performs homomorphic addition.
            val = client_updates[0]
            for c in range(1, len(client_updates)):
                val = val + client_updates[c]
            return val

        elif scheme == 'ckks':
            # Sum of CKKS vectors homomorphically.
            # In tenseal, adding two CKKSVector objects performs homomorphic addition.
            val = client_updates[0]
            for c in range(1, len(client_updates)):
                val = val + client_updates[c]
            return val

        else:
            raise ValueError(f"Unknown encryption scheme: {scheme}")
