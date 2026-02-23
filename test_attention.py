import numpy as np
from attention import scaled_dot_product_attention

def testar_calculo():
    Q = np.array([[1.0, 0.0], [0.0, 1.0]])
    K = np.array([[1.0, 0.0], [0.0, 1.0]])
    V = np.array([[10.0], [20.0]])
    
    saida, pesos = scaled_dot_product_attention(Q, K, V)
    
    # Verifica se os pesos somam 1 (Requisito do Softmax) [cite: 35]
    assert np.allclose(pesos.sum(axis=1), 1.0)
    print("Teste de soma dos pesos: OK")
    print("Saída calculada:\n", saida)

if __name__ == "__main__":
    testar_calculo()