import numpy as np
from attention import scaled_dot_product_attention

def teste_formato_saida():
    Q = np.random.rand(3, 4)
    K = np.random.rand(5, 4)
    V = np.random.rand(5, 2)
    saida, pesos = scaled_dot_product_attention(Q, K, V)
    assert saida.shape == (3, 2)
    print("Teste de formato passou!")

if __name__ == "__main__":
    teste_formato_saida()