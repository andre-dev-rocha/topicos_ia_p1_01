import numpy as np

def softmax(matriz):
    # Versão simples sem o truque de estabilidade ainda
    exponenciais = np.exp(matriz)
    return exponenciais / np.sum(exponenciais, axis=1, keepdims=True)

def scaled_dot_product_attention(Q, K, V):
    # Cálculo básico: softmax(QK^T)V sem o fator de escala sqrt(dk)
    pontuacoes = Q @ K.T
    pesos_atencao = softmax(pontuacoes)
    saida = pesos_atencao @ V
    return saida, pesos_atencao