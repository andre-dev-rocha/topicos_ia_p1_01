import numpy as np

def softmax(matriz):
    # Aplicando exponencial simples (ainda sem o truque de estabilidade)
    exponenciais = np.exp(matriz)
    return exponenciais / np.sum(exponenciais, axis=1, keepdims=True)

def scaled_dot_product_attention(Q, K, V):
    # d_k é a dimensão das chaves (número de colunas de K) 
    dimensao_chaves = K.shape[1]
    fator_escala = np.sqrt(dimensao_chaves) # 

    # Cálculo do produto escalar QK^T 
    pontuacoes = Q @ K.T
    
    # Aplicação do Scaling Factor 
    pontuacoes_normalizadas = pontuacoes / fator_escala
    
    # Aplicação do Softmax por linha [cite: 35]
    pesos_atencao = softmax(pontuacoes_normalizadas)
    
    # Multiplicação pelo Value (V)
    saida = pesos_atencao @ V
    return saida, pesos_atencao