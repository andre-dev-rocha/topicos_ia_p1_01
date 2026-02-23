# -*- coding: utf-8 -*-
"""
Implementação do mecanismo Scaled Dot-Product Attention
conforme descrito em "Attention Is All You Need" (Vaswani et al., 2017).
"""

import numpy as np
from numpy import ndarray


def softmax(matriz: ndarray) -> ndarray:
    """
    Aplica a função softmax linha a linha em uma matriz 2D.

    Utiliza o truque de estabilidade numérica: subtrai o valor máximo
    de cada linha antes de exponenciar, evitando overflow numérico
    sem alterar o resultado matemático.

    Argumentos:
        matriz (ndarray): Matriz 2D de shape (n, m) com valores reais.

    Retorno:
        ndarray: Matriz de mesma shape com softmax aplicado linha a linha.
                 Cada linha soma 1.0.

    Exceções:
        ValueError: Se a entrada não for uma matriz 2D.
    """
    if matriz.ndim != 2:
        raise ValueError(
            f"A entrada deve ser uma matriz 2D. Shape recebido: {matriz.shape}"
        )

    valores_deslocados = matriz - matriz.max(axis=1, keepdims=True)
    exponenciais = np.exp(valores_deslocados)
    somas_por_linha = exponenciais.sum(axis=1, keepdims=True)

    return exponenciais / somas_por_linha


def scaled_dot_product_attention(
    Q: ndarray, K: ndarray, V: ndarray
) -> tuple[ndarray, ndarray]:
    """
    Calcula o Scaled Dot-Product Attention conforme a fórmula:

        Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V

    Onde dₖ é a dimensão das chaves (última dimensão de K), utilizada
    como fator de escala para manter a variância dos scores controlada.

    Argumentos:
        Q (ndarray): Matriz de queries de shape (n_queries, d_k).
        K (ndarray): Matriz de keys de shape (n_keys, d_k).
        V (ndarray): Matriz de values de shape (n_keys, d_v).

    Retorno:
        tuple[ndarray, ndarray]: Uma tupla contendo:
            - output (ndarray): Resultado da atenção, shape (n_queries, d_v).
            - attention_weights (ndarray): Pesos de atenção normalizados,
              shape (n_queries, n_keys). Cada linha soma 1.0.

    Exceções:
        ValueError: Se Q, K ou V não forem matrizes 2D.
        ValueError: Se as dimensões de Q e K forem incompatíveis (d_k diferente).
        ValueError: Se o número de linhas de K e V forem incompatíveis.
    """
    if Q.ndim != 2 or K.ndim != 2 or V.ndim != 2:
        raise ValueError(
            "Q, K e V devem ser matrizes 2D. "
            f"Shapes recebidos: Q={Q.shape}, K={K.shape}, V={V.shape}"
        )

    if Q.shape[1] != K.shape[1]:
        raise ValueError(
            "A dimensão das features de Q e K deve ser igual (d_k). "
            f"Q.shape[1]={Q.shape[1]}, K.shape[1]={K.shape[1]}"
        )

    if K.shape[0] != V.shape[0]:
        raise ValueError(
            "O número de linhas de K e V deve ser igual (n_keys). "
            f"K.shape[0]={K.shape[0]}, V.shape[0]={V.shape[0]}"
        )

    dimensao_chaves = K.shape[1]
    fator_escala = np.sqrt(dimensao_chaves)

    pontuacoes = Q @ K.T
    pontuacoes_normalizadas = pontuacoes / fator_escala
    pesos_atencao = softmax(pontuacoes_normalizadas)
    saida = pesos_atencao @ V

    return saida, pesos_atencao
