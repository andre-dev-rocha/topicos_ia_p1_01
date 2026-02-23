# -*- coding: utf-8 -*-
"""
Testes de validação para o mecanismo Scaled Dot-Product Attention.

Execução: python test_attention.py
"""

import numpy as np
import numpy.testing as npt
from attention import scaled_dot_product_attention

# Dados de entrada fixos para reprodutibilidade
Q = np.array([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0, 0.0],
], dtype=np.float64)

K = np.array([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0, 1.0],
], dtype=np.float64)

V = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
], dtype=np.float64)


def _calcular_saida_esperada() -> np.ndarray:
    """Calcula manualmente a saída esperada para Q, K, V definidos acima."""
    dimensao_chaves = K.shape[1]
    fator_escala = np.sqrt(dimensao_chaves)

    pontuacoes = Q @ K.T
    pontuacoes_normalizadas = pontuacoes / fator_escala

    # softmax linha a linha
    deslocados = pontuacoes_normalizadas - pontuacoes_normalizadas.max(axis=1, keepdims=True)
    exponenciais = np.exp(deslocados)
    pesos = exponenciais / exponenciais.sum(axis=1, keepdims=True)

    return pesos @ V, pesos


SAIDA_ESPERADA, PESOS_ESPERADOS = _calcular_saida_esperada()


# Funções de teste

def teste_pesos_somam_um(pesos_atencao: np.ndarray) -> bool:
    """Verifica que cada linha dos pesos de atenção soma 1.0."""
    try:
        somas_por_linha = pesos_atencao.sum(axis=1)
        npt.assert_array_almost_equal(
            somas_por_linha,
            np.ones(pesos_atencao.shape[0]),
            decimal=6,
            err_msg="Linhas dos pesos de atenção não somam 1.0"
        )
        print("  [Sucesso] teste_pesos_somam_um")
        return True
    except AssertionError as erro:
        print(f"  [Fracasso] teste_pesos_somam_um: {erro}")
        return False


def teste_formato_saida(saida: np.ndarray) -> bool:
    """Verifica que o formato da saída é (n_consultas, d_valores)."""
    try:
        formato_esperado = (Q.shape[0], V.shape[1])
        assert saida.shape == formato_esperado, (
            f"Formato esperado {formato_esperado}, obtido {saida.shape}"
        )
        print("  [Sucesso] teste_formato_saida")
        return True
    except AssertionError as erro:
        print(f"  [Fracasso] teste_formato_saida: {erro}")
        return False


def teste_corretude_numerica(saida: np.ndarray) -> bool:
    """Compara a saída calculada com o valor esperado calculado manualmente."""
    try:
        npt.assert_array_almost_equal(
            saida,
            SAIDA_ESPERADA,
            decimal=6,
            err_msg="Saída numérica diverge do esperado"
        )
        print("  [Sucesso] teste_corretude_numerica")
        return True
    except AssertionError as erro:
        print(f"  [Fracasso] teste_corretude_numerica: {erro}")
        return False


# Execução

def main():

    print("  Scaled Dot-Product Attention — Testes de Validação")

    print("\n-- Entradas ---------------------------------------------")
    print(f"\nQ (consultas) — shape {Q.shape}:\n{Q}")
    print(f"\nK (chaves)    — shape {K.shape}:\n{K}")
    print(f"\nV (valores)   — shape {V.shape}:\n{V}")

    saida, pesos_atencao = scaled_dot_product_attention(Q, K, V)

    print("\n-- Saídas -----------------------------------------------")

    print(f"\nPesos de Atenção — shape {pesos_atencao.shape}:")
    print(np.round(pesos_atencao, 6))
    print(f"\nSaída — shape {saida.shape}:")
    print(np.round(saida, 6))

    print("\n-- Resultados dos Testes -------------------------------\n")
    resultados = [
        teste_pesos_somam_um(pesos_atencao),
        teste_formato_saida(saida),
        teste_corretude_numerica(saida),
    ]

    aprovados = sum(resultados)
    total = len(resultados)

    print(f"  {aprovados}/{total} testes passaram.")



if __name__ == "__main__":
    main()
