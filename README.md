# Scaled Dot-Product Attention -- LAB P1-01 -- André Rocha 

## Descrição

O mecanismo de **atenção** é o componente central da arquitetura Transformer, proposta no paper *"Attention Is All You Need"* (Vaswani et al., 2017). Ao contrário das redes recorrentes, ele processa todas as posições de uma sequência em paralelo, calculando para cada posição o quanto ela deve "prestar atenção" nas demais.

O **Scaled Dot-Product Attention** recebe três matrizes: **Q** (queries), **K** (keys) e **V** (values). A ideia é que cada query "consulta" todas as keys para descobrir quais values são mais relevantes. O resultado é uma combinação ponderada dos values, onde os pesos refletem a afinidade entre queries e keys.

A operação é inteiramente diferenciável e eficiente computacionalmente, podendo ser paralelizada via multiplicação de matrizes — o que torna o Transformer escalável para sequências longas sem o custo sequencial das RNNs.

---

## Como rodar

**Pré-requisitos:** Python 3.10 ou superior e a biblioteca NumPy.

```bash
pip install numpy
python test_attention.py
```

---

## Fórmula de Referência

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

---

## Explicação do Scaling Factor (√dₖ)

Quando a dimensão das keys **dₖ** é grande, os produtos escalares `Q · Kᵀ` crescem em magnitude de forma proporcional a **dₖ**. Isso ocorre porque cada elemento do produto escalar é uma soma de **dₖ** termos, e a variância dessa soma cresce linearmente com **dₖ**.

Valores muito grandes empurram o softmax para regiões de saturação, onde o gradiente é próximo de zero — dificultando o aprendizado por backpropagation. Dividir os scores por **√dₖ** normaliza a variância para ~1, independente da dimensão, mantendo o softmax em uma região de gradiente saudável.

---