# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 09:39:05 2024

@author: StephanNef
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


st.set_page_config(
     page_title="IMP Functional Encryption",
     page_icon="https://upload.wikimedia.org/wikipedia/de/thumb/7/77/Uni_St_Gallen_Logo.svg/2048px-Uni_St_Gallen_Logo.svg.png",
     layout="wide",
)

st.logo("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/HSG_Logo_DE_RGB.svg/1024px-HSG_Logo_DE_RGB.svg.png",link="https://www.unisg.ch/de/")

st.markdown("""
# Quadratic Functional Encryption (QFE)
""")

st.info("**Reference**: C. Elisabetta, D. Catalano, D. Fiore, and R. Gay, “Practical Functional Encryption for Quadratic Functions with Applications to Predicate Encryption,” Cryptology ePrint Archive, 2017. https://eprint.iacr.org/2017/151")



with st.expander("📖 See Bounded QFE Scheme"):
    st.image("figures/qfe.png")


st.markdown("""
In **Bounded Quadratic Functional Encryption (QFE)**, we extend the concept of inner-product FE to allow for computing a **bilinear form** (or a quadratic polynomial)  $x^T F y$, where:
- $x$ and $y$ are vectors.
- $F$ is a matrix.

The scheme we consider here allows encryption of vectors $x$ and $y$, and keys corresponding to a specific matrix $F$. Decryption then yields the value $x^T F y$ (or a related encoded form), without revealing $x$, $y$, or $F$ beyond what is necessary.

Compared to the inner-product FE, this is more complex because:
- We deal with matrices $A, B$ and vectors $a, b, r, s$ that are used to encode the structure of the scheme.
- We operate in a bilinear group (pairing-based), and use pairings $e(g_1, g_2) = g_T$ to achieve the necessary functionality.
- The arithmetic takes place modulo a prime $p$, and careful construction ensures correctness and security.

We break down the scheme into the following steps:
- **Setup:** generates global parameters and master keys $MSK$ and public parameters $MPK$.
- **KeyGen:** uses $MSK$ and a matrix $F$ to produce a secret key $SK_F$.
- **Encrypt:** encrypts vectors $x$ and $y$ using $MSK$.
- **Decrypt:** given $SK_F$ and ciphertext of $x, y$, recovers the value $x^T F y$.

We also rely on several helper functions, see `qfehelpers.py` and PBC Lib (http://crypto.stanford.edu/pbc/manual/) that handle matrix and vector arithmetic modulo p, random generation, and more. These helpers are crucial due to the complexity of the operations.
""")

with st.expander("🤔 About **Bilinear Pairing Groups**:"):
    st.info("Reference, BN254 For The Rest Of Us, Jonathan Wang, https://hackmd.io/@jpw/bn254")
    st.markdown("""
                **Bilinear Pairings and Pairing Groups**

A **bilinear pairing** is a special kind of function defined on two elements from potentially different groups that outputs an element in a third group. 

Formally, a bilinear pairing is a map:
$e: G_1 \\times G_2 \\to G_T$

where $G_1$, $G_2$, and $G_T$ are groups of prime order $p$, and $e$ has the following properties:
1. **Bilinearity:** For all $u$ in $G_1$ and $v$ in $G_2$, and all integers $a, b$,

   $e(u^a, v^b) = e(u, v)^{ab}$

2. **Non-degeneracy:** $e(g_1, g_2) \neq 1$, where $g_1$ and $g_2$ are generators of $G_1$ and $G_2$ respectively. This ensures the pairing encodes meaningful algebraic structure.
3. **Efficiency:** The pairing and group operations can be computed efficiently.

Such pairings are often constructed using **elliptic curves** over finite fields with specially chosen properties. They are fundamental in modern cryptography, enabling advanced protocols such as Identity-Based Encryption (IBE), Attribute-Based Encryption (ABE), and our Quadratic Functional Encryption (QFE) scheme.



**Used PairingGroup("BN254")**

The `PairingGroup("BN254")` refers to initializing a specific type of elliptic curve pairing group known as a **Barreto–Naehrig (BN) curve** at a 254-bit security level. The BN254 curve is a popular choice in cryptographic libraries due to:
- **Well-studied Security:** The BN254 curve is believed to provide about 128-bit security against known attacks.
- **Efficient Implementation:** BN curves support fast pairing computations, making them practical for real-world cryptographic applications.
- **Standardization and Common Use:** BN254 is widely adopted, well-supported, and often cited in research papers and cryptographic software.

By using `PairingGroup("BN254")`, the code:
- Initializes an instance of the pairing group with generators $g_1$ in $G_1$, $g_2 in G_2$, and consequently a computed $g_T$ = $e(g_1, g_2) in G_T$.
- Provides methods for exponentiation, pairing evaluations, and random element generation.

**Why Use Pairing Groups in Quadratic Functional Encryption?**

Quadratic Functional Encryption (QFE) extends beyond the linear functionality of inner-product FE. Instead of just computing $\langle x, y \\rangle$, it enables the secure evaluation of a quadratic form $x^T F y$.

To achieve this more complex functionality securely:
- We need a **richer algebraic structure** that allows representing and manipulating these quadratic relations in the exponent.
- Pairing groups provide a way to combine elements from two source groups $G_1$ and $G_2$ and map them into $G_T$, making it possible to "encode" more complex computations in the exponent.
- The bilinearity of the pairing helps us transform multiplicative relations into additive ones in exponents, enabling advanced schemes like QFE.
- By carefully setting up matrices $A$, $B$ and vectors $a, b, r, s$, along with the pairing operation $e(g_1, g_2) = g_T$, the scheme leverages these pairings to ensure that only the intended quadratic value $x^T F y$ can be extracted during decryption—without revealing the underlying vectors or matrix.

In essence, the **pairing** and the **pairing-friendly elliptic curve (BN254)** enable the construction of a secure, advanced functional encryption scheme for quadratic forms, guaranteeing correct functionality and strong security properties that would be hard to achieve with simpler (non-pairing) groups.

                """)




st.write("---")
st.subheader("1. Step: Setup Phase")


st.markdown("We work in a bilinear group setting with a prime order $p$, we have:")
st.latex("G_1, G_2, GT")
st.markdown("...which are groups of prime order  $p$ with a **bilinear pairing**:")
st.latex("e: G_1 \\times G_2 \\to G_T")

st.markdown("We choose:")
st.latex("g_1 \in G_1, \; g_2 \in G_2,\; g_T = e(g_1, g_2) \in GT.")

st.markdown("We generate matrices $A, B$ and vectors $a, b$ with specific properties:")
st.latex("A, B \in Z_p^{(k+1) \\times k}, \quad a, b \in Z_p^{k+1}")
st.markdown("...and similarly choose random matrices $r$ and $s$ of appropriate dimensions. These encode the complexity of quadratic computations.")

st.markdown("The Master Secret Key $MSK$ includes $A, a, B, b, r, s$.")
st.latex("\\text{MSK} = (A, a, B, b, r, s)")

st.markdown("The Master Public Key $MPK$ includes  $g_1, g_2, g_T$ and  $\langle b, a \\rangle$ (an inner product in $Z_p$):")
st.latex("\\text{MPK} = (g_1, g_2, g_T, \\langle b,a \\rangle)")

st.markdown("**In summary:**")
st.markdown("- Generate group parameters and bases $g_1, g_2$")
st.markdown("- Construct $A, B, a, b, r, s$ with certain linear algebraic properties.")
st.markdown("- Output $MSK$ and $MPK$")

with st.expander("Applied Code for Setup:", expanded=True):
    st.code("""\
def setup(self, p=p_order, k=None):
    m = k
    n = k - 1
    # Generate A,a and B,b using generate_matrix_Lk
    A, a = generate_matrix_Lk(p, k)
    B, b = generate_matrix_Lk(p, k)
    # Generate random r,s
    r = random_int_matrix(1, p, n, k)
    s = random_int_matrix(1, p, m, k)
    
    # mpk and msk
    mpk = MPK(self.g1, self.g2, self.gt, inner_product_mod(b, a, p))
    msk = MSK(A, a, B, b, r, s)
    return mpk, msk
    """, language='python')





st.write("---")
st.subheader("2. Step: Key Generation (KeyGen)")


st.markdown("Given $MSK = (A, a, B, b, r, s)$ and a matrix:")
st.latex("F \in Z_p^{n \\times m}")
st.markdown("...we produce a secret key $SK_F$.")

st.markdown("We choose a random $u$ in $Z_p$ and compute a complex exponentiation involving $A, B, r, s, F$ to derive:")
st.latex("K = g_1^{(\\text{some linear combination of F, A, B, r, s}) - u}, \quad \\tilde{K} = g_2^u")
st.latex("=")
st.latex(r"K = g_1^{\left(\sum_{i,j} F_{i,j} \cdot (r_i^T A^T B s_j)\right) - u}, \quad \tilde{K} = g_2^u")

st.markdown("The idea is that $K$ and $\\tilde{K}$ encode the function $F$ in the exponent. Later, when combined with the ciphertext, this will allow recovering $x^T F y$.")

st.markdown("**In summary:**")
st.markdown("- Input: $MSK$ and $F$.")
st.markdown("- Pick random $u$ in $Z_p$")
st.markdown("- Compute $K, \\tilde{K}$ based on the bilinear form involving $F$.")
st.markdown("- Output $SK_F$ = ($K$, $\\tilde{K}$)")

with st.expander("Applied Code for KeyGen:", expanded=True):
    st.code("""\
def keygen(self, p=p_order, mpk=None, msk=None, F=None):
    u = random.randint(0, p - 1)
    A = msk.A
    B = msk.B
    r = msk.r
    s = msk.s
    g1 = mpk.g1
    g2 = mpk.g2
    n = len(r)
    m = len(s)
    
    # Compute the exponent sum for K using r, s, A, B and F
    # The code computes a complex sum = Σ_i,j F[i,j]*(...) that encodes F
    sum_exp = 0
    ATB = matrix_multiply_mod(transpose_matrix(A), B, p)
    for i in range(n):
        riT_AT_B = vector_matrix_multiply_mod(r[i], ATB, p)
        for j in range(m):
            riT_AT_B_sj = vector_multiply_mod(riT_AT_B, s[j], p)
            sum_exp += (F[i][j] * riT_AT_B_sj) % p
    
    # K = g1^{sum_exp - u}, K_tilde = g2^{u}
    K = g1 ** int(sum_exp - u)
    K_tilde = g2 ** int(u)
    skF = SKF(K, K_tilde)
    return skF
    """, language='python')





st.write("---")
st.subheader("3. Step: Encryption Phase")


st.markdown("To encrypt vectors $x$ in $Z_p^n$ and $y$ in $Z_p^m$ under $MSK$, we compute:")
st.latex("c_i = A r_i + b x_i, \quad \\text{for } i=1,...,n")
st.latex(r"\tilde{c}_j = B s_j + a y_j, \quad \text{for } j=1,\ldots,m")


st.markdown("Here, each $c_i$ and $\\tilde{c}_j$ are vectors in $Z_p^{k+1}$ encoding the encrypted components of $x$ and $y$.")
st.markdown("The ciphertext is then:")
st.latex(r"\text{CT}_{x,y} = (\{c_i\}, \{\tilde{c}_j\})")


st.markdown("**In summary:**")
st.markdown("- Input: $MSK$ and vectors $x, y$")
st.markdown("- Compute $c_i$ and $\\tilde{c}_j$ as linear transformations of $x, y$ using $A, B, a, b, r, s$.")
st.markdown("- Output ciphertext $CT_{x,y}$")

with st.expander("Applied Code for Encryption", expanded=True):
    st.code("""\
def encrypt(self, msk, x, y):
    A = msk.A
    B = msk.B
    a = msk.a
    b = msk.b
    r = msk.r
    s = msk.s

    c = [
        matrix_vector_multiply(A, r[i]) + scalar_multiply(b, x[i])
        for i in range(len(x))
    ]
    c_tilde = [
        matrix_vector_multiply(B, s[j]) + scalar_multiply(a, y[j])
        for j in range(len(y))
    ]

    CT_xy = CTXY(c, c_tilde)
    return CT_xy
    """, language='python')





st.write("---")
st.subheader("4. Step: Decryption Phase")


st.markdown("Given $SK_F = (K, 	ilde{K})$ and $CT_{x,y} = (\\{c_i\\}, \\{\\tilde{c}_j\\})$, we aim to recover $x^T F y$.")
st.markdown("Intuitively, we combine the ciphertext components with the secret key and use pairings to cancel out the random factors and isolate $x^T F y$.")

st.markdown("Decryption computes a value:")
st.latex("D = g_T^{x^T F y}")
st.markdown("...up to some additional factors that we can solve for by discrete logarithm, similar to the inner-product scheme.")

st.markdown("The code attempts to find the integer:")
st.latex(r"v \text{ such that } D = g_T^{v \cdot \langle b,a \rangle}")

st.markdown("...yielding the final result:")           
st.latex(r"v = x^T F y")


st.markdown("**In summary:**")
st.markdown("- Input: $SK_F$ and $CT_{x,y}$")
st.markdown("- Perform pairing-based combination to isolate  $x^T F y$")
st.markdown("- Solve discrete log to recover $x^T F y$")

with st.expander("Applied Code for Decryption:", expanded=True):
    st.code("""\
def decrypt(self, p=p_order, mpk=None, skF=None, CT_xy=None, n=None, m=None, F=None):
    c = CT_xy.c
    c_tilde = CT_xy.c_tilde
    K = skF.K
    K_tilde = skF.K_tilde
    g1 = mpk.g1
    g2 = mpk.g2
    gt = mpk.gt

    # Compute exponent exp = Σ_i,j F[i,j]* (c_i · c_tilde_j)
    exp = 0
    for i in range(n):
        for j in range(m):
            exp += int(F[i][j] * int(dot_product(c[i], c_tilde[j])))

    D = gt**exp
    # Adjust D by secret key parts K, K_tilde to isolate the result
    D *= -(self.group.pair_prod(K, g2))
    D *= -(self.group.pair_prod(g1, K_tilde))

    # Solve discrete log: find v s.t. D = gt^{v * <b,a>}
    # A naive approach tries v = 1,...,p until D matches
    v = 0
    res = self.group.random(GT)
    inner = mpk.baT
    while D != res and v < p:
        v += 1
        res = gt ** int(v * inner)

    return v
    """, language='python')


st.write("---")
st.subheader("Helper Functions and Importance")

st.markdown("""
We rely on a variety of helper functions (see code below) for operations like:
- **Matrix Multiplication Mod p**: `matrix_multiply_mod`, `vector_matrix_multiply_mod`
- **Inner Product Mod p**: `inner_product_mod`
- **Random Generation**: `random_int_matrix`, `random_vector`
- **Matrix and Vector Operations**: `transpose_matrix`, `matrix_vector_multiply`, `dot_product`

These helpers are crucial because the QFE scheme's complexity lies in combining algebraic structures from bilinear groups with heavy modular arithmetic. Without these helpers, the main scheme implementation would be very lengthy and hard to read.

They also ensure modular arithmetic correctness and allow us to focus on the cryptographic logic rather than low-level arithmetic details.
""")

with st.expander("See Python Script for Helper Functions:"):
    st.code("""
            import random


class MPK:
    g1 = None
    g2 = None
    gt = None
    baT = None

    def __init__(self, g1, g2, gt, baT):
        self.g1 = g1
        self.g2 = g2
        self.gt = gt
        self.baT = baT


class MSK:
    A = None
    a = None
    B = None
    b = None
    r = None
    s = None

    def __init__(self, A, a, B, b, r, s):
        self.A = A
        self.a = a
        self.B = B
        self.b = b
        self.r = r
        self.s = s


class SKF:
    K = None
    K_tilde = None

    def __init__(self, K, K_tilde):
        self.K = K
        self.K_tilde = K_tilde


class CTXY:
    c = None
    c_tilde = None

    def __init__(self, c, c_tilde):
        self.c = c
        self.c_tilde = c_tilde


def vector_multiply_mod(vector1, vector2, p):
    \"""
    Multiplies two vectors element-wise under modulo p.

    Args:
        vector1 (list[int]): The first vector.
        vector2 (list[int]): The second vector.
        p (int): The modulus.

    Returns:
        list[int]: The resulting vector after element-wise multiplication under modulo p.
    \"""
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must have the same length")

    sum = 0
    for i in range(len(vector1)):
        sum += vector1[i] * vector2[i]

    return sum % p


def matrix_multiply_mod(A, B, p):
    \"""
    Multiplies two matrices A and B under modulo p.

    Args:
        A (list[list[int]]): The first matrix.
        B (list[list[int]]): The second matrix.
        p (int): The modulus.

    Returns:
        list[list[int]]: The resulting matrix after multiplication under modulo p.
    \"""
    if len(A[0]) != len(B):
        raise ValueError("Number of columns in A must match number of rows in B")

    # Initialize the result matrix with zeros
    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]

    # Perform matrix multiplication with modulo p
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] = (result[i][j] + A[i][k] * B[k][j]) % p

    return result


def vector_matrix_multiply_mod(vector, matrix, p):
    \"""
    Multiplies a vector by a matrix under modulo p.

    Args:
        vector (list[int]): The input vector.
        matrix (list[list[int]]): The input matrix.
        p (int): The modulus.

    Returns:
        list[int]: The resulting vector after multiplication under modulo p.
    \"""
    if len(vector) != len(matrix):
        raise ValueError(
            "The length of the vector must match the number of rows in the matrix"
        )

    result = [0 for _ in range(len(matrix[0]))]

    for j in range(len(matrix[0])):
        for i in range(len(vector)):
            result[j] = (result[j] + vector[i] * matrix[i][j]) % p

    return result


def modular_inverse(a, p):
    \"""
    Computes the modular inverse of a with respect to p using the extended Euclidean algorithm.
    Args:
        a (int): The number to invert.
        p (int): The modulus.
    Returns:
        int: The modular inverse of a modulo p.
    \"""
    t, new_t = 0, 1
    r, new_r = p, a

    while new_r != 0:
        quotient = r // new_r
        t, new_t = new_t, t - quotient * new_t
        r, new_r = new_r, r - quotient * new_r

    if r > 1:
        raise ValueError(f"{a} has no modular inverse modulo {p}")
    if t < 0:
        t += p

    return t


def generate_matrix_Lk(p, k):
    \"""
    Generates a matrix and a vector for the given p and k with arbitrary long integers.

    Args:
        p (int): The modulus for the modular arithmetic.
        k (int): The size of the matrix and vector.

    Returns:
        tuple: A tuple (matrix, vector), where:
            - matrix is a (k+1) x k matrix filled with random values and ones on the last row.
            - vector is a (k+1) x 1 vector with modular inverses and -1 as the last value.
    \"""
    # Initialize matrix and vector
    matrix = [[0 for _ in range(k)] for _ in range(k + 1)]
    vector = [0 for _ in range(k + 1)]

    for i in range(k):
        val = random.randint(1, p - 1)  # Random integer in the range [1, p-1]
        matrix[i][i] = val
        vector[i] = modular_inverse(val, p)

    # Fill the last row of the matrix with ones
    matrix[k] = [1 for _ in range(k)]
    vector[k] = -1

    return matrix, vector


def generate_matrix_Lk_AB(p, k):
    \"""
    Generates a matrix and a vector for the given p and k with arbitrary long integers.

    Args:
        p (int): The modulus for the modular arithmetic.
        k (int): The size of the matrix and vector.

    Returns:
        tuple: A tuple (matrix, vector), where:
            - matrix is a (k+1) x k matrix filled with random values and ones on the last row.
            - vector is a (k+1) x 1 vector with modular inverses and -1 as the last value.
    \"""
    # Initialize matrix and vector
    A, a = generate_matrix_Lk(p, k)
    B, b = generate_matrix_Lk(p, k)

    while inner_product_mod(b, a, p) != 1:
        B, b = generate_matrix_Lk(p, k)

    return A, a, B, b


def matrix_vector_dot(matrix, vector, p):
    \"""
    Computes the dot product of a matrix and a vector, reducing results modulo p.
    \"""
    if len(matrix[0]) != len(vector):
        raise ValueError(
            "Number of columns in the matrix must match the length of the vector"
        )

    # Compute the dot product row-wise, reducing modulo p
    result = [
        sum((row[i] * vector[i]) for i in range(len(vector))) % p for row in matrix
    ]
    return result


def vector_matrix_dot_mod(vector, matrix, p):
    \"""
    Computes the dot product of a vector and a matrix modulo p.

    Args:
        vector (list[int]): The input vector (1D list).
        matrix (list[list[int]]): The input matrix (2D list).
        p (int): The modulus.

    Returns:
        list[int]: Resultant vector after the dot product, reduced modulo p.

    Raises:
        ValueError: If the number of elements in the vector does not match the number of rows in the matrix.
    \"""
    # Ensure vector and matrix dimensions match
    if len(vector) != len(matrix[0]):
        raise ValueError(
            "Number of elements in the vector must match the number of rows in the matrix."
        )

    # Compute the dot product modulo p
    result = [sum(vector[j] * row[j] for j in range(len(vector))) % p for row in matrix]
    return result


def matrix_vector_multiply(matrix, vector):
    \"""
    Multiplies a matrix by a vector.

    Args:
        matrix (list[list[float]]): The input matrix.
        vector (list[float]): The input vector.

    Returns:
        list[float]: The resulting vector after multiplication.
    \"""
    if len(matrix[0]) != len(vector):
        raise ValueError(
            "Number of columns in the matrix must match the length of the vector"
        )

    result = [
        sum(matrix[i][j] * vector[j] for j in range(len(vector)))
        for i in range(len(matrix))
    ]
    return result


def matrix_vector_multiply_mod(matrix, vector, p):
    \"""
    Multiplies a matrix by a vector.

    Args:
        matrix (list[list[float]]): The input matrix.
        vector (list[float]): The input vector.

    Returns:
        list[float]: The resulting vector after multiplication.
    \"""
    if len(matrix[0]) != len(vector):
        raise ValueError(
            "Number of columns in the matrix must match the length of the vector"
        )

    result = matrix_vector_multiply(matrix, vector) % p
    return result


def inner_product_mod(vector1, vector2, p):
    \"""
    Computes the inner product (dot product) of two vectors modulo p.

    Args:
        vector1 (list[int or float]): The first vector.
        vector2 (list[int or float]): The second vector.
        p (int): The modulus.

    Returns:
        int: The inner product of the two vectors modulo p.

    Raises:
        ValueError: If the vectors are not of the same length.
    \"""
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must have the same length")

    # Compute the inner product modulo p
    return sum(vector1[i] * vector2[i] for i in range(len(vector1))) % p


def transpose_vector(vector):
    \"""
    Transposes a vector (1D list to 2D column vector).

    Args:
        vector (list): A 1D list representing the vector.

    Returns:
        list[list]: A 2D list representing the transposed vector (column vector).
    \"""
    return [[element] for element in vector]


def transpose_matrix(matrix):
    \"""
    Transposes a given matrix.

    Args:
        matrix (list[list[int or float]]): A 2D list representing the matrix.

    Returns:
        list[list[int or float]]: The transposed matrix.
    \"""
    # Ensure the matrix is not empty
    if not matrix or not matrix[0]:
        raise ValueError("Matrix cannot be empty")

    # Transpose the matrix
    transposed = [
        [matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))
    ]
    return transposed


def random_int_matrix(low, high, n, m):
    \"""
    Generates a matrix of random integers in the range [low, high) with dimensions (n, m).

    Args:
        low (int): The lower bound (inclusive).
        high (int): The upper bound (exclusive).
        n (int): Number of rows in the matrix.
        m (int): Number of columns in the matrix.

    Returns:
        list[list[int]]: A 2D list (matrix) of random integers.
    \"""
    return [[random.randint(low, high - 1) for _ in range(m)] for _ in range(n)]


def random_vector(low, high, n):
    \"""
    Generates a random vector with elements from range [a, b].

    Args:
        a (int): The lower bound (inclusive).
        b (int): The upper bound (inclusive).
        n (int): The size of the vector.

    Returns:
        list[int]: A vector (list) of random integers.
    \"""
    return [random.randint(low, high - 1) for _ in range(n)]


def transpose(matrix):
    \"""
    Transposes a given matrix.

    Args:
        matrix (list[list[float]]): The matrix to transpose.

    Returns:
        list[list[float]]: The transposed matrix.
    \"""
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]


def dot_product(vector1, vector2):
    \"""
    Computes the dot product of two vectors.

    Args:
        vector1 (list[float]): The first vector.
        vector2 (list[float]): The second vector.

    Returns:
        float: The dot product of the two vectors.
    \"""
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must have the same length.")
    return sum(x * y for x, y in zip(vector1, vector2))


def compute_rT_AT_for_row(r_i, A):
    \"""
    Computes r_i^T * A^T for a single row r_i and the given matrix A.

    Args:
        r_i (list[float]): A single row vector.
        A (list[list[float]]): A 2D list representing the matrix.

    Returns:
        list[float]: Resulting vector after computing r_i^T * A^T.
    \"""
    # Transpose A
    A_T = transpose(A)

    # Compute dot product of r_i with each row of A^T
    result = [dot_product(r_i, row) for row in A_T]
    return result


def matrix_dot_product(A, B):
    \"""
    Computes the dot product of two matrices.

    Args:
        A (list[list[float]]): The first matrix.
        B (list[list[float]]): The second matrix.

    Returns:
        list[list[float]]: The resulting matrix after the dot product.

    Raises:
        ValueError: If the number of columns in A does not match the number of rows in B.
    \"""
    if len(A[0]) != len(B):
        raise ValueError("Number of columns in A must match number of rows in B")

    # Transpose B to make the dot product easier
    B_T = transpose(B)

    # Compute the dot product
    result = [[dot_product(row, col) for col in B_T] for row in A]
    return result


def vector_transposed_mul_matrix_mul_vector(x, F, y, p):
    # Step 1: Compute F * y (mod p)
    Fy = matrix_vector_multiply(F, y)

    # Step 2: Compute x^T * (Fy) (mod p)
    xTFy = sum((x[i] * Fy[i]) % p for i in range(len(x))) % p

    return xTFy


def scalar_multiply(vector, scalar):
    \"""
    Computes the dot product of two matrices.

    Args:
        vector (list[float]): The vector .
        scalar (int): The scalar.

    Returns:
        list[float]: The resulting vector after multiplication with the scalar.
    \"""
    return [scalar * element for element in vector]

            """, language='python')

