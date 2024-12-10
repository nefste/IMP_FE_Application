# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 09:39:05 2024

@author: StephanNef
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.linalg import null_space
import numpy as np

st.set_page_config(
     page_title="IMP Functional Encryption",
     page_icon="🔐",
     layout="wide",
)

st.logo("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/HSG_Logo_DE_RGB.svg/1024px-HSG_Logo_DE_RGB.svg.png",link="https://www.unisg.ch/de/")


st.sidebar.title("Parameters")
p = st.sidebar.number_input('Prime number p', min_value=3, value=101, step=2)
n = st.sidebar.number_input('Dimension n of vector x', min_value=1, value=3)
m = st.sidebar.number_input('Dimension m of vector y', min_value=1, value=3)


p = int(p)
n = int(n)
m = int(m)

st.markdown("""
# Fully Secure Inner Product Functional Encryption (IPFE)
""")


with st.expander("📖 See IPFE-FULLYSEC Scheme"):
    st.image("figures/ipfe-fullsec.png")

st.info("**Reference**: S. Agrawal, B. Libert, and D. Stehle, “Fully Secure Functional Encryption for Inner Products, from Standard Assumptions,” Cryptology ePrint Archive, 2015. https://eprint.iacr.org/2015/608 ")

st.warning("⚠️ This site is still under construction ⚠️")


# st.markdown("""
# We will implement a functional encryption (FE) scheme step by step, explaining each part and providing the corresponding code.
# """)

# st.markdown("## 1. Functional Encryption Overview")

# st.markdown("The goal is to create a functional encryption scheme that works with bilinear maps. We encrypt pairs of vectors and define decryption keys for specific functions represented by a matrix. This allows us to compute through a bilinear map.")

# st.latex(r"(x, y)")
# st.latex(r"F")
# st.latex(r"x^\top F y")

# st.markdown("## 2. Encryption Setup")

# st.markdown("**Explanation:**")

# st.markdown("- **Parameters**: We choose a prime number for our finite field.")
# st.latex(rf"p = {p}")
# st.latex(r"\mathbb{{Z}}_p")

# st.markdown("- **Matrices and Vectors**: We define dimensions for our vectors.")
# st.latex(rf"n = {n}")
# st.latex(rf"m = {m}")
# st.latex(r"\text{Vectors } x \text{ and } y")


# setup_code = f'''
# import numpy as np

# # Set parameters
# p = {p}  # Prime number
# n = {n}  # Dimension of vector x
# m = {m}  # Dimension of vector y

# # Finite field
# Zp = np.arange(p)
# '''

# st.code(setup_code, language='python')

# Zp = np.arange(p)

# st.markdown("## 3. Key Generation (KeyGen)")

# st.markdown("**Explanation:**")

# st.markdown("""
# - We generate random matrices \( A \) and \( B \) from a random distribution.
# - We find vectors \( a_{\perp} \) and \( b_{\perp} \) such that:
# """)

# st.latex(r"a_{\perp} \in \text{Null}(A^\top)")
# st.latex(r"b_{\perp} \in \text{Null}(B^\top)")

# st.markdown("- The master secret key (`msk`) consists of \( A \), \( a_{\perp} \), \( B \), and \( b_{\perp} \).")


# keygen_code = '''
# # Parameter k for matrix dimensions
# k = max(n, m)

# # Generate random matrices A and B
# A = np.random.randint(0, p, (k, n))
# B = np.random.randint(0, p, (k, m))

# # Function to compute a vector in the null space modulo p
# def mod_null_space(matrix, p):
#     ns = null_space(matrix)
#     if ns.size == 0:
#         return np.zeros((matrix.shape[1],), dtype=int)
#     else:
#         vec = ns[:, 0]
#         # Find least common multiple of denominators to scale to integer
#         denominators = [frac.as_integer_ratio()[1] for frac in vec]
#         lcm_denominator = np.lcm.reduce(denominators)
#         vec = (vec * lcm_denominator).astype(int) % p
#         return vec

# a_perp = mod_null_space(A.T, p)
# b_perp = mod_null_space(B.T, p)

# # Master secret key
# msk = {
#     'A': A,
#     'a_perp': a_perp,
#     'B': B,
#     'b_perp': b_perp
# }
# '''

# st.code(keygen_code, language='python')

# k = max(n, m)

# # Generate random matrices A and B
# A = np.random.randint(0, p, (k, n))
# B = np.random.randint(0, p, (k, m))



# def generate_matrix_with_null_space(k, n, p):
#     """
#     Generate a matrix A with dimensions (k, n) over Z_p that guarantees a non-trivial null space.
#     """
#     # Generate a random (k-1, n) matrix
#     base_matrix = np.random.randint(0, p, (k-1, n))
    
#     # Generate a dependent row (sum of previous rows) to ensure a non-trivial null space
#     dependent_row = np.sum(base_matrix, axis=0) % p
    
#     # Append the dependent row to form the matrix
#     matrix = np.vstack([base_matrix, dependent_row])
#     return matrix

# # Example for A and B
# A = generate_matrix_with_null_space(k, n, p)
# B = generate_matrix_with_null_space(k, m, p)


# from sympy import Matrix

# def mod_null_space(matrix, p):
#     # Convert the matrix to the SymPy format, which handles exact arithmetic and modular arithmetic
#     sympy_matrix = Matrix(matrix)
    
#     # Compute the null space of the matrix modulo p
#     null_space_vectors = sympy_matrix.nullspace()  # This gives a basis for the null space

#     if len(null_space_vectors) == 0:
#         return np.zeros((matrix.shape[1],), dtype=int)  # Return zero vector if no null space exists
#     else:
#         # We take the first null space vector, and reduce all entries modulo p
#         null_vector = np.array(null_space_vectors[0].T) % p
#         return np.squeeze(null_vector)  # Return the null space vector modulo p


# # Now, compute a_perp and b_perp as before
# a_perp = mod_null_space(A.T, p)
# b_perp = mod_null_space(B.T, p)




# msk = {
#     'A': A,
#     'a_perp': a_perp,
#     'B': B,
#     'b_perp': b_perp
# }

# st.markdown("### Generated Matrices and Vectors")

# st.markdown("Matrix A:")
# st.write(A)

# st.latex(r"\text{Vector } a_{\perp}:")
# st.write(a_perp)

# st.markdown("Matrix B:")
# st.write(B)

# st.latex(r"\text{Vector } b_{\perp}:")
# st.write(b_perp)

# st.markdown("## 4. Secret Key Generation for Function F")

# st.markdown("**Explanation:**")

# st.markdown("For each \( i \in [n] \) and \( j \in [m] \), we generate random vectors and a random bit:")

# st.latex(r"r_i \leftarrow_r \mathbb{Z}_p^k")
# st.latex(r"s_j \leftarrow_r \mathbb{Z}_p^k")
# st.latex(r"\beta \leftarrow_r \{0,1\}")

# st.markdown("We compute the secret key \( sk_F \) as:")

# st.latex(r"sk_F = \beta + \sum_{i=1}^n \sum_{j=1}^m f_{ij} r_i^\top A^\top B s_j \mod p")


# skf_code = '''
# # Generate random vectors r_i and s_j
# ri = [np.random.randint(0, p, k) for _ in range(n)]  # Each r_i ∈ Z_p^k
# sj = [np.random.randint(0, p, k) for _ in range(m)]  # Each s_j ∈ Z_p^k

# # Generate random bit β ∈ {0,1}
# beta = np.random.randint(0, 2)

# # Function F
# F = np.random.randint(0, p, (n, m))

# # Compute sk_F
# skF = beta
# for i in range(n):
#     for j in range(m):
#         term = F[i, j] * ri[i].T @ A.T @ B @ sj[j]
#         skF += term
# skF = skF % p
# '''

# st.code(skf_code, language='python')

# ri = [np.random.randint(0, p, k) for _ in range(n)]  # Each r_i ∈ Z_p^k
# sj = [np.random.randint(0, p, k) for _ in range(m)]  # Each s_j ∈ Z_p^k

# # Generate random bit β ∈ {0,1}
# beta = np.random.randint(0, 2)

# # Function F
# F = np.random.randint(0, p, (n, m))

# # Compute sk_F
# skF = beta
# for i in range(n):
#     for j in range(m):
#         term = F[i, j] * ri[i].T @ A.T @ B @ sj[j]
#         skF += term
# skF = skF % p

# st.markdown("### Function Matrix F:")
# st.write(F)

# st.latex(r"\text{Random Bit } \beta:")
# st.write(beta)

# st.latex(r"\text{Secret Key } sk_F:")
# st.write(skF)

# st.markdown("## 5. Encryption")

# st.markdown("**Explanation:**")

# st.markdown("To encrypt vectors")
# st.latex(r"x \in \mathbb{Z}_p^n \text{ and } y \in \mathbb{Z}_p^m")
# st.markdown(", we compute:")

# st.latex(r"Ct(x, y) = \left( \left\{ A r_i + b_{\perp} x_i \right\}, \left\{ B s_j + a_{\perp} y_j \right\} \right)")


# encryption_code = '''
# # Message vectors x and y
# x = np.random.randint(0, p, n)
# y = np.random.randint(0, p, m)

# # Compute ciphertext components
# Ct1 = [ (A @ ri[i] + b_perp * x[i]) % p for i in range(n) ]
# Ct2 = [ (B @ sj[j] + a_perp * y[j]) % p for j in range(m) ]
# '''

# st.code(encryption_code, language='python')

# x = np.random.randint(0, p, n)
# y = np.random.randint(0, p, m)

# Ct1 = [ (A @ ri[i] + b_perp * x[i]) % p for i in range(n) ]
# Ct2 = [ (B @ sj[j] + a_perp * y[j]) % p for j in range(m) ]

# st.markdown("### Message Vectors x and y:")
# st.latex(r"\text{Vector } x:")
# st.write(x)
# st.latex(r"\text{Vector } y:")
# st.write(y)

# st.markdown("### Ciphertext Components:")
# st.latex(r"\text{Ciphertext } Ct1:")
# st.write(np.array(Ct1))
# st.latex(r"\text{Ciphertext } Ct2:")
# st.write(np.array(Ct2))

# st.markdown("## 6. Decryption")

# st.markdown("**Explanation:**")

# st.markdown("We compute the bilinear pairing between ciphertext components and use the secret key \( sk_F \) to recover \( x^\top F y \). We also incorporate the random bit \( \\beta \) in the decryption.")

# decryption_code = '''
# # Compute pairing
# pairing = 0
# for i in range(n):
#     for j in range(m):
#         e = Ct1[i].T @ Ct2[j]
#         pairing += e * F[i, j]
# pairing = pairing % p

# # Adjust pairing with skF
# result = (pairing - skF) % p

# # Expected result
# expected = (x.T @ F @ y - beta) % p
# '''

# st.code(decryption_code, language='python')

# # Execute the code
# pairing = 0
# for i in range(n):
#     for j in range(m):
#         e = Ct1[i].T @ Ct2[j]
#         pairing += e * F[i, j]
# pairing = pairing % p

# # Adjust pairing with skF
# result = (pairing - skF) % p

# # Expected result
# expected = (x.T @ F @ y - beta) % p

# st.markdown("### Decryption Result:")
# st.latex(r"\text{Computed Result:}")
# st.write(int(result))
# st.latex(r"\text{Expected Result } x^\top F y - \beta \mod p:")
# st.write(expected)

# if result == expected:
#     st.success("Decryption successful! The computed result matches the expected result.")
# else:
#     st.error("Decryption failed. The computed result does not match the expected result.")

# st.markdown("""
# **Note:** In this implementation, we included the random bit \( beta \) in both the secret key generation and the decryption process, as described in your paper. The computations are adjusted accordingly to ensure the decryption yields the correct result.
# """)


