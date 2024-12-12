# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 09:39:05 2024

@author: StephanNef
"""

import streamlit as st


st.set_page_config(
     page_title="IMP Functional Encryption",
     page_icon="🔐",
     layout="wide",
)

st.logo("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/HSG_Logo_DE_RGB.svg/1024px-HSG_Logo_DE_RGB.svg.png",link="https://www.unisg.ch/de/")



st.markdown("""
# Fully Secure Inner Product Functional Encryption (IPFE)
""")
st.info("**Reference**: S. Agrawal, B. Libert, and D. Stehle, “Fully Secure Functional Encryption for Inner Products, from Standard Assumptions,” Cryptology ePrint Archive, 2015. https://eprint.iacr.org/2015/608 ")


with st.expander("📖 See IPFE-FULLYSEC Scheme"):
    st.image("figures/ipfe-fullsec.png")



st.markdown("""
This application provides a step-by-step explanation of **Fully Secure Functional Encryption (FE) for Inner Products**.  

We will follow the scheme according to the structure and explain each step both mathematically and corresponding implemented code:

- **Setup:** defines the master keys and public parameters based on sampled random exponents.
- **Keygen:** derives a functional decryption key for a chosen vector $x$.
- **Encrypt:** produces a ciphertext for a chosen vector $y$.
- **Decrypt:** given $SK_x$ and a ciphertext for $y$, recovers the value $g^{\langle x,y\\rangle}$

This allows one to compute the inner product $\langle x,y \\rangle$ without revealing $y$ entirely, demonstrating the core idea of Inner Product Functional Encryption.

""")

st.write("---")
st.subheader("1. Step: Setup Phase")

st.markdown("We start with a cyclic group:")
st.latex("G \\text{ is a cyclic group of prime order } q.")

st.markdown("We have generators:")
st.latex("g, h \\in G")

st.markdown("For each dimension:")
st.latex("i \\in \\{1,\\ldots,\\ell\\}")

st.markdown("We sample secret values:")
st.latex("s_i, t_i \\leftarrow \\mathbb{Z}_q")

st.markdown("""Therefore we define:""")

st.latex(r"h_i = g^{s_i} \cdot h^{t_i}, \quad \text{for } i = 1, \ldots, \ell.")

st.markdown("The master secret key (MSK) is:")

st.latex(r"\text{msk} := \{(s_i, t_i)\}_{i=1}^{\ell}")

st.markdown("The master public key (MPK) is:")

st.latex(r"\text{mpk} := (G, g, h, \{h_i\}_{i=1}^{\ell})")


with st.expander("Applied Code for Setup:", expanded=True):
    st.code("""\
def setup(self, l):
    # si, ti are secrets from Z_q
    si = [self.group.random() for _ in range(l)]
    ti = [self.group.random() for _ in range(l)]
    # Compute h_i = g^{s_i} * h^{t_i}
    hi = [(self.g ** si[i]) * (self.h ** ti[i]) for i in range(l)]
    # MPK contains public params
    # MSK contains secret s_i, t_i
    return MPK(self.g, self.h, hi), MSK(si, ti)
    """,language='python')

st.write("---")
st.subheader("2. Step: Key Generation (KeyGen)")

st.markdown("Given the MSK and a vector:")
st.latex("x = (x_1, \\ldots, x_\\ell) \\in \\mathbb{Z}_q^{\\ell}")

st.markdown("We compute the secret key $s_x, t_x$ for $x$ as:")
st.latex("s_x = \\sum_{i=1}^{\\ell} s_i x_i, \\quad t_x = \\sum_{i=1}^{\\ell} t_i x_i.")

st.markdown("The resulting secret key is:")
st.latex("\\text{sk}_x = (s_x, t_x).")

st.markdown("**In summary:**")
st.markdown("- Given $(s_i, t_i)$ from MSK and $x$")
st.markdown("- Compute $s_x, t_x$ as linear combinations.")

with st.expander("Applied Code for KeyGen:", expanded=True):
    st.code("""\
def keygen(self, msk, x, l):
    # Given vector x and msk = (s_i, t_i), 
    # compute s_x = sum(s_i * x_i) and t_x = sum(t_i * x_i)
    sx = sum(int(toInt(msk.si[i])) * x[i] for i in range(l))
    tx = sum(int(toInt(msk.ti[i])) * x[i] for i in range(l))
    return SKx(sx, tx)
    """, language='python')


st.write("---")
st.subheader("3. Step: Encryption Phase")


st.markdown("To encrypt a vector $y$:")
st.latex("y = (y_1, \\ldots, y_\\ell) \\in \\mathbb{Z}_q^{\\ell}")

st.markdown("**Step 3.1.** we sample a random $r$:")
st.latex("r \\leftarrow \\mathbb{Z}_q")

st.markdown("**Step 3.2.** we compute $C$:")
st.latex("C = g^r, \\quad D = h^r")

st.markdown("and for each $i$:")
st.latex("E_i = g^{y_i} \\cdot (h_i)^r")
st.markdown("$E_i$ represents an encrypted portion of the $i$th component of $y$")

st.markdown("The ciphertext is $Cy$:")
st.latex("\\text{Cy} = (C, D, \\{E_i\\}_{i=1}^{\\ell})")

st.markdown("**In summary:**")
st.markdown("- Choose random $r$")
st.markdown("- Compute $C, D$ and each $E_i$")

with st.expander("Applied Code for Encryption", expanded=True):
    st.code("""\
def encrypt(self, mpk, y, l):
    # Random r
    r = int(toInt(self.group.random()))
    # C = g^r, D = h^r
    C = mpk.g ** r
    D = mpk.h ** r
    # E_i = g^{y_i} * (h_i)^r
    Ei = [(mpk.g ** y[i]) * (mpk.hi[i] ** r) for i in range(l)]
    return Cy(C, D, Ei)
    """, language='python')

st.write("---")
st.subheader("4. Step: Decryption Phase")

st.markdown("Given:")
st.latex("\\text{sk}_x = (s_x, t_x), \\quad \\text{Cy} = (C, D, \\{E_i\\}), \\quad x = (x_1, \\ldots, x_\\ell)")

st.markdown("We first compute:")
st.latex("E_x = \\frac{\\prod_{i=1}^{\\ell}(E_i)^{x_i}}{C^{s_x} D^{t_x}}.")

st.markdown("Expanding the terms:")
st.latex("\\prod_{i=1}^{\\ell}(E_i)^{x_i} = \\prod_{i=1}^{\\ell}\\bigl(g^{y_i}(h_i)^r\\bigr)^{x_i} = g^{\\sum_i x_i y_i} \\cdot \\prod_{i=1}^{\\ell}(h_i)^{r x_i}.")

st.markdown("Meanwhile:")
st.latex("C^{s_x}D^{t_x} = (g^r)^{s_x}(h^r)^{t_x} = g^{r s_x}h^{r t_x}.")

st.markdown("Since:")
st.latex("h_i = g^{s_i}h^{t_i},")
st.markdown("...the terms involving $r$ will cancel out upon division.")
st.markdown("After cancellation, we get:")
st.latex("E_x = g^{\\sum_i x_i y_i}.")

st.markdown("To recover the inner product:")
st.latex("\\langle x,y \\rangle = \\sum_i x_i y_i")

st.markdown("...one would need to solve the discrete log base:")
st.latex("g")

st.markdown("The code simply computes the group element:")
st.latex("g^{\\langle x,y \\rangle}")


st.markdown("**In summary:**")
st.markdown("- Raise each $E_i$ to $x_i$ and multiply.")
st.markdown("- Divide by:")
st.latex("C^{s_x} D^{t_x}")
st.markdown("- Obtain")
st.latex("g^{\\langle x,y \\rangle}")

    
    
with st.expander("Applied Code for Decryption:", expanded=True):
    st.code("""\
def decrypt(self, mpk, sk, cy, x, l):
    # Compute numerator = Π_i (E_i)^{x_i}
    tmp = [int(toInt(cy.Ei[i] ** x[i])) for i in range(l)]
    res = 1
    for val in tmp:
        res *= val
    # Compute denominator = C^{s_x} * D^{t_x}
    divisor = (cy.C ** sk.sx) * (cy.D ** sk.tx)
    # Divide to get g^{<x,y>}
    res = res // divisor
    return res
    """, language='python')





