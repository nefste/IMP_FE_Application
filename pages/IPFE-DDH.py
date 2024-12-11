# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 09:39:05 2024

@author: StephanNef
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(
     page_title="IMP Functional Encryption",
     page_icon="🔐",
     layout="wide",
)

st.logo("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/HSG_Logo_DE_RGB.svg/1024px-HSG_Logo_DE_RGB.svg.png",link="https://www.unisg.ch/de/")



# Function to generate a list of large prime numbers
def get_large_primes():
    return [7, 11, 13, 1000003, 1000033, 1000037, 1000039, 1000081]

# Function to get potential generators (primitive roots) for the selected prime p
def get_generators(p):
    # For simplicity, we'll use small integers as potential generators
    # In practice, finding primitive roots requires more complex calculations
    return [2, 3, 5]

# Function to provide predefined vectors for selection
def get_vectors():
    return {
        "Vector A": np.array([1, 2, 3]),
        "Vector B": np.array([4, 5, 6]),
        "Vector C": np.array([7, 8, 9]),
        "Vector D": np.array([2, 3, 5]),
        "Vector E": np.array([6, 7, 8])
    }

# Simulated Group Generation (GroupGen) based on user-selected parameters
def group_gen(p, g):
    # Returns the user-selected prime p and generator g
    return p, g

# Setup function for the IPFE scheme
def setup(l, p, g):
    # Master secret key is a random vector (s1, ..., sl) from Z_p
    msk = np.random.randint(1, p, size=l)
    
    # Master public key (hi = g^si mod p for each si)
    mpk = np.array([pow(g, int(si), p) for si in msk])
    
    # Return both the public and secret keys
    return (mpk, msk)

# Encryption function for the vector x = (x1, ..., xl)
def encrypt(mpk, x, p, g):
    l = len(x)
    # Random value r from Z_p
    r = np.random.randint(1, p)
    
    # Ciphertext component ct0 = g^r mod p
    ct0 = pow(g, r, p)
    
    # Ciphertext components cti = h_i^r * g^xi mod p for each i
    ct = [(pow(int(mpk[i]), r, p) * pow(g, int(x[i]), p)) % p for i in range(l)]
    
    # Return the ciphertext Ct = (ct0, (ct1, ..., ctl)) and the random value r
    return (ct0, ct), r

# Key derivation function for a vector y = (y1, ..., yl)
def key_der(msk, y, p):
    # Compute sky = <msk, y> (dot product) mod (p - 1)
    sky = np.dot(msk, y) % (p - 1)  # Exponents are modulo (p - 1)
    return sky

# Decryption function to recover g^<x, y>
def decrypt(mpk, ct, sky, y, p, g):
    ct0, ct1_l = ct  # ct1_l is the list of cti's
    l = len(ct1_l)
    
    # Compute the product of ct[i]^y[i] mod p
    num = 1
    for i in range(l):
        num = (num * pow(ct1_l[i], int(y[i]), p)) % p  
    
    # Compute the denominator as ct0^sky mod p
    denom = pow(ct0, int(sky), p)
    
    # Compute g^<x, y> = num * denom^{-1} mod p
    denom_inv = pow(denom, -1, p)  # Modular inverse of denom mod p
    result = (num * denom_inv) % p
    return result

# Streamlit App for the IPFE Demo
st.title("Inner Product Functional Encryption (IPFE) under the DDH Assumption")
st.info("**Reference**: M. Abdalla, F. Bourse, A. De Caro, and D. Pointcheval, “Simple Functional Encryption Schemes for Inner Products,” 2015. https://eprint.iacr.org/2015/017.pdf")


st.markdown("""
This application demonstrates the Inner Product Functional Encryption (IPFE) scheme under the Decisional Diffie-Hellman (DDH) assumption. 
We will go through the steps of 
- **Setup**
- **Encryption**
- **Key Derivation**
- **Decryption**
- **Verification**

At each step, we will explain the reasoning behind the operations to help you understand the underlying principles of the IPFE scheme.

""")

with st.expander("📖 See IPFE-DDH Scheme"):
    st.image("figures/ipfe-ddh.png")


with st.expander("🤔 About the Decisional Diffie-Hellman (DDH) Assumption"):
    st.markdown("""
    **Decisional Diffie-Hellman (DDH) Assumption**

    The DDH assumption asserts that in a cyclic group \( G \) of prime order \( p \) with generator \( g \), it is computationally hard to distinguish between the following two tuples:
    """)
    st.latex(r"(g, g^a, g^b, g^{ab})")
    st.latex(r"(g, g^a, g^b, g^c)")


    st.markdown("""where \( a, b, c \) are random elements of: """)
    st.latex(r"\mathbb{Z}_p, \text{ and } c \text{ is independent of } a \text{ and } b.")
    
    st.markdown("""

    **Why DDH Asumption matters in IPFE**

    - **Security Foundation:** The DDH assumption ensures that, without the secret keys, one cannot derive any meaningful information from the ciphertexts.
    - **Encryption Integrity:** It underpins the hardness of distinguishing encrypted inner products from random elements, thus securing the IPFE scheme.

    """)



st.info("⬅️ Please set your own User Inputs in the Sidebar.")


st.write('---')

with st.sidebar:
    # User Inputs
    st.markdown("### ⚙️ User Inputs")
    
    # Select prime number p
    primes = get_large_primes()
    p = st.selectbox("Select a prime number (p):", primes, index=0)
    
    # Select generator g
    generators = get_generators(p)
    g = st.selectbox("Select a generator (g):", generators, index=0)
    
    # Select vectors x and y
    vectors = get_vectors()
    vector_names = list(vectors.keys())
    
    x_name = st.selectbox("Select vector x:", vector_names, index=0)
    x = vectors[x_name]
    
    y_name = st.selectbox("Select vector y:", vector_names, index=1)
    y = vectors[y_name]
    
    st.markdown(f"- **Selected prime (p):** {p}")
    st.markdown(f"- **Selected generator (g):** {g}")
    st.markdown(f"- **Selected vector x:** {x_name} = {x}")
    st.markdown(f"- **Selected vector y:** {y_name} = {y}")

    st.write('---')

# Step 1: Setup Phase
l = len(x)
st.markdown("""
### Step 1: Setup Phase
In this step, we generate the master public key (**mpk**) and the master secret key (**msk**).

**Why:** The setup phase initializes the keys necessary for encryption and decryption.

**How:** We generate a random master secret key vector `msk` and compute the corresponding master public key `mpk` using the selected generator `g` and prime `p`.
""")

mpk, msk = setup(l, p, g)

with st.expander("Applied Code for Setup", expanded=True):
    st.code("""
def setup(l, p, g):
    msk = np.random.randint(1, p, size=l)
    mpk = np.array([pow(g, int(si), p) for si in msk])
    return (mpk, msk)
    """, language='python')

st.markdown(f"- **Master Secret Key (msk):** {msk}")
st.markdown(f"- **Master Public Key (mpk):** {mpk}")

st.write('---')

# Step 2: Encrypt a Vector

# Encrypt the vector x
ct, r = encrypt(mpk, x, p, g)

st.markdown(f"""
### Step 2: Encrypt the Vector x
We will encrypt the message vector **x** = {x}. A random value **r** = {r} will be generated during encryption.

**Why:** Encryption transforms the message vector into a ciphertext that can be partially decrypted.

**How:** We use the public key `mpk` and generator `g` to compute the ciphertext components.
""")

with st.expander("Applied Code for Encryption", expanded=True):
    st.code("""
def encrypt(mpk, x, p, g):
    l = len(x)
    r = np.random.randint(1, p)
    ct0 = pow(g, r, p)
    ct = [(pow(int(mpk[i]), r, p) * pow(g, int(x[i]), p)) % p for i in range(l)]
    return (ct0, ct), r
    """, language='python')


st.markdown("**Ciphertext components:**")

# Replacing with st.latex
st.latex(f"ct0 = g^r \\mod p = {g}^{{{r}}} \\mod {p} = {ct[0]}")

ct_list_str = ', '.join([str(c) for c in ct[1]])
st.latex(f"ct = [ct_1, ct_2, \\dots, ct_l] = [{ct_list_str}]")

st.write('---')

# Continue with the rest of the code...

# Step 3: Derive the Secret Key
st.markdown("""
### Step 3: Derive the Secret Key for Vector y
We now derive the secret key for the target vector **y**.

**Why:** The secret key allows the holder to compute the inner product of x and y with the encrypted vector (without knowing x).

**How:** We compute the dot product of the master secret key `msk` and the vector `y`, modulo `p - 1`.
""")

# Derive the secret key for y
sky = key_der(msk, y, p)

with st.expander("Applied Code for Key Derivation", expanded=True):
    st.code("""
def key_der(msk, y, p):
    sky = np.dot(msk, y) % (p - 1)
    return sky
    """, language='python')

st.markdown("Derived secret key (sky) for vector y:")
st.latex(r"sky = " + str(sky))


st.write('---')

# Step 4: Decrypt and Compute the Inner Product
st.markdown("""### Step 4: Decrypt and Compute the Inner Product
We decrypt the ciphertext to recover the encoded inner product:
    """)

st.latex(r"g^{\langle x, y \rangle} \mod p")

st.markdown("""
**Why:** Decryption allows us to compute the inner product without revealing the original vectors.

**How:** We use the ciphertext, the secret key `sky`, and the vector `y` to compute the result.
""")

# Decrypt to recover g^<x, y>
inner_product_encoded = decrypt(mpk, ct, sky, y, p, g)

with st.expander("Applied Code for Decryption", expanded=True):
    st.code("""
def decrypt(mpk, ct, sky, y, p, g):
    ct0, ct1_l = ct
    l = len(ct1_l)
    num = 1
    for i in range(l):
        num = (num * pow(ct1_l[i], int(y[i]), p)) % p  
    denom = pow(ct0, int(sky), p)
    denom_inv = pow(denom, -1, p)
    result = (num * denom_inv) % p
    return result
    """, language='python')

st.subheader("**Decryption steps explained:**")

# Detailed decryption steps
st.markdown("**1. Compute the Numerator as Product of:**")
st.latex(r"ct_i^{y_i} \mod p")


num = 1
for i in range(l):
    temp = pow(ct[1][i], int(y[i]), p)
    st.latex(f"ct_{{{i+1}}}^{{y_{{{i+1}}}}} \\mod p = {ct[1][i]}^{{{y[i]}}} \\mod {p} = {temp}")
    num = (num * temp) % p

st.latex(f"Numerator = {num}")

with st.expander("Numerator Calculation", expanded=True):
    st.code("""num = 1
    for i in range(l):
        temp = pow(ct[1][i], int(y[i]), p)
        num = (num * temp) % p""")


st.markdown("**2. Compute the Denominator:**")
st.latex(r"ct0^{sky} \mod p")

denom = pow(ct[0], int(sky), p)
st.latex(f"ct0^{{sky}} \\mod p = {ct[0]}^{{{sky}}} \\mod {p}")
st.latex(f"Denominator = {denom}")

with st.expander("Denominator Calculation", expanded=True):
    st.code("""denom = pow(ct[0], int(sky), p)""")

st.markdown("**3. Compute the modular inverse of the denominator:**")
denom_inv = pow(denom, -1, p)
st.latex(f"(ct0^{{sky}})^{{-1}} \\mod p = {denom_inv}")
with st.expander("Modular Inverse of Denom. Calculation", expanded=True):
    st.code("""denom_inv = pow(denom, -1, p)""")

st.markdown("**4. Compute the final result:**")
st.latex(r"(\text{Numerator} \times \text{Denominator}^{-1}) \mod p")
st.latex(f"\\left( {num} \\times {denom_inv} \\right) \\mod {p} = {inner_product_encoded}")

st.write("**Recovered encoded inner product:**")
st.latex(r"g^{\langle x, y \rangle} \mod p ")
st.latex(f"={inner_product_encoded}")

st.write('---')


# Step 5: Verify the Result
st.markdown("""
### Step 5: Verify the Result
We check if the decrypted result matches the expected value calculated directly from vectors x and y.

**Why:** Verification ensures that the encryption and decryption processes are correct.

**How:** We compute the inner product of x and y, and then compute \( g \) to the power of that inner product modulo \( p \).
""")

# Step 1: Compute the inner product ⟨x, y⟩
inner_product = np.dot(x, y)
st.markdown("**Step 1: Compute the Inner Product:**")
inner_product_expanded = ' + '.join([f"{x[i]} \\times {y[i]}" for i in range(len(x))])
st.latex(r"\langle x, y \rangle = " + inner_product_expanded + f" = {inner_product}")

# Step 2: Compute ⟨x, y⟩ mod (p - 1)
inner_product_mod = inner_product % (p - 1)
st.markdown("**Step 2: Continue compute inner product (mod p-1):**")
st.latex(r"\langle x, y \rangle \mod (p - 1) = " + f"{inner_product} \mod {p - 1} = {inner_product_mod}")

# Step 3: Compute Expected Encoded Inner Product
expected_inner_product_encoded = pow(g, int(inner_product_mod), p)
st.markdown("**Step 3: Compute Expected Encoded Inner Product:**")
st.latex(r"g^{\langle x, y \rangle} \mod p = " + f"{g}^{{{inner_product_mod}}} \mod {p} = {expected_inner_product_encoded}")

# Step 4: Compare with Decrypted Result
st.markdown("**Step 4: Compare with Decrypted Result:**")
if inner_product_encoded == expected_inner_product_encoded:
    st.latex(r"\text{Decrypted Result} = " + str(inner_product_encoded) + r" = \text{Expected Result}")
    st.success("✅ Success! The encoded inner product matches the expected encoded inner product.")
else:
    st.latex(r"\text{Decrypted Result} = " + str(inner_product_encoded) + r" \ne \text{Expected Result} = " + str(expected_inner_product_encoded))
    st.error(f"❌ Error! The encoded inner product ({inner_product_encoded}) does NOT match the expected encoded inner product ({expected_inner_product_encoded}).")

with st.sidebar:
    st.header("🎯 Solution")
    if inner_product_encoded == expected_inner_product_encoded:
        st.write(f"**Decrypted Result = {inner_product_encoded} = Expected Result**")
        st.success("✅ Success! The encoded inner product matches the expected encoded inner product.")
    else:
        st.latex(r"\text{Decrypted Result} = " + str(inner_product_encoded) + r" \ne \text{Expected Result} = " + str(expected_inner_product_encoded))
        st.error(f"❌ Error! The encoded inner product ({inner_product_encoded}) does NOT match the expected encoded inner product ({expected_inner_product_encoded}).")

