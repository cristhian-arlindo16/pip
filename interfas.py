import streamlit as st
import numpy as np

def sustitucion(A, b):
    n = len(A)
    x = np.zeros(n)
    
    # Verificar si la matriz es triangular superior
    for i in range(n):
        if A[i][i] == 0:
            st.error("El sistema no puede resolverse por sustitución directamente")
            return None
    
    # Sustitución hacia atrás
    x[n-1] = b[n-1] / A[n-1][n-1]
    for i in range(n-2, -1, -1):
        suma = 0
        for j in range(i+1, n):
            suma += A[i][j] * x[j]
        x[i] = (b[i] - suma) / A[i][i]
    
    return x

def gauss_jordan(A, b):
    n = len(A)
    Ab = np.concatenate((A, b.reshape(n,1)), axis=1)
    
    for i in range(n):
        pivot = Ab[i][i]
        if pivot == 0:
            st.error("El sistema no puede resolverse por Gauss-Jordan")
            return None
            
        Ab[i] = Ab[i] / pivot
        
        for j in range(n):
            if i != j:
                Ab[j] = Ab[j] - Ab[i] * Ab[j][i]
    
    return Ab[:, -1]

def cramer(A, b):
    n = len(A)
    det_A = np.linalg.det(A)
    
    if det_A == 0:
        st.error("El sistema no puede resolverse por Cramer (determinante = 0)")
        return None
        
    x = np.zeros(n)
    
    for i in range(n):
        Ai = A.copy()
        Ai[:, i] = b
        x[i] = np.linalg.det(Ai) / det_A
        
    return x

def main():
    st.title("Sistema de Ecuaciones - Método SEL")
    st.write("Implementación de Sustitución, Gauss-Jordan y Cramer")
    
    n = st.number_input("Tamaño del sistema (nxn)", min_value=2, max_value=5, value=3)
    
    st.write("Ingrese los coeficientes de la matriz A:")
    A = np.zeros((n,n))
    for i in range(n):
        cols = st.columns(n)
        for j in range(n):
            A[i,j] = cols[j].number_input(f"A[{i+1},{j+1}]", value=0.0)
    
    st.write("Ingrese los términos independientes (vector b):")
    b = np.zeros(n)
    cols = st.columns(n)
    for i in range(n):
        b[i] = cols[i].number_input(f"b[{i+1}]", value=0.0)
    
    metodo = st.radio(
        "Seleccione el método de solución:",
        ["Sustitución", "Gauss-Jordan", "Cramer"]
    )
    
    if st.button("Resolver"):
        if metodo == "Sustitución":
            x = sustitucion(A, b)
        elif metodo == "Gauss-Jordan":
            x = gauss_jordan(A, b)
        else:
            x = cramer(A, b)
            
        if x is not None:
            st.write("Solución encontrada:")
            for i in range(n):
                st.write(f"x{i+1} = {x[i]:.4f}")

if __name__ == "__main__":
    main()
