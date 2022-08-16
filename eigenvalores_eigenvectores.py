import numpy as np


def metodoPotencia(A, q_0, m, tol) -> tuple:
    """
    Entrada:
        -   A  : una matriz cuadrada semisimple (que sus eigenvectores forman una base para C^(nxn) )
        -   q_0: un vector C^n donde inician las iteraciones
        -   m  : numero macimo de iteracioens
        -   tol: tolerancia de error
    Salida:
        - sigmai: aproximacion de eigenvalor dominante de la matriz
        - qi    : aproximacion del eigenvector dominante de la matriz
    """
    q_i = q_0/np.linalg.norm(q_0, np.inf)
    iteration = 0
    error = np.inf

    while iteration < m and tol > error:
        q_i_1 = A@q_i
        sigmai = np.linalg.norm(q_i_1, np.inf)
        q_i_1 = q_i_1/sigmai

        error = np.linalg.norm(q_i_1 - q_i, np.inf) / \
            np.linalg.norm(q_i_1, np.inf)
        iteration += 1
        q_i = q_i_1

    return sigmai, q_i


def metodoPotenciaInv(A, q_0, m, tol) -> tuple:
    """
    Entrada:
        -   A  : una matriz cuadrada regular (que sus eigenvectores forman una base para C^(nxn) )
        -   q_0: un vector C^n donde inician las iteraciones
        -   m  : numero macimo de iteracioens
        -   tol: tolerancia de error
    Salida:
        - sigmai: aproximacion de eigenvalor dominante de la matriz inversa de A
        - qi    : aproximacion del eigenvector dominante de la matriz inversa de A
    """
    q_i = q_0/np.linalg.norm(q_0, np.inf)
    iteration = 0
    error = np.inf

    while m > iteration and tol > error:
        q_i_1 = np.linalg.solve(A, q_i)
        sigmai = np.linalg.norm(q_i_1, np.inf)
        q_i_1 = q_i_1/sigmai

        error = np.linalg.norm(q_i_1 - q_i, np.inf) / \
            np.linalg.norm(q_i_1, np.inf)
        iteration += 1
        q_i = q_i_1

    return sigmai, q_i

