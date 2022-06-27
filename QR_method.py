def power_iteration(A,v, num_simulations = 10000):
  b1 = np.random.rand(A.shape[1])
  for _ in range(num_simulations):
    b2 = np.dot(np.linalg.inv(A-v*np.identity(A.shape[1])), b1)
    b2_norm = np.linalg.norm(b2)
    b1 = b2 / b2_norm
  return b1
def gram_schmidt(A):
  m = A.shape[1]
  Q = np.zeros(A.shape, dtype=np.double)
  temp_vector = np.zeros(m, dtype=np.double)
  Q[:, 0] = A[:, 0] / np.linalg.norm(A[:, 0], ord=2)
  for i in range(1, m):
    q = Q[:, :i]
    temp_vector = np.sum(np.sum(q * A[:, i, None], axis=0) * q, axis=1)
    Q[:, i] = A[:, i] - temp_vector
    Q[:, i] /= np.linalg.norm(Q[:, i], ord=2)
  return Q
def QR(A):
  Q = gram_schmidt(A)
  R = Q.T @ A
  return (Q, R)
def QR_eigenvalues(A, max_iterations=10000):
  n = A.shape[0]
  λ = np.zeros(n, dtype=np.double)
  max_error = 1e-6
  for k in range(max_iterations):
    if np.abs(A[-1, -2]) <= max_error:
        n -= 1
        λ[n] = A[-1, -1]
        A = A[:-1, :-1]
    μ1, μ2 = eigs_2x2(A)
    if n == 2:
        λ[0] = μ1
        λ[1] = μ2
        break
    p = np.array([μ1 - A[-1, -1], μ2 - A[-1, -1]]).argmin()
    α = μ1 if p == 0 else μ2
    I = np.eye(n)
    Q, R = QR(A - α * I)
    A = R @ Q + α * I
  return λ
def eigs_2x2(A):
  b = -(A[-1, -1] + A[-2, -2])
  c = A[-1, -1] * A[-2, -2] - A[-2, -1] * A[-1, -2]
  d = np.sqrt(b ** 2 - 4 * c)
  if b > 0:
    return (-2 * c / (b + d), -(b + d) / 2)
  else:
    return ((d - b) / 2, 2 * c / (d - b))


