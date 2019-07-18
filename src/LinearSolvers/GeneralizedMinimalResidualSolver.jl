module GeneralizedMinimalResidualSolver

export GeneralizedMinimalResidual

using ..LinearSolvers
using ..MPIStateArrays

using LinearAlgebra
using LazyArrays
using StaticArrays

const LS = LinearSolvers

"""
    GeneralizedMinimalResidual(M, Q, tolerance)

This is an object for solving linear systems using an iterative Krylov method.
The constructor parameter `M` is the number of steps after which the algorithm
is restarted (if it has not converged), `Q` is a reference state used only to
allocate the solver internal state, and `tolerance` specifies the convergence
threshold based on the residual norm. Since the amount of additional memory
required by the solver is  roughly `(M + 1) * size(Q)` in practical applications `M` 
should be kept small. This object is intended to be passed to the [`linearsolve!`](@ref)
command.

This uses the restarted Generalized Minimal Residual method of Saad and Schultz (1986).

### References
    @article{saad1986gmres,
      title={GMRES: A generalized minimal residual algorithm for solving nonsymmetric linear systems},
      author={Saad, Youcef and Schultz, Martin H},
      journal={SIAM Journal on scientific and statistical computing},
      volume={7},
      number={3},
      pages={856--869},
      year={1986},
      publisher={SIAM}
    }
"""
struct GeneralizedMinimalResidual{M, MP1, MMP1, T, AT} <: LS.AbstractIterativeLinearSolver
  krylov_basis::NTuple{MP1, AT}
  H::MArray{Tuple{MP1, M}, T, 2, MMP1}
  g0::MArray{Tuple{MP1, 1}, T, 2, MP1}
  tolerance::MArray{Tuple{1}, T, 1, 1}

  function GeneralizedMinimalResidual(M, Q::AT, tolerance) where AT
    krylov_basis = ntuple(i -> similar(Q), M + 1)
    H = @MArray zeros(M + 1, M)
    g0 = @MArray zeros(M + 1)

    new{M, M + 1, M * (M + 1), eltype(Q), AT}(krylov_basis, H, g0, (tolerance,))
  end
end

const weighted = true

function LS.initialize!(linearoperator!, Q, Qrhs, solver::GeneralizedMinimalResidual)
    g0 = solver.g0
    krylov_basis = solver.krylov_basis

    @assert size(Q) == size(krylov_basis[1])

    # store the initial residual in krylov_basis[1]
    linearoperator!(krylov_basis[1], Q)
    krylov_basis[1] .*= -1
    krylov_basis[1] .+= Qrhs

    residual_norm = norm(krylov_basis[1], weighted)
    fill!(g0, 0)
    g0[1] = residual_norm
    krylov_basis[1] ./= residual_norm

    threshold = solver.tolerance[1] * norm(Qrhs, weighted)
end

function LS.doiteration!(linearoperator!, Q, Qrhs,
                         solver::GeneralizedMinimalResidual{M}, threshold) where M
 
  krylov_basis = solver.krylov_basis
  H = solver.H
  g0 = solver.g0

  converged = false
  residual_norm = typemax(eltype(Q))
  
  Ω = LinearAlgebra.Rotation{eltype(Q)}([])
  j = 1
  for outer j = 1:M

    # Arnoldi using the Modified Gram Schmidt orthonormalization
    linearoperator!(krylov_basis[j + 1], krylov_basis[j])
    for i = 1:j
      H[i, j] = dot(krylov_basis[j + 1], krylov_basis[i], weighted)
      @. krylov_basis[j + 1] -= H[i, j] * krylov_basis[i]
    end
    H[j + 1, j] = norm(krylov_basis[j + 1], weighted)
    krylov_basis[j + 1] ./= H[j + 1, j]
   
    # apply the previous Givens rotations to the new column of H
    @views H[1:j, j:j] .= Ω * H[1:j, j:j]

    # compute a new Givens rotation to zero out H[j + 1, j]
    G, _ = givens(H, j, j + 1, j)

    # apply the new rotation to H and the rhs
    H .= G * H
    g0 .= G * g0

    # compose the new rotation with the others
    Ω = lmul!(G, Ω)

    residual_norm = abs(g0[j + 1])

    if residual_norm < threshold
      converged = true
      break
    end
  end

  # solve the triangular system
  y = @views UpperTriangular(H[1:j, 1:j]) \ g0[1:j]

  # compose the solution
  expr_Q = Q
  for i = 1:j
    expr_Q = @~ @. expr_Q + y[i] * krylov_basis[i]
  end
  Q .= expr_Q

  # if not converged restart
  converged || LS.initialize!(linearoperator!, Q, Qrhs, solver)
  
  (converged, j, residual_norm / threshold * solver.tolerance[1])
end

end
