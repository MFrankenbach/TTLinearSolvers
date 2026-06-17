import LinearSolvers: myVec, conjugate_gradient
using LinearAlgebra

@testset "CG @ myVec" begin
    function test_conjugate_gradient(; tol=1.e-6)
        for N in 10:30:200
            Amat = randn(N,N)
            Amat = Amat'*Amat / norm(Amat)^2 + 0.1*I # Make it symmetric positive definite
            sol = randn(N)
            b = myVec(Amat*sol)
    
            A = (v::myVec) -> myVec(Amat * v.data)  # Define A as a function
    
            x0 = myVec(zeros(N))  # Initial guess as myVec
            sol_test = conjugate_gradient(A, b, x0; tol=tol, maxiter=N, howverbose=0)
            @test norm(A(sol_test).data-b.data) / norm(b) < tol
        end
    end

    test_conjugate_gradient(;tol=1.e-10)
end