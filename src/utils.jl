ldiv(J::AbstractArray, p) = J \ p

function ldiv(J::AnyCuMatrix, p::AnyCuVecOrMat)
    m, n = size(J)

    if m == n
        return J \ p
    end

    Q, R = qr(J)
    q = reshape(p, :, 1)
    return UpperTriangular(R) \ view(Q' * q, 1:n)
end
