"""
JohnsonSu(ξ,λ,γ,δ)

The *Johnson distribution* with location `ξ` and scale `λ` has probability density function

```math
f(x; \\xi,\\lambda,\\gamma,\\delta) = \\frac{\\delta}{\\sqrt(2*\\pi)} + \\frac{1}{sqrt(x^2+1)} * exp(-\\frac{1}{2}\\lambda+\\delta*log(x+sqrt(x^2 +1))^-2))
```

```
julia
```
"""
struct JohnsonSu{T<:Real} <: ContinuousUnivariateDistribution
    ξ::T
    λ::T
    γ::T
    δ::T
    JohnsonSu{T}(ξ::T, λ::T, γ::T, δ::T) where {T<:Real} = new{T}(ξ,λ,γ,δ)
end

function JohnsonSu(ξ::T, λ::T, γ::T, δ::T) where {T <: Real}
    @check_args(JohnsonSu, -Inf < ξ < Inf && λ > 0 && -Inf < γ < Inf && δ > 0)
    return JohnsonSu{T}(ξ,λ,γ,δ)
end
#using Distributions
#using Distributions: Normal
#### Outer constructors
#JohnsonSu{T<:Real}(ξ::T, λ::T, γ::T, δ::T) = JohnsonSu{T}(ξ, λ, γ, δ)
JohnsonSu(ξ::Real, λ::Real, γ::Real, δ::Real) = JohnsonSu(promote(ξ, λ, γ, δ)...)
JohnsonSu(ξ::Integer, λ::Integer, γ::Integer, δ::Integer) = JohnsonSu(Float64(ξ), Float64(λ), Float64(γ), Float64(δ))
JohnsonSu() = JohnsonSu(1.0, 1.0, 1.0, 1.0)

@distr_support JohnsonSu -Inf Inf

#### Conversions
function convert(::Type{JohnsonSu{T}}, ξ::Real, λ::Real, γ::Real, δ::Real) where T<:Real
 JohnsonSu(T(ξ), T(λ), T(γ), T(δ))
end

#convert{T <: Real, S <: Float64}(::Type{JohnsonSu{T}}, d::JohnsonSu{S}) = JohnsonSu( T(d.ξ), T(d.λ),  T(d.γ), T(d.δ))

#### Parameters
peakness(d::JohnsonSu) = d.ξ
shape(d::JohnsonSu) = d.λ
location(d::JohnsonSu) = d.γ
scale(d::JohnsonSu) = d.δ

params(d::JohnsonSu) = (d.ξ, d.λ, d.γ, d.δ)
@inline partype(d::JohnsonSu{T}) where {T<:Real} = T

#### Statistics
mean(d::JohnsonSu) = d.ξ-d.λ*exp.(d.δ^2/2)*sinh(d.γ/d.δ)
median(d::JohnsonSu) = d.ξ+d.λ*sinh(-(d.γ/d.δ))
var(d::JohnsonSu) = d.λ^2/2*(exp.(d.δ^-2)-1)*(exp.(d.δ^-2)*cosh(2d.γ/d.δ)+1)
mode(d::JohnsonSu) = d.ξ-d.λ*exp.(d.δ^2/2)*sinh(d.γ/d.δ) # copied from mean function.
skewness(d::JohnsonSu) = -(exp.(d.δ^-2)^(1/2)*(exp.(d.δ^-2)-1)^2*(exp.(d.δ^-2)*(exp.(d.δ^-2)+2)*sinh(3*(d.λ/d.δ))+3*sinh((d.λ/d.δ)))/sqrt(2)*((exp.(d.δ^-2)-1)*(exp.(d.δ^-2)*cosh(2*(d.λ/d.δ))+1))^3/2)
kurtosis(d::JohnsonSu) = (exp.(d.δ^-2)^2*((exp.(d.δ^-2)^4+2*exp.(d.δ^-2)^3+3*exp.(d.δ^-2)^2-3)*cosh(4*(d.λ/d.δ)+(4*(exp.(d.δ^-2)+2))*cosh(2*(d.λ/d.δ))))+3*(2*exp.(d.δ^-2)+1)) / (2(exp.(d.δ^-2)*cosh(2*(d.λ/d.δ))+1)^2)

#### Evaluation
@_delegate_statsfuns JohnsonSu ξ, λ, γ, δ

mgf(d::JohnsonSu, t::Real) = exp(t * d.ξ + d.λ^2/2 * t^2)
cf(d::JohnsonSu, t::Real) = exp(im * t * d.ξ - d.λ^2/2 * t^2)

#### Sampling
rand(d::JohnsonSu) = rand(GLOBAL_RNG, d)
rand(rng::AbstractRNG, d::JohnsonSu) = d.ξ + d.λ * randn(rng)

#### Fitting
struct JohnsonSuStats <: SufficientStats
    s::Float64    # (weighted) sum of x
    m::Float64    # (weighted) mean of x
    s2::Float64   # (weighted) sum of (x - μ)^2
    s3::Float64    # total sample weight
end

function suffstats(::Type{JohnsonSu}, x::AbstractArray{T}) where T<:Real
    n = length(x)

    # compute m
    s = x[1]
    for i = 2:n
        @inbounds s += x[i]
    end

    m = s / n

    # compute s2
    s2 = abs2(x[1] - m)
    for i = 2:n
        @inbounds s2 += abs2(x[i] - m)
    end

    # compute s3
    s3 = kurtosis(x)

    JohnsonSuStats(m, s, s2, s3)
end

# fit_mle based on sufficient statistics
fit_mle(::Type{JohnsonSu}, ss::JohnsonSuStats) = JohnsonSu(ss.m, ss.s, ss.s2, ss.s3)

@quantile_newton(JohnsonSu)

pdf(d::JohnsonSu, x::Real) = (d.δ/d.λ*sqrt.(2π)*1/(sqrt.(1+(x-d.ξ/d.λ).^2 +1)))exp.(-1/2*(d.γ+d.δ*asinh.(x-d.ξ/d.λ).^2))
#cdf(d::JohnsonSu, x::Real) = cdf(Normal(0,1),x)*(d.γ + d.δ*sinh.((x-d.ξ)/d.λ))

function fit_mle(::Type{JohnsonSu}, x::AbstractArray{T}; xi::Float64=NaN, lambda::Float64=NaN, gamma::Float64=NaN, delta::Float64=NaN) where T<:Real
    if isnan(xi)
        if isnan(lambda)
            fit_mle(JohnsonSu, suffstats(JohnsonSu, x))
        else
            g = NormalKnownSigma(lambda)
            fit_mle(g, suffstats(g, x))
        end
    else
        if isnan(lambda)
            g = NormalKnownMu(gamma)
            fit_mle(g, suffstats(g, x))
        else
            JohnsonSu(xi, lambda, gamma, delta)
        end
    end
end

function fit_mle(::Type{JohnsonSu}, x::AbstractArray{T}, w::AbstractArray{Float64}; xi::Float64=NaN, lambda::Float64=NaN, gamma::Float64=NaN, delta::Float64=NaN) where T<:Real
    if isnan(mu)
        if isnan(sigma)
            fit_mle(JohnsonSu, suffstats(JohnsonSu, x, w))
        else
            g = NormalKnownSigma(sigma)
            fit_mle(g, suffstats(g, x, w))
        end
    else
        if isnan(sigma)
            g = NormalKnownMu(mu)
            fit_mle(g, suffstats(g, x, w))
        else
            JohnsonSu(mu, sigma)
        end
    end
end
