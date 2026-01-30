"""
    minkowski_interval_abundance_2d_inclusive_hypergeom(
        m::Int, n::Int
    )

Exact 2D Minkowski causal-diamond interval abundance.
Inclusive interval size m (endpoints counted).

- m = 1 : returns n
- m ≥ 2 : exact 2F2 expression
"""
function minkowski_interval_abundance_2d_inclusive_hypergeom(
    m::Int,
    n::Int
)
    @assert m ≥ 1
    @assert n ≥ 1

    if m == 1
        return float(n)
    end

    mp = m - 2

    return n^(mp + 2) / (factorial(mp) * (1 + mp)^2 + (2 + mp)^2) *
           pFq((mp + 1.0, mp + 1.0),
               (mp + 3.0, mp + 3.0),
               -n)
end

"""
    minkowski_interval_abundance_2d_asymptotic(m::Int, n::Real)

Asymptotic approximation (large n) of the 2D Minkowski causal-diamond
interval abundance for *inclusive* interval size m.

Implements:
n * (-2 + γ - H_m + log n)
+ (1 + m) * (2 + log n - ψ(2 + m))
"""
function minkowski_interval_abundance_2d_inclusive_asymptotic(
    m::Int,
    n::Int,
)::Float64
    @assert m ≥ 1
    @assert n > 0

    if m == 1
        return float(n)
    end

    mp = m - 2

    return n * (-2 +
                MathConstants.eulergamma -
                harmonic(mp) +
                log(n)) +
           (1 + mp) * (2 +
                      log(n) -
                      digamma(2 + mp))
end

function minkowski_interval_abundance_2d_inclusive_asymptotic(
    m::Int,
    n::Int,
)::Float64
    @assert m ≥ 1
    @assert n > 0

    if m == 1
        return float(n)
    end

    mp = m - 2

    return 1 + 2 * mp -2n + (1 + mp + n) * (log(n) - polygamma(0, 1 + mp))
end


"""
    load_histograms_from_paths(
        paths::Vector{<:AbstractString},
        histname::Symbol;
        filters::Union{Nothing,Vector{Union{Nothing,Function}}}=nothing,
    )::Vector{Vector{Dict}}

Load a single histogram field `histname` from multiple `statistics.jld2` files.

Returns a vector `out` such that:
- `out[i]` is a `Vector{Dict}` containing the histograms from `paths[i]`
- only the requested histogram field is loaded (RAM-safe)
- optional per-file filters can be applied

`filters[i] === nothing` means no filtering for that file.
"""
function load_histograms_from_paths(
    paths::Vector{<:AbstractString},
    histname::Symbol;
    filters::Union{Nothing,Vector{Union{Nothing,Function}}}=nothing,
)::Vector{Vector{Dict}}
    n = length(paths)
    filters === nothing && (filters = fill(nothing, n))
    @assert length(filters) == n

    out = Vector{Vector{Dict}}(undef, n)

    Threads.@threads for i in 1:n
        path   = paths[i]
        filter = filters[i]

        hists = Dict[]

        JLD2.jldopen(path, "r") do f
            nbatches = f["meta/nbatches"]

            for b in 1:nbatches
                batch = f["batches/$b"]

                for x in batch
                    filter !== nothing && !filter(x) && continue
                    push!(hists, getfield(x, histname))
                end
            end
        end
        
        println("  Loaded $(length(hists)) histograms from $path")

        out[i] = hists
    end

    return out
end


"""
    densify_hists(hists::Vector{<:AbstractDict})

Convert sparse histogram dictionaries to a dense matrix with consistent binning.
Returns a matrix of size (Nsamples, nbins).
"""
function densify_hists(hists::Vector{<:AbstractDict})
    min_k = minimum(minimum(keys(h)) for h in hists)
    max_k = maximum(maximum(keys(h)) for h in hists)
    shift = (min_k == 0)

    nbins = shift ? max_k + 1 : max_k
    dense = Matrix{Float64}(undef, length(hists), nbins)

    for (i, h) in enumerate(hists)
        fill!(view(dense, i, :), 0.0)
        for (k, v) in h
            idx = shift ? k + 1 : k
            dense[i, idx] = v
        end
    end

    return dense
end

"""
    average_histogram_with_std(hists::AbstractVector{<:AbstractDict})

Given a vector of sparse histograms `hists` (each mapping `k::Int -> count::Int`),
compute the mean and standard deviation per bin.

Conventions:
- If any histogram contains bin `k = 0`, bins are treated as 0-based and shifted.
- Otherwise, bins are treated as already 1-based and are not shifted.
- Missing bins are treated as zero.

Returns:
- mean :: Vector{Float64}
- std  :: Vector{Float64}
"""
function average_histogram_with_std(
    hists::AbstractVector{<:AbstractDict},
)::Tuple{Vector{Float64},Vector{Float64}}
    isempty(hists) && return Float64[], Float64[]
    X = densify_hists(hists)
    # X: (Nsamples, nbins)
    mean_vec = vec(Statistics.mean(X; dims=1))
    # Population std (numerically safe)
    std_vec = sqrt.(max.(vec(Statistics.mean(X.^2; dims=1)) .- mean_vec.^2, 0.0))
    return mean_vec, std_vec
end

function replace_zeros(σ::AbstractVector{<:Real}; ϵ::Real=1e-3)
    nz = σ[σ .> 0]
    isempty(nz) && return σ
    epsσ = minimum(nz) * ϵ
    σ_new = copy(σ)
    @inbounds for i in eachindex(σ)
        if σ[i] == 0
            σ_new[i] = epsσ
        end
    end
    return σ_new
end

function abundance_shift(hist::Dict{Int,Int};)
    out = Dict{Int,Int}()
    for k in keys(hist)
        if k > 1
            out[k-2] = hist[k]
        end
    end
    return out
end

"""
    fit_histogram_bins(
        y_values::Vector{Float64},
        f::Function,
        param_syms::Tuple{Vararg{Symbol}},
        bin_lo::Int,
        bin_hi::Int;
        stds::Union{Nothing,Vector{Float64}} = nothing,    
        x_values::Union{Nothing,Vector{Float64}} = nothing,
        init = nothing,
        bounds = nothing,
        chisq::Bool = false,
        multistart::Int = 1,
    )

Least-squares fit of a model `f(x, params)` to histogram bins [bin_lo, bin_hi]
from `(y_values, stds)` as returned by `average_histogram_with_std`.

`param_syms` defines the parameter order in the returned `NamedTuple`.
`init` may be a vector or a `NamedTuple`. `bounds` may be a tuple of
`(lower, upper)` vectors or `(lower_nt, upper_nt)` NamedTuples.
"""
function fit_histogram_bins(
    y_values::Vector{Float64},
    f::Function,
    param_syms::Tuple{Vararg{Symbol}},
    bin_lo::Int,
    bin_hi::Int;
    stds::Union{Nothing,Vector{Float64}} = nothing,
    minimize_χ²::Bool = false,
    x_values::Union{Nothing,Vector{Float64}} = nothing,
    init::Union{Nothing,NamedTuple} = nothing,
    bounds::Union{Nothing,Tuple{NamedTuple,NamedTuple}} = nothing,
    goodness_of_fit::Bool = false,
    ϵ::Real = 1e-3,
    multistart::Int = 1,
    rng::Union{Nothing,AbstractRNG} = nothing,
    optim_options = nothing,
    method = Optim.NelderMead(),
    autodiff = nothing,
    verbose::Bool = false,
    std_fn::Union{Nothing,Function} = nothing,
    verbose_step::Union{Nothing,Int} = nothing,
)
    @assert 1 <= bin_lo <= bin_hi <= length(y_values) "bin range out of bounds"
    
    if std_fn !== nothing && stds === nothing
        error("std_fn requires stds to be provided")
    end

    if !isnothing(stds)
        @assert length(stds) == length(y_values) "stds length must match y_values length"
        stds = replace_zeros(stds; ϵ = ϵ)
        bad = (.!isfinite.(stds)) .| (stds .<= 0)
        if any(bad)
            nz = stds[isfinite.(stds) .& (stds .> 0)]
            fillval = isempty(nz) ? ϵ : minimum(nz) * ϵ
            @warn "stds contain non-finite or non-positive values; adjusting to eps*min nonzero std to avoid infinite chi-squared." eps = ϵ
            stds = copy(stds)
            stds[bad] .= fillval
        end
    else
        @assert !minimize_χ² "minimize_χ² requires stds to be provided"
    end

    p = length(param_syms)
    @assert multistart ≥ 1 "multistart must be ≥ 1"

    to_vec(nt::NamedTuple) = [getfield(nt, s) for s in param_syms]
    to_nt(v::AbstractVector) = NamedTuple{param_syms}(Tuple(v))

    init_vec = if init === nothing
        ones(p)
    else
        to_vec(init)
    end

    bounds_vec = nothing
    if bounds !== nothing
        if bounds isa Tuple && length(bounds) == 2
            lower, upper = bounds
            lower_vec = to_vec(lower)
            upper_vec = to_vec(upper)
            @assert length(lower_vec) == p && length(upper_vec) == p "bounds length must match param_syms"
            bounds_vec = (lower_vec, upper_vec)
        else
            error("bounds must be a tuple (lower, upper)")
        end
    end

    if x_values !== nothing
        @assert length(x_values) >= bin_hi "x_values length must be larger than bin_hi"
        xs = x_values[bin_lo:bin_hi]
    else
        xs = collect(bin_lo:bin_hi)
    end
    ys = y_values[bin_lo:bin_hi]

    function obj(x)
        v = bounds_vec === nothing ? x : clamp.(x, bounds_vec[1], bounds_vec[2])
        params = to_nt(v)
        preds = f.(xs, Ref(params))
        if minimize_χ²
            σ = std_fn === nothing ? stds[bin_lo:bin_hi] : std_fn(ys, preds, stds[bin_lo:bin_hi], params)
            r = (ys .- preds) ./ σ
        else
            r = ys .- preds
        end
        return sum(r .^ 2)
    end

    function solve(x0)
        if isnothing(optim_options)
            result = autodiff === nothing ?
                Optim.optimize(obj, x0, method) :
                Optim.optimize(obj, x0, method; autodiff = autodiff)
        else
            result = autodiff === nothing ?
                Optim.optimize(obj, x0, method, optim_options) :
                Optim.optimize(obj, x0, method, optim_options; autodiff = autodiff)
        end
        xopt = Optim.minimizer(result)
        fmin = Optim.minimum(result)
        return xopt, fmin
    end

    function score_for(xopt)
        x = bounds_vec === nothing ? xopt : clamp.(xopt, bounds_vec[1], bounds_vec[2])
        params = to_nt(x)
        preds = f.(xs, Ref(params))
        residuals = ys .- preds
        if !isnothing(stds)
            σ = std_fn === nothing ? stds[bin_lo:bin_hi] : std_fn(ys, preds, stds[bin_lo:bin_hi], params)
            dof = length(ys) - p
            return dof > 0 ? sum((residuals ./ σ) .^ 2) / dof : NaN
        else
            denom = similar(ys)
            @inbounds for i in eachindex(ys)
                denom[i] = ys[i] == 0 ? eps() : ys[i]
            end
            rel = residuals ./ denom
            return sqrt(Statistics.mean(rel .^ 2))
        end
    end

    best_x, best_f = solve(init_vec)
    best_score = score_for(best_x)
    label = isnothing(stds) ? "rel_rms" : "χ²"
    step = if verbose
        verbose_step === nothing ? max(1, round(Int, multistart * 0.1)) : max(1, verbose_step)
    else
        0
    end
    if verbose
        println("multistart 1: ", label, " = ", best_score, " (best = ", best_score, ")")
        flush(stdout)
    end

    if multistart > 1
        rng = rng === nothing ? Random.GLOBAL_RNG : rng
        scale = bounds_vec === nothing ? [init === 0 ? 1. : abs(init) for init in init_vec] : bounds_vec[2] .- bounds_vec[1]
        for i in 2:multistart
            x0 = if bounds_vec === nothing
                init_vec .+ (2 .* rand(rng, p) .- 1) .* scale
            else
                lower, upper = bounds_vec
                x = similar(init_vec)
                for j in 1:p
                    lo = lower[j]
                    hi = upper[j]
                    if isfinite(lo) && isfinite(hi)
                        x[j] = lo + rand(rng) * (hi - lo)
                    else
                        s = init_vec[j] == 0 ? 1.0 : abs(init_vec[j])
                        x[j] = init_vec[j] + (2 * rand(rng) - 1) * s
                        if isfinite(lo)
                            x[j] = max(x[j], lo)
                        end
                        if isfinite(hi)
                            x[j] = min(x[j], hi)
                        end
                    end
                end
                x
            end
            xopt, fmin = solve(x0)
            score = score_for(xopt)
            if fmin < best_f
                best_x, best_f = xopt, fmin
                best_score = score
            end
            if verbose && (i % step == 0 || i == multistart)
                println("multistart ", i, ": ", label, " = ", score, " (best = ", best_score, ")")
                flush(stdout)
            end
        end
    end

    if verbose
        println("multistart done: best ", label, " = ", best_score)
        flush(stdout)
    end

    xopt = bounds_vec === nothing ? best_x : clamp.(best_x, bounds_vec[1], bounds_vec[2])
    params = to_nt(xopt)

    if goodness_of_fit
        preds = f.(xs, Ref(params))
        residuals = ys .- preds
        if !isnothing(stds)
            σ = std_fn === nothing ? stds[bin_lo:bin_hi] : std_fn(ys, preds, stds[bin_lo:bin_hi], params)
            dof = length(ys) - p
            if dof <= 0
                @warn "chi-squared undefined: degrees of freedom <= 0" dof = dof
                χ² = NaN
            else
                χ² = sum((residuals ./ σ) .^ 2) / dof
            end
            return (params = params, rel_residuals = residuals ./ ys, χ² = χ²)
        end
        return (params = params, rel_residuals = residuals ./ ys)
    end

    return params
end

"""
    logticks(lo::Real, hi::Real; base::Real=10)

Return logarithmic major ticks and labels for the interval [lo, hi].

Rules:
- lo > 0 is required.
- Minor ticks are always dense: {1,2,…,9} × base^k within [lo, hi].
- Major ticks are chosen automatically to yield 3–6 ticks, with preference order:
  5 → 4 → 6 → 3.
- Allowed major tick patterns:
  1. decades: base^k
  2. sparse: {1,5} × base^k
  3. dense:  {1,…,9} × base^k
- Labels use powers (e.g. 10^3) for decades, numeric strings otherwise.

Returns `(major_ticks, major_labels)`; minor ticks are set separately via the theme.
"""
function _logticks_internal(lo::Real, hi::Real; base::Real = 10.0)
    lo <= 0 && throw(ArgumentError("logticks requires lo > 0"))

    logb(x) = log(x) / log(base)

    pmin = floor(Int, logb(lo))
    pmax = ceil(Int,  logb(hi))

    # helper for LaTeX labels
    latex_label(v) = begin
        if v < 1e-3 || v ≥ 1e4
            k = round(Int, logb(v))
            LaTeXString("\$10^{$k}\$")
        elseif v ≥ 1
            # integers like 1, 10, 100, 1000 without scientific notation
            LaTeXString("\$$(Int(round(v)))\$")
        else
            # numbers in (1e-3, 1): fixed-point, no exponent
            LaTeXString(@sprintf("\$%.3f\$", v))
        end
    end

    # candidate generators for major ticks
    candidates = [
        (:decades, p -> [base^Float64(p)]),
        (:sparse,  p -> [1*base^Float64(p), 5*base^Float64(p)]),
        (:dense,   p -> [m*base^Float64(p) for m in 1:9]),
    ]

    preferred_counts = (5, 4, 6, 3)

    best_ticks = Float64[]
    best_labels = String[]
    best_kind = :decades

    for target in preferred_counts
        for (kind, gen) in candidates
            ticks = Float64[]
            for p in pmin:pmax
                for v in gen(p)
                    lo ≤ v ≤ hi && push!(ticks, v)
                end
            end
            sort!(ticks)

            if length(ticks) == target
                labels = latex_label.(ticks)
                return ticks, labels, kind
            end

            if isempty(best_ticks) && 3 ≤ length(ticks) ≤ 6
                best_ticks = ticks
                best_labels = latex_label.(ticks)
                best_kind = kind
            end
        end
    end

    if !isempty(best_ticks)
        return best_ticks, best_labels, best_kind
    end

    # fallback: decades subset
    ticks = [base^Float64(p) for p in pmin:pmax if lo ≤ base^Float64(p) ≤ hi]
    labels = latex_label.(ticks)

    return ticks, labels, :decades
end

function logticks(lo::Real, hi::Real; base::Real = 10.0)
    ticks, labels, _ = _logticks_internal(lo, hi; base = base)
    return ticks, labels
end

function logminorticks(lo::Real, hi::Real; base::Real = 10.0)
    ticks, _, kind = _logticks_internal(lo, hi; base = base)

    logb(x) = log(x) / log(base)
    pmin = floor(Int, logb(lo))
    pmax = ceil(Int,  logb(hi))

    if kind == :dense
        return Float64[]
    end

    mults = kind == :decades ? (2:9) : [2, 3, 4, 6, 7, 8, 9]
    minor = Float64[]
    for p in pmin:pmax
        for m in mults
            v = m * base^Float64(p)
            lo ≤ v ≤ hi && push!(minor, v)
        end
    end

    sort!(minor)
    return minor
end

function Makie.get_minor_tickvalues(
    ::typeof(logminorticks),
    ::Union{typeof(log), typeof(log10), typeof(log2)},
    ::Any,
    lo::Real,
    hi::Real,
)
    return logminorticks(lo, hi)
end

function apply_paper_theme!(;
    double_column::Bool = false,
    magnification::Real = 1.0,
    logscale_x::Bool = false,
    logscale_y::Bool = false,
    legendpos = :rt,
    legendpadding = nothing,
    legendmargin = nothing,
    n_Legend_columns::Int = 1,
)
    # physical sizes (in cm → inches)
    cm = 1 / 2.54
    dpi = 96
    pt = 4/3
    width_cm = double_column ? 17.8 : 8.6
    height_cm = 0.75 * width_cm

    s(x) = x * magnification

    figsize = (s(width_cm * cm * dpi), s(height_cm * cm * dpi))

    # base typography (independent of figsize)
    base_fontsize = 11pt
    labelsize     = 11pt
    ticklabelsize = 11pt

    # line and tick sizes
    linewidth     = 1.5   
    ticksize      = 6
    minorticksize = 5

    axis_kwargs = (
        xlabelsize = s(labelsize),
        ylabelsize = s(labelsize),
        titlesize  = s(labelsize),

        xticklabelsize = s(ticklabelsize),
        yticklabelsize = s(ticklabelsize),

        xgridvisible = false,
        ygridvisible = false,
        gridcolor = (:black, 0.15),
        gridwidth = s(0.5),

        spinewidth = s(2.0),
        spinecolor = :black,
        leftspinevisible   = true,
        rightspinevisible  = true,
        topspinevisible    = true,
        bottomspinevisible = true,

        xtickwidth  = s(1.4),
        ytickwidth  = s(1.4),
        xticksize   = s(ticksize),
        yticksize   = s(ticksize),
        xminortickwidth = s(0.8),
        yminortickwidth = s(0.8),
        xminorticksize = s(minorticksize),
        yminorticksize = s(minorticksize),

        xticksmirrored = true,
        yticksmirrored = true,
        xtickalign  = 1.0,
        ytickalign  = 1.0,
        xminortickalign = 1.,
        yminortickalign = 1.,

        xminorticksvisible = true,
        yminorticksvisible = true,
    )

    if logscale_x
        axis_kwargs = merge(axis_kwargs, (
            xticks = logticks,
            xminorticks = logminorticks,
        ))
    end

    if logscale_y
        axis_kwargs = merge(axis_kwargs, (
            yticks = logticks,
            yminorticks = logminorticks,
        ))
    end

    set_theme!(
        Theme(
            figure_padding = s(10),

            fonts = (
                regular      = "CMU Serif",
                italic       = "CMU Serif Italic",
                bold         = "CMU Serif Bold",
                bold_italic  = "CMU Serif Bold Italic",
            ),
            fontsize = s(base_fontsize),

            Axis = axis_kwargs,

            Legend = (
                framevisible = true,
                framewidth = s(2.0),
                framecolor = :black,
                padding = legendpadding === nothing ? (s(6), s(6), s(6), s(6)) : legendpadding,
                margin = legendmargin === nothing ? (s(11), s(11), s(11), s(11)) : legendmargin,
                labelsize = s(labelsize),
                position = legendpos,
                nbanks = n_Legend_columns,
            ),

            palette = (
                color = [
                    colorant"#F1C21B",  # IBM Yellow
                    colorant"#D12771",  # IBM Magenta
                    colorant"#009D9A",  # IBM Teal
                    colorant"#FA4D56",  # IBM Red
                    colorant"#6F6F6F",  # IBM Gray
                    colorant"#0F62FE",  # IBM Blue
                    colorant"#24A148",  # IBM Green
                ],
            ),

            Lines = (
                linewidth = s(linewidth),
            ),

            Band = (
                color = (:auto, 0.3),
            ),
        )
    )
    return figsize
end

"""
    plot_mean_histograms_with_std(
        data::Vector{Tuple{Vector{Float64},Vector{Float64}}};
        xlim::Union{Tuple{Float64,Float64},Nothing} = nothing,
        ylim::Union{Tuple{Float64,Float64},Nothing} = nothing,
        logscale_x::Bool = false,
        logscale_y::Bool = false,
        normalize::Bool = false,
        plotlabel::Union{AbstractString,Nothing} = nothing,
        xlabel::Union{AbstractString,Nothing} = nothing,
        ylabel::Union{AbstractString,Nothing} = nothing,
        hist_labels::Union{Nothing,Vector{<:AbstractString}} = nothing,
        double_column::Bool = false,
        magnification::Real = 1.0,
        legendpos = :rt,
        legendpadding = nothing,
        legendmargin = nothing,
        n_Legend_columns::Int = 1,
        linewidth::Union{Nothing,Real} = nothing,
        plot_types::Union{Nothing,Vector{Symbol}} = nothing,
        markersize::Union{Nothing,Real} = nothing,
        return_axis::Bool = false,
    )::Figure

Plot mean histograms with ±1σ bands.

Each element of `data` must be `(mean, std)`, where both are vectors
defined on the same binning.
If `normalize` is true, each histogram is scaled so its maximum is 1.
"""
function plot_mean_histograms_with_std(
    data::Vector{Tuple{Vector{Float64},Vector{Float64}}};
    xlim::Union{Tuple{Float64,Float64},Nothing} = nothing,
    ylim::Union{Tuple{Float64,Float64},Nothing} = nothing,
    logscale_x::Bool = false,
    logscale_y::Bool = false,
    normalize::Bool = false,
    plotlabel::Union{AbstractString,Nothing} = nothing,
    xlabel::Union{AbstractString,Nothing} = nothing,
    ylabel::Union{AbstractString,Nothing} = nothing,
    hist_labels::Union{Nothing,Vector{<:AbstractString}} = nothing,
    double_column::Bool = false,
    magnification::Real = 1.0,
    legendpos = :rt,
    legendpadding = nothing,
    legendmargin = nothing,
    n_Legend_columns::Int = 1,
    linewidth::Union{Nothing,Real} = nothing,
        markersize::Union{Nothing,Real} = nothing,
        plot_types::Union{Nothing,Vector{Symbol}} = nothing,
        return_axis::Bool = false,
)::Union{Figure, Tuple{Figure, Axis}}

    if hist_labels !== nothing
        @assert length(hist_labels) == length(data) "hist_labels and data must have same length"
    end
    if plot_types !== nothing
        @assert length(plot_types) == length(data) "plot_types and data must have same length"
    end

    figsize = apply_paper_theme!(
        double_column = double_column,
        magnification = magnification,
        logscale_x = logscale_x,
        logscale_y = logscale_y,
        legendpos = legendpos,
        legendpadding = legendpadding,
        legendmargin = legendmargin,
        n_Legend_columns = n_Legend_columns,
    )

    fig = Figure(size = figsize)
    ax = Axis(
        fig[1, 1];
        xscale = logscale_x ? log10 : identity,
        yscale = logscale_y ? log10 : identity,
    )

    ax.ylabel = ylabel === nothing ? (normalize ? "normalized count" : "count") : ylabel
    xlabel !== nothing && (ax.xlabel = xlabel)
    plotlabel !== nothing && (ax.title  = plotlabel)

    xlim !== nothing && xlims!(ax, xlim...)
    ylim !== nothing && ylims!(ax, ylim...)

    eps = if logscale_y
        if ylim !== nothing
            ylim[1] * 1e-3
        else
            minpos = minimum(v for (m, _) in data for v in m if v > 0)
            minpos * 1e-3
        end
    else
        -Inf
    end

    for (i, (mean, std)) in enumerate(data)
        @assert length(mean) == length(std)

        if normalize && !isempty(mean)
            max_mean = maximum(mean)
            if max_mean > 0
                mean = mean ./ max_mean
                std = std ./ max_mean
            end
        end

        colors_obs = Makie.theme(:palette).color
        colors = colors_obs isa Observables.Observable ? Observables.to_value(colors_obs) : colors_obs
        color = colors[mod1(i, length(colors))]

        x   = collect(1:length(mean))
        ylo = mean .- std
        yhi = mean .+ std

        if logscale_y
            mask = mean .> 0
            x = x[mask]
            mean = mean[mask]
            ylo = ylo[mask]
            yhi = yhi[mask]

            ylo = max.(ylo, eps)
            yhi = max.(yhi, eps)
        end

        plot_type = plot_types === nothing ? :line : plot_types[i]
        if plot_type == :line
            band!(ax, x, ylo, yhi; color = (color, 0.2))
            if hist_labels === nothing
                isnothing(linewidth) ? lines!(ax, x, mean; color = color) : lines!(ax, x, mean; color = color, linewidth = linewidth)
            else
                isnothing(linewidth) ? lines!(ax, x, mean; color = color, label = hist_labels[i]) : lines!(ax, x, mean; color = color, linewidth = linewidth, label = hist_labels[i])
            end
        elseif plot_type == :scatter
            if hist_labels === nothing
                isnothing(markersize) ? scatter!(ax, x, mean; color = color) : scatter!(ax, x, mean; color = color, markersize = markersize)
            else
                isnothing(markersize) ? scatter!(ax, x, mean; color = color, label = hist_labels[i]) : scatter!(ax, x, mean; color = color, label = hist_labels[i], markersize = markersize)
            end
            err = mean .- ylo
            Makie.errorbars!(ax, x, mean, err, err; color = color)
        else
            error("plot_types entries must be :line or :scatter")
        end
    end

    if hist_labels !== nothing
        legend_kwargs = (position = legendpos,)
        legendpadding !== nothing && (legend_kwargs = merge(legend_kwargs, (padding = legendpadding,)))
        legendmargin !== nothing && (legend_kwargs = merge(legend_kwargs, (margin = legendmargin,)))
        n_Legend_columns > 1 && (legend_kwargs = merge(legend_kwargs, (nbanks = n_Legend_columns,)))
        axislegend(ax; legend_kwargs...)
    end

    return return_axis ? (fig, ax) : fig
end

"""
    compute_sigma_evolution(
        X::AbstractMatrix{<:Real};
        batchsize::Int = 1,
        bin_average::Int = 1,
    )::Tuple{Vector{Int}, Matrix{Float64}}

Compute σ_k(N) for increasing sample size N and (optionally) averaged bins.

Inputs
------
- X: matrix of size (Nsamples, nbins), output of `densify_hists`
- batchsize: step size in sample number N
- bin_average: number of original bins merged into one (≥ 1)

Returns
-------
- Ns :: Vector{Int}
    sample sizes used (N values)
- σ  :: Matrix{Float64}
    σ[j, k] = std of bin k at sample size Ns[j]
"""
function compute_sigma_evolution(
    X::AbstractMatrix{<:Real};
    batchsize::Int = 1,
    bin_average::Int = 1,
)::Tuple{Vector{Int}, Matrix{Float64}}

    @assert batchsize ≥ 1
    @assert bin_average ≥ 1

    Nsamples, nbins = size(X)

    # ---- bin averaging -----------------------------------------------------
    nbins_eff = cld(nbins, bin_average)

    Xb = if bin_average == 1
        X
    else
        Xavg = zeros(Float64, Nsamples, nbins_eff)
        for k in 1:nbins_eff
            lo = (k-1)*bin_average + 1
            hi = min(k*bin_average, nbins)
            Xavg[:, k] .= mean(@view X[:, lo:hi]; dims=2)
        end
        Xavg
    end

    # ---- sample sizes ------------------------------------------------------
    Ns = collect(batchsize:batchsize:Nsamples)
    nsteps = length(Ns)

    # ---- cumulative statistics --------------------------------------------
    csum  = cumsum(Xb; dims=1)
    csum2 = cumsum(Xb.^2; dims=1)


    σ = zeros(Float64, nsteps, nbins_eff)

    for (j, N) in enumerate(Ns)
        μ   = view(csum, N, :) ./ N
        var = max.(view(csum2, N, :) ./ N .- μ.^2, 0.0)
        σ[j, :] .= sqrt.(var)
    end

    return Ns, σ
end

"""
    compute_mu_evolution(
        X::AbstractMatrix{<:Real};
        batchsize::Int = 1,
        bin_average::Int = 1,
    )::Tuple{Vector{Int}, Matrix{Float64}}

Compute μₖ(n) (cumulative means) over sample size N.

Returns:
- Ns :: Vector{Int}
- μ  :: Matrix{Float64}  (μ[j, k] = mean of bin k at sample size Ns[j])
"""
function compute_mu_evolution(
    X::AbstractMatrix{<:Real};
    batchsize::Int = 1,
    bin_average::Int = 1,
)::Tuple{Vector{Int}, Matrix{Float64}}

    @assert batchsize ≥ 1
    @assert bin_average ≥ 1

    Nsamples, nbins = size(X)

    # ---- bin averaging -----------------------------------------------------
    nbins_eff = cld(nbins, bin_average)

    Xb = if bin_average == 1
        X
    else
        Xavg = zeros(Float64, Nsamples, nbins_eff)
        for k in 1:nbins_eff
            lo = (k-1)*bin_average + 1
            hi = min(k*bin_average, nbins)
            Xavg[:, k] .= mean(@view X[:, lo:hi]; dims=2)
        end
        Xavg
    end

    # ---- sample sizes ------------------------------------------------------
    Ns = collect(batchsize:batchsize:Nsamples)
    nsteps = length(Ns)

    # ---- cumulative means --------------------------------------------------
    csum = cumsum(Xb; dims=1)
    μ = zeros(Float64, nsteps, nbins_eff)

    for (j, N) in enumerate(Ns)
        μ[j, :] .= view(csum, N, :) ./ N
    end

    return Ns, μ
end

"""
   convergence_plots_std_change(
        hists::Vector{<:AbstractDict{<:Integer,<:Integer}};
        batchsize::Int = 1,
        bin_average::Bool = false,
        xlim = nothing,
        ylim1 = nothing,
        ylim2 = nothing,
        xlabel::Union{Nothing,AbstractString} = nothing,
        n_Legend_columns::Int = 1,
    )::Tuple{Figure,Figure}

Compute convergence of histogram standard deviations.

Returns:
1. Figure with Δσₖ(N) vs bin index (log-y), one line per batch.
2. Figure with ⟨Δσ⟩ vs sample size N (log-log).
"""
function convergence_plots_std_change(
    hists::Vector{<:AbstractDict};
    batchsize::Int = 1,
    bin_average::Int = 1,
    xlim = nothing,
    ylim1 = nothing,
    ylim2 = nothing,
    xlabel::AbstractString = "histogram bins",
    n_Legend_columns::Int = 1,
    double_column::Bool=false,
    magnification::Real=1,
)::Figure

    X = densify_hists(hists)
    nbins = size(X, 2)

    Ns, σ = compute_sigma_evolution(
        X;
        batchsize = batchsize,
        bin_average = bin_average,
    )

    nsteps, nbins_eff = size(σ)

    Δσ = Vector{Vector{Float64}}(undef, nsteps)
    Δσ_avg = zeros(Float64, nsteps)

    for j in 1:nsteps
        if j == 1
            Δσ[j] = σ[j, :]
        else
            Δσ[j] = abs.(σ[j, :] .- σ[j-1, :])
        end
        Δσ_avg[j] = mean(Δσ[j])
    end
    # Set up continuous colormap for sample size
    cmap = :viridis
    Ns_min, Ns_max = minimum(Ns), maximum(Ns)
    normN(N) = (N - Ns_min) / (Ns_max - Ns_min)

    figsize = apply_paper_theme!(
        double_column = double_column,
        magnification = magnification,
        logscale_y = true,
        n_Legend_columns = n_Legend_columns,
    )

    fig = Figure(size=figsize)

    ax1 = Axis(
        fig[1, 1];
        yscale = log10,
    )

    for (i, dσ) in enumerate(Δσ)
        y = dσ
        x = if bin_average == 1
            collect(1:length(y))
        else
            centers = Vector{Float64}(undef, length(y))
            for k in 1:length(y)
                lo = (k - 1) * bin_average + 1
                hi = min(k * bin_average, nbins)
                centers[k] = (lo + hi) / 2
            end
            centers
        end
        mask = y .> 0
        lines!(ax1, x[mask], y[mask];
               color = get(ColorSchemes.viridis, normN(Ns[i])))
    end

    ax1.xlabel = xlabel
    ax1.ylabel = bin_average > 1 ? "Δσ (bin-averaged)" : "Δσₖ"
    xlim !== nothing && xlims!(ax1, xlim...)
    ylim1 !== nothing && ylims!(ax1, ylim1...)

    apply_paper_theme!(
        double_column = false,
        magnification = magnification,
        logscale_x = true,
        logscale_y = true,
        n_Legend_columns = n_Legend_columns,
    )

    ax2 = Axis(
        fig[2, 1];
        xscale = log10,
        yscale = log10,
    )
    mask = Δσ_avg .> 0
    lines!(ax2, Ns[mask], Δσ_avg[mask];
           color = get.(Ref(ColorSchemes.viridis), normN.(Ns[mask])))

    ax2.xlabel = "sample size"
    ax2.ylabel = "⟨Δσ⟩"
    ylim2 !== nothing && ylims!(ax2, ylim2...)

    # Add a colorbar legend for sample size N
    Colorbar(fig[1:2, 2];
        colormap = cmap,
        limits = (Ns_min, Ns_max),
        label = "sample size"
    )

    return fig
end

# Fit σ(n) ≈ σinf + A n^{-α} using reduced objective (see boxed equation)
function fit_sigma_infty_alpha(
    σn::AbstractVector{<:Real},
    ns::AbstractVector{<:Real};
    σinf_init::Union{Nothing,Real}=nothing,
    α_init::Real = 0.5,
    bounds_σinf::Union{Nothing,Tuple{Real,Real}}=nothing,
    bounds_α::Tuple{Real,Real} = (1e-3, 5.0),
    fix_sigma_inf::Bool = true,
)::NamedTuple
    # Initial guess

    if σinf_init === nothing
        σinf_init = σn[end]
    end

    if bounds_σinf === nothing
        bounds_σinf = (0.5 * σn[end], 1.5 * σn[end])
    end   

    if fix_sigma_inf
        σinf_fixed = σinf_init
        if σinf_fixed === nothing
            σinf_fixed = σn[end]
        end
        if bounds_σinf !== nothing
            σinf_fixed = clamp(σinf_fixed, bounds_σinf[1], bounds_σinf[2])
        end

        function obj(x)
            α = clamp(x[1], bounds_α[1], bounds_α[2])
            numer = sum(ns.^(-α) .* abs.(σn .- σinf_fixed))
            denom = sum(ns.^(-2α))
            A = numer / denom
            r = abs.(σn .- σinf_fixed) .- A .* ns.^(-α)
            return sum(r.^2)
        end

        result = Optim.optimize(obj, [α_init], NelderMead())
        α_hat = Optim.minimizer(result)[1]
        α_hat = clamp(α_hat, bounds_α[1], bounds_α[2])
        σinf_hat = σinf_fixed
        A_hat = sum(ns.^(-α_hat) .* abs.(σn .- σinf_hat)) / sum(ns.^(-2α_hat))
        fmin = Optim.minimum(result)
    else
        x0 = [σinf_init, α_init]

        # Objective function (reduced, as in boxed equation)
        function obj(x)
            σinf = length(x) == 2 ? x[1] : (σinf_init === nothing ? σn[end] : σinf_init)
            α = length(x) == 2 ? x[2] : x[1]
            # Clamp α to bounds
            α = clamp(α, bounds_α[1], bounds_α[2])
            # Clamp σinf to bounds
            σinf = clamp(σinf, bounds_σinf[1], bounds_σinf[2])
            # Compute A (reduced least squares)
            numer = sum(ns.^(-α) .* abs.(σn .- σinf))
            denom = sum(ns.^(-2α))
            A = numer / denom
            r = abs.(σn .- σinf) .- A .* ns.^(-α)
            return sum(r.^2)
        end
        
        # Run optimization (Nelder-Mead, no gradients)
        result = Optim.optimize(obj, x0, NelderMead())
        xopt = Optim.minimizer(result)
        σinf_hat, α_hat = xopt
        # Clamp to bounds for output
        α_hat = clamp(α_hat, bounds_α[1], bounds_α[2])
        if bounds_σinf !== nothing
            σinf_hat = clamp(σinf_hat, bounds_σinf[1], bounds_σinf[2])
        end
        # Compute A at optimum
        A_hat = sum(ns.^(-α_hat) .* abs.(σn .- σinf_hat)) / sum(ns.^(-2α_hat))
        fmin = Optim.minimum(result)
    end
    return (σinf = σinf_hat, α = α_hat, A = A_hat, objective = fmin)
end

# Fit μ(n) ≈ μinf + B n^{-β} using reduced objective (analogous to σ-fit)
function fit_mu_infty_beta(
    μn::AbstractVector{<:Real},
    ns::AbstractVector{<:Real};
    μinf_init::Union{Nothing,Real}=nothing,
    β_init::Real = 0.5,
    bounds_μinf::Union{Nothing,Tuple{Real,Real}}=nothing,
    bounds_β::Tuple{Real,Real} = (1e-3, 5.0),
    fix_sigma_inf::Bool = true,
)::NamedTuple
    if μinf_init === nothing
        μinf_init = μn[end]
    end

    if bounds_μinf === nothing
        δ = abs(μn[end])
        δ == 0 && (δ = 1.0)
        bounds_μinf = (μn[end] - δ, μn[end] + δ)
    end

    if fix_sigma_inf
        μinf_fixed = μinf_init
        if μinf_fixed === nothing
            μinf_fixed = μn[end]
        end
        μinf_fixed = clamp(μinf_fixed, bounds_μinf[1], bounds_μinf[2])

        function obj(x)
            β = clamp(x[1], bounds_β[1], bounds_β[2])
            numer = sum(ns.^(-β) .* abs.(μn .- μinf_fixed))
            denom = sum(ns.^(-2β))
            B = numer / denom
            r = abs.(μn .- μinf_fixed) .- B .* ns.^(-β)
            return sum(r.^2)
        end

        result = Optim.optimize(obj, [β_init], NelderMead())
        β_hat = Optim.minimizer(result)[1]
        β_hat = clamp(β_hat, bounds_β[1], bounds_β[2])
        μinf_hat = μinf_fixed
        B_hat = sum(ns.^(-β_hat) .* abs.(μn .- μinf_hat)) / sum(ns.^(-2β_hat))
        fmin = Optim.minimum(result)
    else
        x0 = [μinf_init, β_init]

        function obj(x)
            μinf = length(x) == 2 ? x[1] : (μinf_init === nothing ? μn[end] : μinf_init)
            β = length(x) == 2 ? x[2] : x[1]
            β = clamp(β, bounds_β[1], bounds_β[2])
            μinf = clamp(μinf, bounds_μinf[1], bounds_μinf[2])
            numer = sum(ns.^(-β) .* abs.(μn .- μinf))
            denom = sum(ns.^(-2β))
            B = numer / denom
            r = abs.(μn .- μinf) .- B .* ns.^(-β)
            return sum(r.^2)
        end

        result = Optim.optimize(obj, x0, NelderMead())
        xopt = Optim.minimizer(result)
        μinf_hat, β_hat = xopt
        β_hat = clamp(β_hat, bounds_β[1], bounds_β[2])
        μinf_hat = clamp(μinf_hat, bounds_μinf[1], bounds_μinf[2])
        B_hat = sum(ns.^(-β_hat) .* abs.(μn .- μinf_hat)) / sum(ns.^(-2β_hat))
        fmin = Optim.minimum(result)
    end
    return (μinf = μinf_hat, β = β_hat, B = B_hat, objective = fmin)
end

"""
    fit_sigma_convergence(
        X::AbstractMatrix{<:Real};
        batchsize::Int = 1,
        bin_average::Int = 1,
        bounds_σinf::Union{Nothing,Tuple{Real,Real}} = nothing,
        bounds_α::Tuple{Real,Real} = (1e-3, 5.0),
        α_init::Real = 0.5,
    )

Compute σₖ(n) via `compute_sigma_evolution`, then fit
    σₖ(n) ≈ σ∞ + A n^{-α}
for:
1. every (possibly averaged) histogram bin k
2. the bin-averaged mean ⟨σ(n)⟩

Returns a NamedTuple with fields:
- Ns               :: Vector{Int}
- σ                :: Matrix{Float64}   (σ[j, k])
- bin_fits         :: Vector{Union{NamedTuple,Missing}} (may contain `missing` if not enough data in a bin)
- mean_fit         :: NamedTuple
"""
function fit_sigma_convergence(
    X::AbstractMatrix{<:Real};
    batchsize::Int = 1,
    bin_average::Int = 1,
    bounds_σinf::Union{Nothing,Tuple{Real,Real}} = nothing,
    bounds_α::Tuple{Real,Real} = (1e-3, 5.0),
    σinf_init::Union{Nothing,Real} = nothing,
    α_init::Real = 0.5,
    fix_sigma_inf::Bool = true,
)

    # ---- compute σ evolution ---------------------------------------------
    Ns, σ = compute_sigma_evolution(
        X;
        batchsize = batchsize,
        bin_average = bin_average,
    )

    ns = Float64.(Ns)
    nsteps, nbins = size(σ)

    # ---- per-bin fits -----------------------------------------------------
    bin_fits = Vector{Union{NamedTuple,Missing}}(undef, nbins)

    for k in 1:nbins
        σn = view(σ, :, k)

        # include zeros; only drop non-finite values
        mask = isfinite.(σn)
        ns_k = ns[mask]
        σn_k = σn[mask]

        if length(σn_k) < 3
            bin_fits[k] = missing
        else
            bin_fits[k] = fit_sigma_infty_alpha(
                σn_k,
                ns_k;
                σinf_init = σinf_init,
                bounds_σinf = bounds_σinf,
                bounds_α = bounds_α,
                α_init = α_init,
                fix_sigma_inf = fix_sigma_inf,
            )
        end
    end

    # ---- mean σ(n) fit ----------------------------------------------------
    σmean = vec(mean(σ; dims=2))
    mask = isfinite.(σmean)
    ns_m = ns[mask]
    σm   = σmean[mask]

    mean_fit = fit_sigma_infty_alpha(
        σm,
        ns_m;
        σinf_init = σm[end],
        bounds_σinf = bounds_σinf,
        bounds_α = bounds_α,
        α_init = α_init,
        fix_sigma_inf = fix_sigma_inf,
    )

    return (
        Ns = Ns,
        σ = σ,
        bin_fits = bin_fits,
        mean_fit = mean_fit,
    )
end

"""
    fit_mu_convergence(
        X::AbstractMatrix{<:Real};
        batchsize::Int = 1,
        bin_average::Int = 1,
        bounds_μinf::Union{Nothing,Tuple{Real,Real}} = nothing,
        bounds_β::Tuple{Real,Real} = (1e-3, 5.0),
        μinf_init::Union{Nothing,Real} = nothing,
        β_init::Real = 0.5,
    )

Compute μₖ(n) via `compute_mu_evolution`, then fit
    μₖ(n) ≈ μ∞ + B n^{-β}
for:
1. every (possibly averaged) histogram bin k
2. the bin-averaged mean ⟨μ(n)⟩
"""
function fit_mu_convergence(
    X::AbstractMatrix{<:Real};
    batchsize::Int = 1,
    bin_average::Int = 1,
    bounds_μinf::Union{Nothing,Tuple{Real,Real}} = nothing,
    bounds_β::Tuple{Real,Real} = (1e-3, 5.0),
    μinf_init::Union{Nothing,Real} = nothing,
    β_init::Real = 0.5,
    fix_sigma_inf::Bool = true,
)
    Ns, μ = compute_mu_evolution(
        X;
        batchsize = batchsize,
        bin_average = bin_average,
    )

    ns = Float64.(Ns)
    nsteps, nbins = size(μ)

    bin_fits = Vector{Union{NamedTuple,Missing}}(undef, nbins)

    for k in 1:nbins
        μn = view(μ, :, k)
        mask = isfinite.(μn)
        ns_k = ns[mask]
        μn_k = μn[mask]

        if length(μn_k) < 3
            bin_fits[k] = missing
        else
            bin_fits[k] = fit_mu_infty_beta(
                μn_k,
                ns_k;
                μinf_init = μinf_init,
                bounds_μinf = bounds_μinf,
                bounds_β = bounds_β,
                β_init = β_init,
                fix_sigma_inf = fix_sigma_inf,
            )
        end
    end

    μmean = vec(mean(μ; dims=2))
    mask = isfinite.(μmean)
    ns_m = ns[mask]
    μm   = μmean[mask]

    mean_fit = fit_mu_infty_beta(
        μm,
        ns_m;
        μinf_init = μm[end],
        bounds_μinf = bounds_μinf,
        bounds_β = bounds_β,
        β_init = β_init,
        fix_sigma_inf = fix_sigma_inf,
    )

    return (
        Ns = Ns,
        μ = μ,
        bin_fits = bin_fits,
        mean_fit = mean_fit,
    )
end

"""
    plot_alpha_bins(
        X::AbstractMatrix{<:Real};
        batchsize::Int = 1,
        bin_average::Int = 1,
        bounds_σinf = nothing,
        bounds_α = (1e-3, 5.0),
        α_init = 0.5,
        xlabel::AbstractString = "bin index",
        ylabel::AbstractString = "α",
        double_column::Bool = false,
        magnification::Real = 1.0,
        n_Legend_columns::Int = 1,
    )::Figure

Plot fitted α for all (possibly bin-averaged) histogram bins, together with:
- a horizontal line at α = 1/2
- a horizontal line at α from the mean σ(n)

Bins that do not admit a fit are skipped.
"""
function plot_alpha_bins(
    X::AbstractMatrix{<:Real};
    batchsize::Int = 1,
    bin_average::Int = 1,
    bounds_σinf = nothing,
    bounds_α = (1e-3, 5.0),
    σinf_init::Union{Nothing,Real} = nothing,
    α_init = 0.5,
    N0::Union{Real,Nothing} = nothing,
    bin_plot::Union{Int,Nothing} = nothing,
    xlabel::AbstractString = "histogram bins",
    ylabel::AbstractString = "α",
    double_column::Bool = false,
    magnification::Real = 1.0,
    n_Legend_columns::Int = 1,
    fix_sigma_inf::Bool = true,
    flag_zero_frac::Real = 0.05,
    legendpos = :rt,
    legendpadding = nothing,
    legendmargin = nothing,
    legend::Bool = false,
    plot_mean::Bool = false,
)::Figure

    # --- compute fits ------------------------------------------------------
    nbins_orig = size(X, 2)
    fit = fit_sigma_convergence(
        X;
        batchsize = batchsize,
        bin_average = bin_average,
        bounds_σinf = bounds_σinf,
        bounds_α = bounds_α,
        α_init = α_init,
        fix_sigma_inf = fix_sigma_inf,
    )

    σ = fit.σ
    Ns = fit.Ns
    ns = Float64.(Ns)

    if N0 === nothing
        bin_fits = fit.bin_fits
        mean_fit = fit.mean_fit
    else
        nsteps, nbins = size(σ)
        bin_fits = Vector{Union{NamedTuple,Missing}}(undef, nbins)

        for k in 1:nbins
            σn = view(σ, :, k)
            mask = (σn .> 0) .& (ns .>= N0)
            ns_k = ns[mask]
            σn_k = σn[mask]

            if length(σn_k) < 3
                bin_fits[k] = missing
            else
                bin_fits[k] = fit_sigma_infty_alpha(
                    σn_k,
                    ns_k;
                    σinf_init = σinf_init,
                    bounds_σinf = bounds_σinf,
                    bounds_α = bounds_α,
                    α_init = α_init,
                    fix_sigma_inf = fix_sigma_inf,
                )
            end
        end

        σmean = vec(mean(σ; dims=2))
        mask = (σmean .> 0) .& (ns .>= N0)
        ns_m = ns[mask]
        σm = σmean[mask]
        mean_fit = fit_sigma_infty_alpha(
            σm,
            ns_m;
            σinf_init = σm[end],
            bounds_σinf = bounds_σinf,
            bounds_α = bounds_α,
            α_init = α_init,
            fix_sigma_inf = fix_sigma_inf,
        )
    end
    
    # --- collect valid bin results ----------------------------------------
    αs = Float64[]
    bins = Int[]
    flagged = Bool[]

    for (k, f) in enumerate(bin_fits)
        f === missing && continue
        push!(bins, k)
        push!(αs, f.α)
        σn = view(σ, :, k)
        mask = isfinite.(σn)
        if N0 !== nothing
            mask = mask .& (Ns .>= N0)
        end
        σn_k = σn[mask]
        ns_k = ns[mask]
        if length(σn_k) < 3
            push!(flagged, true)
        else
            frac_zero = mean(σn_k .== 0)
            push!(flagged, frac_zero > flag_zero_frac)
        end
    end

    @assert !isempty(αs) "No bins with valid α fits"

    # --- theme -------------------------------------------------------------
    figsize = apply_paper_theme!(
        double_column = double_column,
        magnification = magnification,
        logscale_x = false,
        logscale_y = false,
        legendpos = legendpos,
        legendpadding = legendpadding,
        legendmargin = legendmargin,
        n_Legend_columns = n_Legend_columns,
    )

    fig_height = bin_plot === nothing ? figsize[2] : 1.6 * figsize[2]
    fig = Figure(size = (figsize[1], fig_height))
    ax  = Axis(fig[1, 1])

    ax.xlabel = xlabel
    ax.ylabel = ylabel

    # --- plot per-bin α ----------------------------------------------------
    xbins = if bin_average == 1
        bins
    else
        centers = Vector{Float64}(undef, length(bins))
        for (i, k) in enumerate(bins)
            lo = (k - 1) * bin_average + 1
            hi = min(k * bin_average, nbins_orig)
            centers[i] = (lo + hi) / 2
        end
        centers
    end

    #scatter!(ax, xbins, αs; markersize = 8)
    lines!(ax, xbins, αs)
    colors_obs = Makie.theme(:palette).color
    colors = colors_obs isa Observables.Observable ? Observables.to_value(colors_obs) : colors_obs
    flag_color = colors[mod1(2, length(colors))]
    i = 1
    while i <= length(flagged)
        if flagged[i]
            j = i
            while j < length(flagged) && flagged[j+1]
                j += 1
            end
            lines!(ax, xbins[i:j], αs[i:j]; color = flag_color)
            i = j + 1
        else
            i += 1
        end
    end

    if bin_plot !== nothing
        bin_plot_avg = bin_average == 1 ? bin_plot : cld(bin_plot, bin_average)
        idx = findfirst(==(bin_plot_avg), bins)
        if idx !== nothing
            scatter!(ax, [xbins[idx]], [αs[idx]]; markersize = 8, color = :black)
        end
    end
    ylo = minimum(αs)
    yhi = maximum(αs)
    pad = yhi == ylo ? 0.05 * max(abs(yhi), 1.0) : 0.05 * (yhi - ylo)
    ylims!(ax, ylo - pad, yhi + pad)

    # --- reference lines ---------------------------------------------------
    hlines!(ax, [0.5];
        linestyle = :dash,
        linewidth = 2 * magnification,
        color = :black,
        label = L"\alpha = \frac{1}{2}",
    )
    hlines!(ax, [0.0];
        linestyle = :solid,
        linewidth = 2 * magnification,
        color = :black,
        label = L"\alpha = 0",
    )

    mean_color = colorant"#D12771"
    if plot_mean
        hlines!(ax, [mean_fit.α];
            linestyle = :dot,
            linewidth = 2 * magnification,
            color = mean_color,
            label = L"\alpha_{\mathrm{mean}}",
        )
    end

    if legend
        legend_kwargs = (position = legendpos,)
        legendpadding !== nothing && (legend_kwargs = merge(legend_kwargs, (padding = legendpadding,)))
        legendmargin !== nothing && (legend_kwargs = merge(legend_kwargs, (margin = legendmargin,)))
        n_Legend_columns > 1 && (legend_kwargs = merge(legend_kwargs, (nbanks = n_Legend_columns,)))
        axislegend(ax; legend_kwargs...)
    end

    if bin_plot !== nothing
        @assert 1 <= bin_plot <= nbins_orig "bin_plot out of range"
        bin_plot_avg = bin_average == 1 ? bin_plot : cld(bin_plot, bin_average)
        @assert 1 <= bin_plot_avg <= length(bin_fits) "bin_plot out of range"
        fit = bin_fits[bin_plot_avg]
        @assert fit !== missing "bin_plot has no valid fit"

        σn = view(σ, :, bin_plot_avg)
        mask = isfinite.(σn)
        if N0 !== nothing
            mask = mask .& (Ns .>= N0)
        end
        ns = Float64.(Ns[mask])
        σn = σn[mask]

        σinf = fit.σinf
        σinf_rounded = round(σinf, sigdigits = 1)
        A = fit.A
        α = fit.α
        A_rounded = round(A, sigdigits = 1)
        upper = σinf .+ A .* ns.^(-α)
        lower = σinf .- A .* ns.^(-α)

        mask = isfinite.(σn) .& isfinite.(upper) .& isfinite.(lower)
        ns = ns[mask]
        σn = σn[mask]
        upper = upper[mask]
        lower = lower[mask]

        ax2 = Axis(fig[2, 1])

        ax2.xlabel = "sample size N"
        ax2.ylabel = "σ(N)"
        ax2.title = latexstring("\\mathrm{bin} = $(bin_plot),\\ A = $(A_rounded),\\ \\sigma_\\infty = $(σinf_rounded)")
        xlims!(ax2, 0, maximum(ns))
        ylo = minimum(σn)
        yhi = maximum(σn)
        pad = yhi == ylo ? 0.05 * max(abs(yhi), 1.0) : 0.05 * (yhi - ylo)
        ylims!(ax2, ylo - pad, yhi + pad)

        lines!(ax2, ns, σn; color = :black)
        scatter!(ax2, ns, σn; markersize = 6, color = :black)

        lines!(ax2, ns, upper; linestyle = :dash, color = mean_color)
        lines!(ax2, ns, lower; linestyle = :dash, color = mean_color)
        hlines!(ax2, [σinf]; linestyle = :dot, color = :black)
        band!(ax2, ns, lower, upper; color = (mean_color, 0.15))
    end

    return fig
end

"""
    plot_beta_bins(
        X::AbstractMatrix{<:Real};
        batchsize::Int = 1,
        bin_average::Int = 1,
        bounds_μinf = nothing,
        bounds_β = (1e-3, 5.0),
        μinf_init = nothing,
        β_init = 0.5,
        xlabel::AbstractString = "bin index",
        ylabel::AbstractString = "β",
        double_column::Bool = false,
        magnification::Real = 1.0,
        n_Legend_columns::Int = 1,
    )::Figure

Plot fitted β for all (possibly bin-averaged) histogram bins, together with:
- a horizontal line at β = 1/2
- a horizontal line at β from the mean μ(n)
"""
function plot_beta_bins(
    X::AbstractMatrix{<:Real};
    batchsize::Int = 1,
    bin_average::Int = 1,
    bounds_μinf = nothing,
    bounds_β = (1e-3, 5.0),
    μinf_init = nothing,
    β_init = 0.5,
    N0::Union{Real,Nothing} = nothing,
    bin_plot::Union{Int,Nothing} = nothing,
    xlabel::AbstractString = "histogram bins",
    ylabel::AbstractString = "β",
    double_column::Bool = false,
    magnification::Real = 1.0,
    n_Legend_columns::Int = 1,
    fix_sigma_inf::Bool = true,
    flag_zero_frac::Real = 0.05,
    legendpos = :rt,
    legendpadding = nothing,
    legendmargin = nothing,
    legend::Bool = false,
    plot_mean::Bool = false,
)::Figure

    nbins_orig = size(X, 2)

    fit = fit_mu_convergence(
        X;
        batchsize = batchsize,
        bin_average = bin_average,
        bounds_μinf = bounds_μinf,
        bounds_β = bounds_β,
        μinf_init = μinf_init,
        β_init = β_init,
        fix_sigma_inf = fix_sigma_inf,
    )

    μ = fit.μ
    Ns = fit.Ns
    ns = Float64.(Ns)

    if N0 === nothing
        bin_fits = fit.bin_fits
        mean_fit = fit.mean_fit
    else
        nsteps, nbins = size(μ)
        bin_fits = Vector{Union{NamedTuple,Missing}}(undef, nbins)

        for k in 1:nbins
            μn = view(μ, :, k)
            mask = isfinite.(μn) .& (ns .>= N0)
            ns_k = ns[mask]
            μn_k = μn[mask]

            if length(μn_k) < 3
                bin_fits[k] = missing
            else
                bin_fits[k] = fit_mu_infty_beta(
                    μn_k,
                    ns_k;
                    μinf_init = μinf_init,
                    bounds_μinf = bounds_μinf,
                    bounds_β = bounds_β,
                    β_init = β_init,
                    fix_sigma_inf = fix_sigma_inf,
                )
            end
        end

        μmean = vec(mean(μ; dims=2))
        mask = isfinite.(μmean) .& (ns .>= N0)
        ns_m = ns[mask]
        μm   = μmean[mask]

        mean_fit = fit_mu_infty_beta(
            μm,
            ns_m;
            μinf_init = μm[end],
            bounds_μinf = bounds_μinf,
            bounds_β = bounds_β,
            β_init = β_init,
            fix_sigma_inf = fix_sigma_inf,
        )
    end

    βs = Float64[]
    bins = Int[]
    flagged = Bool[]

    for (k, f) in enumerate(bin_fits)
        f === missing && continue
        push!(bins, k)
        push!(βs, f.β)
        μn = view(μ, :, k)
        mask = isfinite.(μn)
        if N0 !== nothing
            mask = mask .& (Ns .>= N0)
        end
        μn_k = μn[mask]
        ns_k = ns[mask]
        if length(μn_k) < 3
            push!(flagged, true)
        else
            frac_zero = mean(μn_k .== 0)
            push!(flagged, frac_zero > flag_zero_frac)
        end
    end

    @assert !isempty(βs) "No bins with valid β fits"

    figsize = apply_paper_theme!(
        double_column = double_column,
        magnification = magnification,
        logscale_x = false,
        logscale_y = false,
        legendpos = legendpos,
        legendpadding = legendpadding,
        legendmargin = legendmargin,
        n_Legend_columns = n_Legend_columns,
    )

    fig_height = bin_plot === nothing ? figsize[2] : 1.6 * figsize[2]
    fig = Figure(size = (figsize[1], fig_height))
    ax  = Axis(fig[1, 1])

    ax.xlabel = xlabel
    ax.ylabel = ylabel

    xbins = if bin_average == 1
        bins
    else
        centers = Vector{Float64}(undef, length(bins))
        for (i, k) in enumerate(bins)
            lo = (k - 1) * bin_average + 1
            hi = min(k * bin_average, nbins_orig)
            centers[i] = (lo + hi) / 2
        end
        centers
    end

    #scatter!(ax, xbins, βs; markersize = 8)
    lines!(ax, xbins, βs)
    colors_obs = Makie.theme(:palette).color
    colors = colors_obs isa Observables.Observable ? Observables.to_value(colors_obs) : colors_obs
    flag_color = colors[mod1(2, length(colors))]
    i = 1
    while i <= length(flagged)
        if flagged[i]
            j = i
            while j < length(flagged) && flagged[j+1]
                j += 1
            end
            lines!(ax, xbins[i:j], βs[i:j]; color = flag_color)
            i = j + 1
        else
            i += 1
        end
    end

    if bin_plot !== nothing
        bin_plot_avg = bin_average == 1 ? bin_plot : cld(bin_plot, bin_average)
        idx = findfirst(==(bin_plot_avg), bins)
        if idx !== nothing
            scatter!(ax, [xbins[idx]], [βs[idx]]; markersize = 8, color = :black)
        end
    end
    ylo = minimum(βs)
    yhi = maximum(βs)
    pad = yhi == ylo ? 0.05 * max(abs(yhi), 1.0) : 0.05 * (yhi - ylo)
    ylims!(ax, ylo - pad, yhi + pad)

    hlines!(ax, [0.5];
        linestyle = :dash,
        linewidth = 2 * magnification,
        color = :black,
        label = L"\beta = \frac{1}{2}",
    )
    hlines!(ax, [0.0];
        linestyle = :solid,
        linewidth = 2 * magnification,
        color = :black,
        label = L"\beta = 0",
    )

    mean_color = colorant"#D12771"
    if plot_mean
        hlines!(ax, [mean_fit.β];
            linestyle = :dot,
            linewidth = 2 * magnification,
            color = mean_color,
            label = L"\beta_{\mathrm{mean}}",
        )
    end

    if legend
        legend_kwargs = (position = legendpos,)
        legendpadding !== nothing && (legend_kwargs = merge(legend_kwargs, (padding = legendpadding,)))
        legendmargin !== nothing && (legend_kwargs = merge(legend_kwargs, (margin = legendmargin,)))
        n_Legend_columns > 1 && (legend_kwargs = merge(legend_kwargs, (nbanks = n_Legend_columns,)))
        axislegend(ax; legend_kwargs...)
    end

    if bin_plot !== nothing
        @assert 1 <= bin_plot <= nbins_orig "bin_plot out of range"
        bin_plot_avg = bin_average == 1 ? bin_plot : cld(bin_plot, bin_average)
        @assert 1 <= bin_plot_avg <= length(bin_fits) "bin_plot out of range"
        fit = bin_fits[bin_plot_avg]
        @assert fit !== missing "bin_plot has no valid fit"

        μn = view(μ, :, bin_plot_avg)
        mask = isfinite.(μn)
        if N0 !== nothing
            mask = mask .& (Ns .>= N0)
        end
        ns = Float64.(Ns[mask])
        μn = μn[mask]

        μinf = fit.μinf
        B = fit.B
        β = fit.β
        B_rounded = round(B, sigdigits = 1)
        upper = μinf .+ B .* ns.^(-β)
        lower = μinf .- B .* ns.^(-β)

        mask = isfinite.(μn) .& isfinite.(upper) .& isfinite.(lower)
        ns = ns[mask]
        μn = μn[mask]
        upper = upper[mask]
        lower = lower[mask]

        ax2 = Axis(fig[2, 1])

        ax2.xlabel = L"sample size $N$"
        ax2.ylabel = L"\mu(N)"
        ax2.title = latexstring("\\mathrm{bin} = $(bin_plot),\\ B = $(B_rounded),\\ \\mu_\\infty = $(μinf)")
        xlims!(ax2, 0, maximum(ns))
        ylo = minimum(μn)
        yhi = maximum(μn)
        pad = yhi == ylo ? 0.05 * max(abs(yhi), 1.0) : 0.05 * (yhi - ylo)
        ylims!(ax2, ylo - pad, yhi + pad)

        lines!(ax2, ns, μn; color = :black)
        scatter!(ax2, ns, μn; markersize = 6, color = :black)

        lines!(ax2, ns, upper; linestyle = :dash, color = mean_color)
        lines!(ax2, ns, lower; linestyle = :dash, color = mean_color)
        hlines!(ax2, [μinf]; linestyle = :dot, color = :black)
        band!(ax2, ns, lower, upper; color = (mean_color, 0.15))
    end

    return fig
end

function create_grid_and_plot(
    size::Int,
    lattice::String,
    rotation_angle::Float64;
    box::Tuple{Tuple{Float64,Float64},Tuple{Float64,Float64}}=((-1.,-1.),(1.,1.)),
    segment_ratio::Float64=2.,
    segment_angle::Float64=60.,
    shell_thickness::Union{Nothing,Float64} = nothing,
    markersize::Real = 4,
    magnification::Float64 = 1.,
    fig_path::Union{Nothing,String}=nothing,
    )
    mink = CS.MinkowskiManifold{2}()
    quad_grid_unsorted = QG.generate_grid_2d_in_box(
        size,
        lattice,
        box; 
        rotate_deg = rotation_angle, 
        b = segment_ratio, 
        gamma_deg = segment_angle,
        shell_thickness = shell_thickness)

    quad_grid = QG.sort_grid_by_time_from_manifold(mink, quad_grid_unsorted)

    figsize = apply_paper_theme!(; magnification = magnification)
    fig = Figure(size = figsize)
    ax = Axis(fig[1,1])
    ax.xlabel="x"
    ax.ylabel="t"
    scatter!(ax,quad_grid; markersize = magnification * markersize)

    if !isnothing(fig_path)
        save(fig_path, fig)
    end
    return fig, ax
end

function fourier_transform_grid_deviation(
    comp_hist::Vector{Float64}, 
    size::Int64, 
    lattice::String; 
    P_max::Float64=300., 
    rng::Random.AbstractRNG=Random.GLOBAL_RNG, 
    segment_ratio::Float64=1., 
    segment_angle::Float64=60., 
    rotation_angle::Union{Float64,Nothing}=nothing,
    fig_path::Union{Nothing,String}=nothing,
    magnification::Real=1.,
    linewidth::Real=1,
    ylim::Union{Tuple{Float64,Float64},Nothing} = nothing,
    xtick_fracs::Union{Nothing,Vector{<:Any}}=nothing,
    max_peak_order::Int = 5 
    )
    
    grid_cset, _, _, _ = QG.create_grid_causet_in_boundary_2D_polynomial_manifold(size, lattice, CS.BoxBoundary{2}(((-1.,-1.),(1.,1.))), rng, 1, 2.; a = 1., b = segment_ratio, gamma_deg=segment_angle, rotate_deg=rotation_angle)
    grid_cset_abundances = CS.cardinality_abundances(grid_cset)
    idx = findfirst(iszero, comp_hist)-1
    @show idx
    r_comp_grid_man = grid_cset_abundances[2:idx] ./ comp_hist[2:idx] .- 1
    r_dev_fou_comp_grid_man = abs.(fft(r_comp_grid_man))
    r_dev_freqs_comp_grid_man = (0:length(r_dev_fou_comp_grid_man)-1) ./ length(r_dev_fou_comp_grid_man)  # cycles per sample    

    f_min = 1 / P_max

    freqs = r_dev_freqs_comp_grid_man
    half = 1:fld(length(freqs), 2) 
    keep = [i for i in half if freqs[i] >= f_min]
    peak_idx = keep[argmax(r_dev_fou_comp_grid_man[keep])]
    f_peak = r_dev_freqs_comp_grid_man[peak_idx]
    P_est = 1 / f_peak           # period in “bins”

    @show f_peak P_est

    min_freq_for_peaks = 1 / 13
    keep_for_peaks = [i for i in keep if r_dev_freqs_comp_grid_man[i] >= min_freq_for_peaks]
    if !isempty(keep_for_peaks)
        idxs = sortperm(r_dev_fou_comp_grid_man[keep_for_peaks]; rev=true)
        printed_periods = Float64[]
        for i in idxs
            f = r_dev_freqs_comp_grid_man[keep_for_peaks[i]]
            P = 1 / f
            # Skip peaks that are effectively the same period as an earlier (stronger) peak.
            if any(abs(P - P0) <= 0.02 for P0 in printed_periods)
                continue
            end
            A = 2 * abs(r_dev_fou_comp_grid_man[keep_for_peaks[i]]) / length(r_comp_grid_man)
            println("f = ", f, "  P ≈ ", P, "  A = ", A)
            push!(printed_periods, P)
            if length(printed_periods) >= max_peak_order
                break
            end
        end
    end

    figsize = apply_paper_theme!(; magnification = magnification)
    fig = Figure(size = figsize)
    ax = Axis(fig[1,1])

    if xtick_fracs !== nothing
        xticks = collect(xtick_fracs)
        if !isempty(xticks)
            labels = map(xtick_fracs) do x
                if x isa Rational
                    n = numerator(x)
                    d = denominator(x)
                    if d == 1
                        string(n)
                    elseif n < 0
                        LaTeXStrings.LaTeXString("-\\frac{$(abs(n))}{$d}")
                    else
                        LaTeXStrings.LaTeXString("\\frac{$n}{$d}")
                    end
                else
                    Printf.@sprintf("%.2f", Float64(x))
                end
            end
            ax.xticks = (xticks, labels)
            vlines!(ax, xticks; color=(:black,1.), linestyle=:dash, linewidth = magnification * linewidth)
        end
    end

    lines!(ax, r_dev_freqs_comp_grid_man[keep], r_dev_fou_comp_grid_man[keep]; linewidth = magnification * linewidth)
    ax.xlabel = "frequency (cycles per bin)"
    ax.ylabel = L"\mathcal{F}(\mathcal{S}_n^{\mathrm{grid}} / \mathcal{S}_n^{\mathrm{man}} -1)"

    xlims!(ax, (0.,0.51))
    if !isnothing(ylim)
        ylims!(ax, ylim)
    end

    ax.xminorticksvisible = false
    ax.xminorgridvisible = false

    if !isnothing(fig_path)
       save(fig_path, fig)
    end

    fig
end
