#!/usr/bin/env julia

import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using InteractiveUtils
using LinearAlgebra
using Dates

using ArgParse

s = ArgParseSettings()

@add_arg_table s begin
    "--kind"
        help = "Dataset kind (e.g. minkowski_sprinkling, minkowski_quasicrystal, grid, ...)"
        arg_type = String
        required = true

    "--D"
        help = "Dimensionality (currently only supported for minkowski_sprinkling and manifoldlike_simply_connected kinds)"
        arg_type = Int
        required = false
    
    "--cut_restriction"
        arg_type = String
        help = "Restricts allowed topological cuts (for kind manifoldlike_non_simply_connected). Can be \"boundary_cuts\" or \"free_cuts\")"
        required = false

    "--size"
        help = "Causal set size"
        arg_type = Int
        required = true

    "--num_csets"
        help = "Number of causal sets to generate"
        arg_type = Int
        required = true

    "--seed"
        help = "Global RNG seed"
        arg_type = Int
        required = true

    "--num_processes"
        help = "Number of worker processes for statistics"
        arg_type = Int
        default = 1

    "--batchsize"
        help = "Batch size for dataset creation"
        arg_type = Int
        default = 100

    "--outdir"
        help = "Output directory for dataset, statistics, and config"
        arg_type = String
        required = true
end

args = parse_args(s)

outdir = args["outdir"]
isdir(outdir) || mkpath(outdir)

dataset_out = joinpath(outdir, "dataset.jld2")
stats_out   = joinpath(outdir, "statistics.jld2")
config_out  = joinpath(outdir, "config.yaml")

# ---------------------------------------------------------------------
# Overwriting warnings
# ---------------------------------------------------------------------
if isfile(dataset_out)
    @warn "Dataset output file already exists and will be overwritten" path=dataset_out
end

if isfile(stats_out)
    @warn "Statistics output file already exists and will be overwritten" path=stats_out
end

# ---------------------------------------------------------------------
# Paths to scripts
# ---------------------------------------------------------------------
dataset_script = joinpath(@__DIR__, "make_analysis_dataset_sequential.jl")
stats_script   = joinpath(@__DIR__, "make_analysis_statistics.jl")

isfile(dataset_script) || error("Dataset script not found: $dataset_script")
isfile(stats_script)   || error("Statistics script not found: $stats_script")

# ---------------------------------------------------------------------
# Snapshot scripts for reproducibility
# ---------------------------------------------------------------------
dataset_script_copy = joinpath(outdir, basename(dataset_script))
stats_script_copy   = joinpath(outdir, basename(stats_script))

cp(dataset_script, dataset_script_copy; force=true)
cp(stats_script,   stats_script_copy;   force=true)

driver_script = @__FILE__
driver_script_copy = joinpath(outdir, basename(driver_script))
cp(driver_script, driver_script_copy; force=true)

# ---------------------------------------------------------------------
# Snapshot environment files for reproducibility
# ---------------------------------------------------------------------
project_toml = joinpath(@__DIR__, "Project.toml")
manifest_toml = joinpath(@__DIR__, "Manifest.toml")

if isfile(project_toml)
    cp(project_toml, joinpath(outdir, "Project.toml"); force=true)
else
    @warn "Project.toml not found in analysis directory"
end

if isfile(manifest_toml)
    cp(manifest_toml, joinpath(outdir, "Manifest.toml"); force=true)
else
    @warn "Manifest.toml not found in analysis directory"
end

using YAML

config = Dict(
    "kind"              => args["kind"],
    "size"              => args["size"],
    "num_csets"         => args["num_csets"],
    "seed"              => args["seed"],
    "batchsize"         => args["batchsize"],
    "num_processes"     => args["num_processes"],
    "dataset_out"       => dataset_out,
    "stats_out"         => stats_out,
    "julia_version"     => string(VERSION),
    "os"                => Sys.KERNEL,
    "arch"              => Sys.ARCH,
    "julia_optimization" => "O3",
    "julia_threads"      => Threads.nthreads(),
    "blas_threads"       => LinearAlgebra.BLAS.get_num_threads(),
    "datetime_utc"       => Dates.format(Dates.now(Dates.UTC), dateformat"yyyy-mm-ddTHH:MM:SSZ"),
)
args["D"] !== nothing && (config["dimension"] = args["D"])
args["cut_restriction"] !== nothing && (config["cut_restriction"] = args["cut_restriction"])

YAML.write_file(config_out, config)

readme_out = joinpath(outdir, "README.txt")

open(readme_out, "w") do io
    println(io, "REPRODUCIBILITY INFORMATION")
    println(io, "==========================")
    println(io)
    println(io, "This directory contains a self-contained computational artifact")
    println(io, "generated with the QuantumGrav analysis pipeline.")
    println(io)
    println(io, "Contents:")
    println(io, "  - dataset.jld2        : generated causal-set dataset")
    println(io, "  - statistics.jld2     : derived statistics")
    println(io, "  - config.yaml         : all runtime parameters and metadata")
    println(io, "  - Project.toml        : Julia project environment")
    println(io, "  - Manifest.toml       : exact dependency snapshot")
    println(io, "  - make_analysis_*.jl  : exact scripts executed")
    println(io, "  - make_analysis_dataset_and_statistics.jl : pipeline driver script")
    println(io)
    println(io, "To reproduce this dataset and statistics:")
    println(io)
    println(io, "1. Ensure Julia with version:")
    println(io, "     ", string(VERSION))
    println(io)
    println(io, "2. Activate the environment in this directory:")
    println(io, "     julia --project=. ")
    println(io)
    println(io, "3. Re-run the pipeline using the copied driver script in this directory:")
    println(io, "     julia -O3 make_analysis_dataset_and_statistics.jl \\")
    println(io, "         --kind ", args["kind"], " \\")
    if args["D"] !== nothing
        println(io, "         --D ", args["D"], " \\")
    end 
    if args["cut_restriction"] !== nothing
        println(io, "         --cut_restriction ", args["cut_restriction"], " \\")
    end
    println(io, "         --size ", args["size"], " \\")
    println(io, "         --num_csets ", args["num_csets"], " \\")
    println(io, "         --batchsize ", args["batchsize"], " \\")
    println(io, "         --num_processes ", args["num_processes"], " \\")
    println(io, "         --seed ", args["seed"], " \\")
    println(io, "         --outdir <NEW_OUTPUT_DIRECTORY>")
    println(io)
    println(io, "Notes:")
    println(io, "  - For bitwise-identical dependency resolution, use the provided Manifest.toml.")
    println(io, "  - Results may still depend on OS, architecture, and BLAS implementation (see config.yaml).")
end

# ---------------------------------------------------------------------
# Run dataset creation
# ---------------------------------------------------------------------

cmd = `julia -O3 $dataset_script_copy
    --kind $(args["kind"])
    --size $(args["size"])
    --N $(args["num_csets"])
    --seed $(args["seed"])
    --batchsize $(args["batchsize"])
    --out $dataset_out
`
args["D"] !== nothing && (cmd = `$cmd --D $(args["D"])`)
args["cut_restriction"] !== nothing && (cmd = `$cmd --cut_restriction $(args["cut_restriction"])`)
run(cmd)

# Ensure dataset was written
isfile(dataset_out) || error("Dataset creation failed; output not found.")

# ---------------------------------------------------------------------
# Run statistics
# ---------------------------------------------------------------------

run(`julia -O3 $stats_script_copy
    --in $dataset_out
    --out $stats_out
    --num_processes $(args["num_processes"])
`)

isfile(stats_out) || error("Statistics computation failed; output not found.")

@info "Pipeline completed successfully"
