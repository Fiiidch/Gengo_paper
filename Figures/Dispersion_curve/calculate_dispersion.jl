# Dispersion Curve Calculation Script
# This script calculates dispersion curves along x and y axes for the plate object

# Include necessary packages
using HDF5
using FFTW
using LinearAlgebra
using Plots

# Parameters
data_freq = 625_000   # Sampling frequency in Hz
n_t = Int(6250/5)     # Number of time steps per measurement
num_noise = 5         # Number of repeated measurements in time axis

# User selection - which source and measurement to use
selected_source = 5        # Source number (1-5)
selected_measurement = 1   # Measurement number (1-5)

# Directory paths
raw_data_dir = "Real_data/"
defect_dir = "Real_data/defect/"

# Read sensor grid positions from one of the h5 files
println("Reading sensor grid positions...")
h5file = h5open(raw_data_dir * "src_01_full_measurement.h5", "r")
xyz_raw = read(h5file, "XYZ")
close(h5file)

# Convert to vector format for easier distance calculations
xyz_vectors = [xyz_raw[:, i] for i in 1:size(xyz_raw, 2)]

println("Grid dimensions: ", size(xyz_raw))
println("Number of sensors: ", length(xyz_vectors))

# Read source positions from defect directory
println("\nReading source positions...")
h5file = h5open(defect_dir * "src_01_defect.h5", "r")
sources_raw = read(h5file, "XYZ")
close(h5file)

# Convert sources to vector format
sources = [sources_raw[:, i] for i in 1:size(sources_raw, 2)]

println("Number of sources: ", length(sources))
println("Source positions:")
for (i, src) in enumerate(sources)
    println("  Source $i: ", round.(src, digits=5))
end

# Find sensor indices corresponding to source positions
println("\nMatching sources to sensor grid...")
sensor_indices = Int[]
sensor_locations = []

for (i, src) in enumerate(sources[1:5])
    # Compute distances to all xyz points
    dists = [norm(x - src) for x in xyz_vectors]
    idx = argmin(dists)
    dist_vec = xyz_vectors[idx] .- src
    println("Source $i matched to sensor index $idx")
    println("  Position: ", round.(xyz_vectors[idx], digits=5))
    println("  Distance from source: ", round.(dist_vec, digits=5))
    
    push!(sensor_indices, idx)
    push!(sensor_locations, xyz_vectors[idx])
end

# Define h5 file paths for all 5 sources
h5_files = [
    raw_data_dir * "src_01_full_measurement.h5",
    raw_data_dir * "src_02_full_measurement.h5",
    raw_data_dir * "src_03_full_measurement.h5",
    raw_data_dir * "src_04_full_measurement.h5",
    raw_data_dir * "src_05_full_measurement.h5"
]

println("\nH5 files defined:")
for (i, file) in enumerate(h5_files)
    println("  Source $i: $file")
end

println("\n" * "="^60)
println("Setup complete!")
println("Sensor indices: ", sensor_indices)
println("="^60)

"""
    load_measurement_data(source::Int)

Load vibration data from a specific source, averaging all 5 measurements for noise reduction.

# Arguments
- `source::Int`: Source number (1-5)

# Returns
- `Matrix{Float64}`: Averaged data matrix of shape (n_t × num_sensors)
"""
function load_measurement_data(source::Int)
    # Validate inputs
    if !(1 <= source <= 5)
        error("Source must be between 1 and 5, got $source")
    end
    
    println("\nLoading data from source $source (averaging all 5 measurements)...")
    
    # Open the h5 file for the selected source
    h5file = h5open(h5_files[source], "r")
    key = "vib_z"
    measurements = read(h5file, key)[:,:]
    close(h5file)
    
    # Ensure data has correct length (pad or truncate if needed)
    if size(measurements, 1) < 6250
        pad_size = 6250 - size(measurements, 1)
        measurements = vcat(measurements, zeros(pad_size, size(measurements, 2)))
    elseif size(measurements, 1) > 6250
        measurements = measurements[1:6250, :]
    end
    
    # Average all 5 measurements
    averaged_data = zeros(Float64, n_t, size(measurements, 2))
    
    for n in 1:num_noise
        measurement_range = 1 + (n - 1) * n_t : n * n_t
        averaged_data .+= measurements[measurement_range, :]
    end
    
    averaged_data ./= num_noise
    
    println("  Data shape: ", size(averaged_data))
    println("  Time steps: ", size(averaged_data, 1), " samples")
    println("  Sensors: ", size(averaged_data, 2))
    println("  Averaged over $num_noise measurements")
    
    return averaged_data
end

"""
    extract_axis_slice(source::Int, axis::String; n_points::Int=104, time_range=nothing, space_range=nothing)

Extract a slice of data along x or y axis through the specified source.

# Arguments
- `source::Int`: Source number (1-5)
- `axis::String`: Either "x" or "y" to specify the slice direction
- `n_points::Int`: Number of points to extract along the slice (default: 104)
- `time_range`: Tuple (t_min, t_max) in ms to extract subset of time (default: nothing = full range)
- `space_range`: Tuple (s_min, s_max) in m to extract subset of space (default: nothing = full range)

# Returns
- `Matrix{Float64}`: Data matrix of shape (n_t × n_points) containing time series along the slice
- `Vector{Float64}`: Positions along the slice axis
- `Vector{Int}`: Sensor indices used for the slice
"""
function extract_axis_slice(source::Int, axis::String; n_points::Int=104, time_range=nothing, space_range=nothing)
    # Validate axis input
    if !(axis in ["x", "y"])
        error("Axis must be either 'x' or 'y', got '$axis'")
    end
    
    # Load the full measurement data (averaged over all 5 measurements)
    data = load_measurement_data(source)
    
    # Get source location
    source_pos = sensor_locations[source]
    println("\nExtracting $axis-axis slice through source $source at position: ", round.(source_pos, digits=5))
    
    # Determine which coordinate to vary (x=1, y=2) and which to keep constant (z=3 is always constant)
    vary_idx = axis == "x" ? 1 : 2
    const_idx = axis == "x" ? 2 : 1
    
    # Find all sensors that are close to the line through the source
    # The line varies along vary_idx, keeps const_idx constant
    const_value = source_pos[const_idx]
    z_value = source_pos[3]
    
    # Filter sensors that are close to the desired line
    # Allow some tolerance for imperfect grid
    tolerance = 0.003  #  tolerance
    
    candidate_indices = Int[]
    candidate_positions = Float64[]
    
    for (idx, pos) in enumerate(xyz_vectors)
        # Check if this sensor is close to our line (same const_idx value and z value)
        if abs(pos[const_idx] - const_value) < tolerance && abs(pos[3] - z_value) < tolerance
            push!(candidate_indices, idx)
            push!(candidate_positions, pos[vary_idx])  # Store the varying coordinate
        end
    end
    
    println("Found $(length(candidate_indices)) sensors along the $axis-axis within tolerance")
    
    # Sort by position along the varying axis
    sort_order = sortperm(candidate_positions)
    sorted_indices = candidate_indices[sort_order]
    sorted_positions = candidate_positions[sort_order]
    
    # Apply space range filter if specified
    if !isnothing(space_range)
        space_mask = (sorted_positions .>= space_range[1]) .& (sorted_positions .<= space_range[2])
        sorted_indices = sorted_indices[space_mask]
        sorted_positions = sorted_positions[space_mask]
        println("Applied space range filter: $(space_range[1]) to $(space_range[2]) m")
        println("Remaining points after space filter: $(length(sorted_indices))")
    end
    
    # If we have more points than needed, subsample evenly
    if length(sorted_indices) > n_points
        step = length(sorted_indices) / n_points
        selected = [sorted_indices[round(Int, 1 + (i-1)*step)] for i in 1:n_points]
        selected_pos = [sorted_positions[round(Int, 1 + (i-1)*step)] for i in 1:n_points]
    else
        selected = sorted_indices
        selected_pos = sorted_positions
    end
    
    println("Selected $(length(selected)) points for dispersion analysis")
    println("Position range: $(round(minimum(selected_pos), digits=5)) to $(round(maximum(selected_pos), digits=5))")
    
    # Extract the data for these sensors
    slice_data = data[:, selected]
    
    # Apply time range filter if specified
    if !isnothing(time_range)
        # Convert time in ms to sample indices
        dt = 2.0 / n_t  # Time step in ms
        t_start_idx = max(1, round(Int, time_range[1] / dt) + 1)
        t_end_idx = min(n_t, round(Int, time_range[2] / dt) + 1)
        slice_data = slice_data[t_start_idx:t_end_idx, :]
        println("Applied time range filter: $(time_range[1]) to $(time_range[2]) ms")
        println("Time indices: $t_start_idx to $t_end_idx")
    end
    
    return slice_data, selected_pos, selected
end

"""
    plot_space_time_slice(source::Int, axis::String; n_points::Int=104, time_range=nothing, space_range=nothing, freq_range=nothing)

Plot the space-time slice showing wave propagation along x or y axis, and compute the dispersion curve.

# Arguments
- `source::Int`: Source number (1-5)
- `axis::String`: Either "x" or "y" to specify the slice direction
- `n_points::Int`: Number of points to extract along the slice (default: 104)
- `time_range`: Tuple (t_min, t_max) in ms to plot subset of time (default: nothing = (0, 2))
- `space_range`: Tuple (s_min, s_max) in m to plot subset of space (default: nothing = (0, 0.25))
- `freq_range`: Tuple (f_min, f_max) in kHz to limit frequency range in plots (default: nothing = auto)

# Returns
- Combined plot with space-time slice and dispersion curve
"""
function plot_space_time_slice(source::Int, axis::String; n_points::Int=104, time_range=nothing, space_range=nothing, freq_range=nothing)
    # Extract the slice data (averaged over all 5 measurements)
    slice_data, positions, indices = extract_axis_slice(source, axis, n_points=n_points, time_range=time_range, space_range=space_range)
    
    # Determine actual ranges for axes
    actual_time_range = isnothing(time_range) ? (0, 2) : time_range
    actual_space_range = isnothing(space_range) ? (0, 0.25) : space_range
    
    # Create space and time axes
    space_axis = range(actual_space_range[1], actual_space_range[2], length=size(slice_data, 2))
    time_axis = range(actual_time_range[1], actual_time_range[2], length=size(slice_data, 1))
    
    # Create the space-time heatmap
    plt1 = heatmap(space_axis, time_axis, slice_data,
        xlabel="Position along $(uppercase(axis))-axis [m]",
        ylabel="Time [ms]",
        title="Space-Time Plot CFRP 1.3mm",
        colorbar_title="\n Unitless Velocity",
        aspect_ratio=:auto,
        c=:seismic,
        right_margin=5Plots.mm)
    
    # Compute 2D FFT for dispersion curve
    fft_2d = fftshift(fft(slice_data))
    fft_magnitude = abs.(fft_2d)
    
    # Create frequency and wavenumber axes
    dt = (actual_time_range[2] - actual_time_range[1]) / (size(slice_data, 1) - 1) * 1e-3  # Convert ms to s
    dx = (actual_space_range[2] - actual_space_range[1]) / (size(slice_data, 2) - 1)
    
    freq_axis = fftshift(fftfreq(size(slice_data, 1), 1/dt)) / 1000  # Convert to kHz
    k_axis = fftshift(fftfreq(size(slice_data, 2), 1/dx)) / 1000     # Convert to rad/mm
    
    # Determine frequency range for plots
    actual_freq_range = isnothing(freq_range) ? (0, maximum(freq_axis)) : (freq_range[1], freq_range[2])
    
    # Extract dispersion curve: phase velocity vs frequency
    # For each frequency, find the wavenumber with maximum energy
    freq_positive_idx = freq_axis .>= 0
    freq_positive = freq_axis[freq_positive_idx]
    fft_positive_freq = fft_magnitude[freq_positive_idx, :]
    
    # Find peak wavenumber for each frequency
    phase_velocities = Float64[]
    wavelengths = Float64[]
    frequencies_for_velocity = Float64[]
    
    for i in 1:length(freq_positive)
        # Apply frequency range filter
        in_range = isnothing(freq_range) ? true : (freq_positive[i] >= freq_range[1] && freq_positive[i] <= freq_range[2])
        
        if freq_positive[i] > 0.1 && in_range  # Skip very low frequencies (noise) and apply range
            # Find peak in wavenumber for this frequency
            max_idx = argmax(fft_positive_freq[i, :])
            k_peak = abs(k_axis[max_idx])
            
            if k_peak > 0.01  # Skip near-zero wavenumbers
                # Phase velocity = ω/k = 2πf/k
                omega = 2π * freq_positive[i] * 1000  # Convert kHz to Hz
                k_actual = k_peak * 1000  # Convert to rad/m
                v_phase = omega / k_actual  # m/s
                
                # Wavelength λ = 2π/k = v/f
                lambda = 2π / k_actual * 1000  # Convert to mm
                
                push!(frequencies_for_velocity, freq_positive[i])
                push!(phase_velocities, v_phase)
                push!(wavelengths, lambda)
            end
        end
    end
    
    # Create combined dispersion plot with dual y-axes
    # First plot: Phase velocity (left y-axis) with both series for combined legend
    plt2 = plot(frequencies_for_velocity, phase_velocities,
                xlabel="Frequency [kHz]",
                ylabel="Phase Velocity [m/s]",
                label="Phase Velocity",
                linewidth=1.5,
                linestyle=:dash,
                color=:blue,
                marker=:circle,
                markersize=4,
                markerstrokewidth=0,
                xlims=actual_freq_range,
                grid=true)
    
    # Add invisible dummy series for wavelength to appear in same legend
    plot!(plt2, Float64[], Float64[],
          label="Wavelength",
          linewidth=1.5,
          linestyle=:dash,
          color=:red,
          marker=:circle,
          markersize=4,
          legend=:right)
    
    # Add wavelength on secondary y-axis (no legend to avoid duplication)
    plot!(twinx(), frequencies_for_velocity, wavelengths,
          ylabel="Wavelength [mm]",
          label="",
          linewidth=1.5,
          linestyle=:dash,
          color=:red,
          marker=:circle,
          markersize=4,
          markerstrokewidth=0,
          xlims=actual_freq_range,
          legend=false,
          grid=true)
    
    # Combine plots (space-time and dual-axis dispersion)
    plt = plot(plt1, plt2, layout=(1, 2), size=(1400, 500))
    
    println("\nPlots created: Space-Time slice and Dispersion Curve (dual-axis) along $axis-axis")
    
    # Save the plots separately to Dispersion_curve folder
    filename1 = "Dispersion_curve/space_time_slice_src$(source)_$(axis)axis.png"
    savefig(plt1, filename1)
    println("Space-time plot saved to: $filename1")
    
    filename2 = "Dispersion_curve/dispersion_dual_axis_src$(source)_$(axis)axis.png"
    savefig(plt2, filename2)
    println("Dispersion curve (dual-axis) saved to: $filename2")
    
    return plt
end

# Example usage: Load averaged data from selected source
selected_data = load_measurement_data(selected_source)
println("\nData loaded successfully!")

# Example: Extract x-axis slice
slice_data_x, positions_x, indices_x = extract_axis_slice(selected_source, "x")
println("\nX-axis slice extracted!")
println("Slice data shape: ", size(slice_data_x))

# Example: Plot space-time slice with time range 0.08 to 0.17 ms and frequency range 1-150 kHz
plt_x = plot_space_time_slice(selected_source, "x", time_range=(0.08, 0.17), freq_range=(1, 150))
display(plt_x)


