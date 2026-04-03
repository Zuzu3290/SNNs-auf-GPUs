! ============================================
! LIF Neuron Core - Fortran 2008 Implementation
! ============================================

module lif_neuron_module
    implicit none
    private
    public :: lif_neuron, neuron_layer, snn_network
    
    ! Define neuron parameters
    type :: lif_neuron_type
        real :: voltage
        real :: threshold
        real :: tau              ! Leak rate (0.0-1.0)
        real :: reset_voltage
        logical :: spiked
    end type lif_neuron_type
    
    ! Network structure
    type :: snn_network_type
        integer :: input_size
        integer :: hidden_size
        integer :: output_size
        integer :: timesteps
        
        ! Weights
        real, allocatable :: weights_input_hidden(:,:)
        real, allocatable :: weights_hidden_output(:,:)
        real, allocatable :: bias_hidden(:)
        real, allocatable :: bias_output(:)
        
        ! Neurons
        type(lif_neuron_type), allocatable :: hidden_neurons(:)
        type(lif_neuron_type), allocatable :: output_neurons(:)
        
        ! Activity
        integer, allocatable :: hidden_spikes(:,:)
        integer, allocatable :: output_spikes(:,:)
        real, allocatable :: hidden_voltages(:,:)
        real, allocatable :: output_voltages(:,:)
    end type snn_network_type

contains

! ============================================
! LIF Neuron Update
! ============================================
subroutine lif_neuron(neuron, input_current, dt)
    type(lif_neuron_type), intent(inout) :: neuron
    real, intent(in) :: input_current
    real, intent(in), optional :: dt
    
    real :: decay_rate, dt_local
    
    ! Default timestep
    dt_local = 1.0
    if (present(dt)) dt_local = dt
    
    ! LIF dynamics: V(t) = tau * V(t-1) + (1-tau) * I(t)
    decay_rate = neuron%tau
    neuron%voltage = decay_rate * neuron%voltage + &
                     (1.0 - decay_rate) * input_current
    
    ! Check for spike
    neuron%spiked = .false.
    if (neuron%voltage > neuron%threshold) then
        neuron%spiked = .true.
        neuron%voltage = neuron%reset_voltage
    end if
    
end subroutine lif_neuron

! ============================================
! Initialize SNN Network
! ============================================
subroutine initialize_network(network, input_size, hidden_size, output_size, timesteps)
    type(snn_network_type), intent(out) :: network
    integer, intent(in) :: input_size, hidden_size, output_size, timesteps
    
    real :: sigma
    integer :: i, j
    
    ! Set dimensions
    network%input_size = input_size
    network%hidden_size = hidden_size
    network%output_size = output_size
    network%timesteps = timesteps
    
    ! Allocate weight matrices
    allocate(network%weights_input_hidden(input_size, hidden_size))
    allocate(network%weights_hidden_output(hidden_size, output_size))
    allocate(network%bias_hidden(hidden_size))
    allocate(network%bias_output(output_size))
    
    ! Allocate neurons
    allocate(network%hidden_neurons(hidden_size))
    allocate(network%output_neurons(output_size))
    
    ! Allocate spike and voltage records
    allocate(network%hidden_spikes(hidden_size, timesteps))
    allocate(network%output_spikes(output_size, timesteps))
    allocate(network%hidden_voltages(hidden_size, timesteps))
    allocate(network%output_voltages(output_size, timesteps))
    
    ! Initialize weights (small random values)
    sigma = 0.01
    
    call random_number(network%weights_input_hidden)
    network%weights_input_hidden = sigma * (network%weights_input_hidden - 0.5)
    
    call random_number(network%weights_hidden_output)
    network%weights_hidden_output = sigma * (network%weights_hidden_output - 0.5)
    
    ! Initialize biases
    network%bias_hidden = 0.01
    network%bias_output = 0.01
    
    ! Initialize neuron parameters
    do i = 1, hidden_size
        network%hidden_neurons(i)%voltage = 0.0
        network%hidden_neurons(i)%threshold = 1.0
        network%hidden_neurons(i)%tau = 0.25
        network%hidden_neurons(i)%reset_voltage = 0.0
        network%hidden_neurons(i)%spiked = .false.
    end do
    
    do i = 1, output_size
        network%output_neurons(i)%voltage = 0.0
        network%output_neurons(i)%threshold = 1.0
        network%output_neurons(i)%tau = 0.25
        network%output_neurons(i)%reset_voltage = 0.0
        network%output_neurons(i)%spiked = .false.
    end do
    
    ! Initialize records
    network%hidden_spikes = 0
    network%output_spikes = 0
    network%hidden_voltages = 0.0
    network%output_voltages = 0.0
    
    print *, "✓ Network initialized"
    print *, "  Input neurons: ", input_size
    print *, "  Hidden neurons: ", hidden_size
    print *, "  Output neurons: ", output_size
    print *, "  Timesteps: ", timesteps
    
end subroutine initialize_network

! ============================================
! Forward Pass
! ============================================
subroutine forward_pass(network, input_spikes)
    type(snn_network_type), intent(inout) :: network
    integer, intent(in) :: input_spikes(:)
    
    integer :: t, i, j, n_input, n_hidden, n_output
    real :: current
    real, allocatable :: hidden_currents(:), output_currents(:)
    
    n_input = network%input_size
    n_hidden = network%hidden_size
    n_output = network%output_size
    
    allocate(hidden_currents(n_hidden))
    allocate(output_currents(n_output))
    
    ! Loop through timesteps
    do t = 1, network%timesteps
        
        ! ===== Stage 1: Compute hidden layer currents =====
        hidden_currents = network%bias_hidden
        
        do i = 1, n_hidden
            do j = 1, n_input
                if (input_spikes(j) == 1) then
                    hidden_currents(i) = hidden_currents(i) + &
                                       network%weights_input_hidden(j, i)
                end if
            end do
        end do
        
        ! ===== Stage 2: Update hidden neurons =====
        do i = 1, n_hidden
            call lif_neuron(network%hidden_neurons(i), hidden_currents(i))
            
            ! Record spike and voltage
            if (network%hidden_neurons(i)%spiked) then
                network%hidden_spikes(i, t) = 1
            else
                network%hidden_spikes(i, t) = 0
            end if
            network%hidden_voltages(i, t) = network%hidden_neurons(i)%voltage
        end do
        
        ! ===== Stage 3: Compute output layer currents =====
        output_currents = network%bias_output
        
        do i = 1, n_output
            do j = 1, n_hidden
                if (network%hidden_spikes(j, t) == 1) then
                    output_currents(i) = output_currents(i) + &
                                       network%weights_hidden_output(j, i)
                end if
            end do
        end do
        
        ! ===== Stage 4: Update output neurons =====
        do i = 1, n_output
            call lif_neuron(network%output_neurons(i), output_currents(i))
            
            ! Record spike and voltage
            if (network%output_neurons(i)%spiked) then
                network%output_spikes(i, t) = 1
            else
                network%output_spikes(i, t) = 0
            end if
            network%output_voltages(i, t) = network%output_neurons(i)%voltage
        end do
        
    end do
    
    deallocate(hidden_currents, output_currents)
    
end subroutine forward_pass

! ============================================
! Predict Class from Output Spikes
! ============================================
function predict_class(network) result(predicted_class)
    type(snn_network_type), intent(in) :: network
    integer :: predicted_class
    
    integer :: i, max_spikes
    integer :: spike_counts(network%output_size)
    
    ! Count spikes for each output neuron
    spike_counts = sum(network%output_spikes, dim=2)
    
    ! Find neuron with maximum spikes
    predicted_class = maxloc(spike_counts, dim=1)
    
end function predict_class

! ============================================
! Print Network Statistics
! ============================================
subroutine print_statistics(network)
    type(snn_network_type), intent(in) :: network
    
    integer :: i, j
    integer :: total_hidden_spikes, total_output_spikes
    real :: avg_hidden_voltage, avg_output_voltage
    
    total_hidden_spikes = sum(network%hidden_spikes)
    total_output_spikes = sum(network%output_spikes)
    avg_hidden_voltage = sum(network%hidden_voltages) / &
                        real(size(network%hidden_voltages))
    avg_output_voltage = sum(network%output_voltages) / &
                        real(size(network%output_voltages))
    
    print *, ""
    print *, "📊 Network Statistics"
    print *, "===================="
    print *, "Total hidden spikes: ", total_hidden_spikes
    print *, "Total output spikes: ", total_output_spikes
    print *, "Avg hidden voltage: ", avg_hidden_voltage
    print *, "Avg output voltage: ", avg_output_voltage
    print *, "Hidden neuron firing rate: ", &
            real(total_hidden_spikes) / (network%hidden_size * network%timesteps) * 100, "%"
    print *, "Output neuron firing rate: ", &
            real(total_output_spikes) / (network%output_size * network%timesteps) * 100, "%"
    
end subroutine print_statistics

! ============================================
! Cleanup
! ============================================
subroutine cleanup_network(network)
    type(snn_network_type), intent(inout) :: network
    
    deallocate(network%weights_input_hidden)
    deallocate(network%weights_hidden_output)
    deallocate(network%bias_hidden)
    deallocate(network%bias_output)
    deallocate(network%hidden_neurons)
    deallocate(network%output_neurons)
    deallocate(network%hidden_spikes)
    deallocate(network%output_spikes)
    deallocate(network%hidden_voltages)
    deallocate(network%output_voltages)
    
    print *, "✓ Network cleaned up"
    
end subroutine cleanup_network

end module lif_neuron_module

! ============================================
! MAIN PROGRAM
! ============================================
program snn_fpga_simulation
    use lif_neuron_module
    implicit none
    
    type(snn_network_type) :: network
    integer, allocatable :: input_spikes(:)
    integer :: i, j, predicted_class
    real :: start_time, end_time
    
    ! Parameters
    integer, parameter :: INPUT_SIZE = 784
    integer, parameter :: HIDDEN_SIZE = 256
    integer, parameter :: OUTPUT_SIZE = 10
    integer, parameter :: TIMESTEPS = 100
    integer, parameter :: NUM_SAMPLES = 100
    
    print *, "🧠 FPGA Neuromorphic SNN - Fortran Simulation"
    print *, "=============================================="
    print *, ""
    
    ! Initialize network
    call initialize_network(network, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, TIMESTEPS)
    print *, ""
    
    ! Allocate input spikes
    allocate(input_spikes(INPUT_SIZE))
    
    ! Benchmark
    print *, "⚡ Running benchmark with ", NUM_SAMPLES, " samples"
    print *, ""
    
    call cpu_time(start_time)
    
    ! Process multiple samples
    do i = 1, NUM_SAMPLES
        ! Generate random input spikes
        call random_number(input_spikes)
        input_spikes = merge(1, 0, input_spikes < 0.3)
        
        ! Forward pass
        call forward_pass(network, input_spikes)
        
        ! Predict class
        predicted_class = predict_class(network)
        
        if (mod(i, 10) == 0) then
            print *, "Sample ", i, " - Predicted class: ", predicted_class
        end if
    end do
    
    call cpu_time(end_time)
    
    print *, ""
    print *, "⏱️  Timing:"
    print *, "Total time: ", end_time - start_time, " seconds"
    print *, "Time per sample: ", (end_time - start_time) / NUM_SAMPLES * 1000, " ms"
    print *, "Throughput: ", NUM_SAMPLES / (end_time - start_time), " samples/sec"
    
    ! Print final statistics
    call print_statistics(network)
    
    ! Cleanup
    deallocate(input_spikes)
    call cleanup_network(network)
    
    print *, ""
    print *, "✅ Simulation complete!"
    
end program snn_fpga_simulation