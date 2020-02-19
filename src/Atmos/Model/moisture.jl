export DryModel, EquilMoist, NonEquilMoist

#### Moisture component in atmosphere model
abstract type MoistureModel end

vars_state(::MoistureModel, FT) = @vars()
vars_gradient(::MoistureModel, FT) = @vars()
vars_diffusive(::MoistureModel, FT) = @vars()
vars_aux(::MoistureModel, FT) = @vars()

function atmos_nodal_update_aux!(::MoistureModel, m::AtmosModel, state::Vars,
                                 aux::Vars, t::Real)
end
function flux_moisture!(::MoistureModel, ::AtmosModel, flux::Grad, state::Vars, aux::Vars, t::Real)
end
function diffusive!(::MoistureModel, diffusive, ∇transform, state, aux, t)
end
function flux_diffusive!(::MoistureModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real, D_t)
end
function flux_nondiffusive!(::MoistureModel, flux::Grad, state::Vars, aux::Vars, t::Real)
end
function gradvariables!(::MoistureModel, ::AtmosModel, transform::Vars, state::Vars, aux::Vars, t::Real)
end

@inline function internal_energy(moist::MoistureModel, orientation::Orientation, state::Vars, aux::Vars)
  MoistThermodynamics.internal_energy(state.ρ, state.ρe, state.ρu, gravitational_potential(orientation, aux))
end
@inline temperature(moist::MoistureModel, orientation::Orientation, state::Vars, aux::Vars) = air_temperature(thermo_state(moist, orientation, state, aux))
@inline pressure(moist::MoistureModel, orientation::Orientation, state::Vars, aux::Vars) = air_pressure(thermo_state(moist, orientation, state, aux))
@inline soundspeed(moist::MoistureModel, orientation::Orientation, state::Vars, aux::Vars) = soundspeed_air(thermo_state(moist, orientation, state, aux))

@inline function total_specific_enthalpy(moist::MoistureModel, orientation::Orientation, state::Vars, aux::Vars)
  phase = thermo_state(moist, orientation, state, aux)
  R_m = gas_constant_air(phase)
  T = air_temperature(phase)
  e_tot = state.ρe * (1/state.ρ)
  e_tot + R_m*T
end



"""
    DryModel

Assumes the moisture components is in the dry limit.
"""
struct DryModel <: MoistureModel
end

vars_aux(::DryModel,FT) = @vars(θ_v::FT)
@inline function atmos_nodal_update_aux!(moist::DryModel, atmos::AtmosModel,
                                         state::Vars, aux::Vars, t::Real)
  e_int = internal_energy(moist, atmos.orientation, state, aux)
  TS = PhaseDry(e_int, state.ρ)
  aux.moisture.θ_v = virtual_pottemp(TS)
  nothing
end

thermo_state(moist::DryModel, orientation::Orientation, state::Vars, aux::Vars) = PhaseDry(internal_energy(moist, orientation, state, aux), state.ρ)

"""
    EquilMoist

Assumes the moisture components are computed via thermodynamic equilibrium.
"""
Base.@kwdef struct EquilMoist <: MoistureModel
  maxiter::Int = 3
end
vars_state(::EquilMoist,FT) = @vars(ρq_tot::FT)
vars_gradient(::EquilMoist,FT) = @vars(q_tot::FT, h_tot::FT)
vars_diffusive(::EquilMoist,FT) = @vars(∇q_tot::SVector{3,FT})
vars_aux(::EquilMoist,FT) = @vars(temperature::FT, θ_v::FT, q_liq::FT)

@inline function atmos_nodal_update_aux!(moist::EquilMoist, atmos::AtmosModel,
                                         state::Vars, aux::Vars, t::Real)
  FT=eltype(state)
  e_int = internal_energy(moist, atmos.orientation, state, aux)
  TS = PhaseEquil(e_int, state.ρ, state.moisture.ρq_tot/state.ρ, FT(1))
  aux.moisture.temperature = air_temperature(TS)
  aux.moisture.θ_v = virtual_pottemp(TS)
  aux.moisture.q_liq = PhasePartition(TS).liq
  nothing
end

function thermo_state(moist::EquilMoist, orientation::Orientation, state::Vars, aux::Vars)
  e_int = internal_energy(moist, orientation, state, aux)
  FT = eltype(state)
  return PhaseEquil{FT}(e_int, state.ρ, state.moisture.ρq_tot/state.ρ, aux.moisture.temperature)
end

function gradvariables!(moist::EquilMoist, atmos::AtmosModel, transform::Vars, state::Vars, aux::Vars, t::Real)
  ρinv = 1/state.ρ
  transform.moisture.q_tot = state.moisture.ρq_tot * ρinv
  phase = thermo_state(moist, atmos.orientation, state, aux)
  R_m = gas_constant_air(phase)
  T = air_temperature(phase)
  e_tot = state.ρe * ρinv
  transform.moisture.h_tot = e_tot + R_m*T
end

function diffusive!(moist::EquilMoist, diffusive::Vars, ∇transform::Grad, state::Vars, aux::Vars, t::Real)
  # diffusive flux of q_tot
  diffusive.moisture.∇q_tot = ∇transform.moisture.q_tot
end

function flux_moisture!(moist::EquilMoist, atmos::AtmosModel, flux::Grad, state::Vars, aux::Vars, t::Real)
  ρ = state.ρ
  u = state.ρu / ρ
  z = altitude(atmos.orientation, aux)
  usub = subsidence_velocity(atmos.subsidence, z)
  ẑ = vertical_unit_vector(atmos.orientation, aux)
  u_tot = u .- usub * ẑ
  flux.moisture.ρq_tot += u_tot * state.moisture.ρq_tot
end

function flux_diffusive!(moist::EquilMoist, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real, D_t)
  d_q_tot = (-D_t) .* diffusive.moisture.∇q_tot
  flux_diffusive!(moist, flux, state, d_q_tot)
end
#TODO: Consider whether to not pass ρ and ρu (not state), foc BCs reasons
function flux_diffusive!(moist::EquilMoist, flux::Grad, state::Vars, d_q_tot)
  flux.ρ += d_q_tot * state.ρ
  flux.ρu += d_q_tot .* state.ρu'
  flux.moisture.ρq_tot += d_q_tot * state.ρ
end

"""
    NonEquilMoist

Assumes the moisture components are computed via thermodynamic non-equilibrium.
"""
struct NonEquilMoist <: MoistureModel end
vars_state(::NonEquilMoist    ,FT) = @vars(ρq_tot::FT, ρq_liq::FT, ρq_ice::FT)
vars_gradient(::NonEquilMoist ,FT) = @vars(q_tot::FT,q_liq::FT, q_ice::FT, h_tot::FT)
vars_diffusive(::NonEquilMoist,FT) = @vars(ρd_q_tot::SVector{3,FT}, ρd_q_liq::SVector{3,FT}, ρd_q_ice::SVector{3,FT})
vars_aux(::NonEquilMoist      ,FT) = @vars(temperature::FT, θ_v::FT, q_liq::FT, q_ice::FT, p::FT)

@inline function atmos_nodal_update_aux!(moist::NonEquilMoist, atmos::AtmosModel,
                                         state::Vars, aux::Vars, t::Real)
  TS = thermo_state(moist, atmos.orientation, state, aux)
  aux.moisture.temperature = air_temperature(TS)
  aux.moisture.θ_v = virtual_pottemp(TS)
  aux.moisture.q_liq = PhasePartition(TS).liq
  aux.moisture.q_ice = PhasePartition(TS).ice
  aux.moisture.p = air_pressure(TS)
  return nothing
end

function thermo_state(moist::NonEquilMoist, orientation::Orientation, state::Vars, aux::Vars)
  FT = eltype(state)
  ρinv = 1/state.ρ
  e_int = internal_energy(moist, orientation, state, aux)
  q_pt = PhasePartition{FT}(state.moisture.ρq_tot*ρinv, state.moisture.ρq_liq*ρinv, state.moisture.ρq_ice*ρinv)
  return PhaseNonEquil{FT}(e_int, state.ρ, q_pt)
end

function gradvariables!(moist::NonEquilMoist, atmos::AtmosModel, transform::Vars, state::Vars, aux::Vars, t::Real)
  ρinv = 1/state.ρ
  transform.moisture.q_tot = state.moisture.ρq_tot * ρinv
  phase = thermo_state(moist, atmos.orientation, state, aux)
  R_m = gas_constant_air(phase)
  T = air_temperature(phase)
  e_tot = state.ρe * ρinv
  transform.moisture.h_tot = e_tot + R_m*T
  transform.moisture.q_liq = state.moisture.ρq_liq * ρinv
  transform.moisture.q_ice = state.moisture.ρq_ice * ρinv
end
function flux_moisture!(moist::NonEquilMoist, atmos::AtmosModel, flux::Grad, state::Vars, aux::Vars, t::Real)
  ρ = state.ρ
  u = state.ρu / ρ
  z = altitude(atmos.orientation, aux)
  usub = subsidence_velocity(atmos.subsidence, z)
  ẑ = vertical_unit_vector(atmos.orientation, aux)
  u_tot = u .- usub * ẑ
  flux.moisture.ρq_tot += u_tot * state.moisture.ρq_tot
end
function diffusive!(moist::NonEquilMoist, diffusive::Vars, ∇transform::Grad, state::Vars, aux::Vars, t::Real, )
  diffusive.moisture.ρd_q_tot =  ∇transform.moisture.q_tot
  diffusive.moisture.ρd_q_liq =  ∇transform.moisture.q_liq
  diffusive.moisture.ρd_q_ice =  ∇transform.moisture.q_ice
end

function flux_diffusive!(moist::NonEquilMoist, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real, D_t)
  d_q_tot = (-D_t) .* diffusive.moisture.ρd_q_tot
  d_q_liq = (-D_t) .* diffusive.moisture.ρd_q_liq
  d_q_ice = (-D_t) .* diffusive.moisture.ρd_q_ice
  u = state.ρu / state.ρ
  flux.ρ += d_q_tot * state.ρ
  flux.ρu += d_q_tot .* u' * state.ρ
  flux.moisture.ρq_tot += d_q_tot * state.ρ
  flux.moisture.ρq_liq += d_q_liq * state.ρ
  flux.moisture.ρq_ice += d_q_ice * state.ρ
end
