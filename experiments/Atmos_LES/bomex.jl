#=
@article{doi:10.1175/1520-0469(2003)60<1201:ALESIS>2.0.CO;2,
author = {Siebesma, A. Pier and Bretherton, 
          Christopher S. and Brown, 
          Andrew and Chlond, 
          Andreas and Cuxart, 
          Joan and Duynkerke, 
          Peter G. and Jiang, 
          Hongli and Khairoutdinov, 
          Marat and Lewellen, 
          David and Moeng, 
          Chin-Hoh and Sanchez, 
          Enrique and Stevens, 
          Bjorn and Stevens, 
          David E.},
title = {A Large Eddy Simulation Intercomparison Study of Shallow Cumulus Convection},
journal = {Journal of the Atmospheric Sciences},
volume = {60},
number = {10},
pages = {1201-1219},
year = {2003},
doi = {10.1175/1520-0469(2003)60<1201:ALESIS>2.0.CO;2},
URL = {https://journals.ametsoc.org/doi/abs/10.1175/1520-0469%282003%2960%3C1201%3AALESIS%3E2.0.CO%3B2},
eprint = {https://journals.ametsoc.org/doi/pdf/10.1175/1520-0469%282003%2960%3C1201%3AALESIS%3E2.0.CO%3B2}
=# 

using Distributions
using Random
using StaticArrays
using Test
using DocStringExtensions
using LinearAlgebra

using CLIMA
using CLIMA.Atmos
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.GenericCallbacks
using CLIMA.Mesh.Filters
using CLIMA.MoistThermodynamics
using CLIMA.PlanetParameters
using CLIMA.VariableTemplates

import CLIMA.DGmethods: vars_state, vars_aux
import CLIMA.Atmos: atmos_source!, altitude
import CLIMA.DGmethods: boundary_state!
import CLIMA.Atmos: atmos_boundary_state!, atmos_boundary_flux_diffusive!, flux_diffusive!, thermo_state

# ---------------------------- Begin Boundary Conditions ----------------- #
"""
  BOMEX_BC <: BoundaryCondition
  Prescribes boundary conditions for Dynamics of Marine Stratocumulus Case
#Fields
$(DocStringExtensions.FIELDS)
"""
struct BOMEX_BC{FT} <: BoundaryCondition
  "Friction velocity"
  u_τ::FT  
  "Sensible Heat Flux"
  w′θ′::FT
  "Latent Heat Flux"
  w′qt′::FT
end

"""
    atmos_boundary_state!(nf::Union{NumericalFluxNonDiffusive, NumericalFluxGradient},
                          bc::BOMEX_BC, args...)

For the non-diffussive and gradient terms we just use the `NoFluxBC`
"""
atmos_boundary_state!(nf::Union{NumericalFluxNonDiffusive, NumericalFluxGradient},
                      bc::BOMEX_BC, 
                      args...) = atmos_boundary_state!(nf, NoFluxBC(), args...)

"""
    atmos_boundary_flux_diffusive!(nf::NumericalFluxDiffusive,
                                   bc::BOMEX_BC, atmos::AtmosModel,
                                   F,
                                   state⁺, diff⁺, aux⁺, n⁻,
                                   state⁻, diff⁻, aux⁻,
                                   bctype, t,
                                   state1⁻, diff1⁻, aux1⁻)

When `bctype == 1` the `NoFluxBC` otherwise the specialized BOMEX BC is used
"""
#TODO This needs to be in sync with the new boundary condition interfaces
function atmos_boundary_flux_diffusive!(nf::CentralNumericalFluxDiffusive,
                                        bc::BOMEX_BC, 
                                        atmos::AtmosModel, F,
                                        state⁺, diff⁺, hyperdiff⁺, aux⁺,
                                        n⁻,
                                        state⁻, diff⁻, hyperdiff⁻, aux⁻,
                                        bctype, t,
                                        state1⁻, diff1⁻, aux1⁻)
  
  FT = eltype(state⁺)
  TS = thermo_state(atmos.moisture, atmos.orientation, state⁺, aux⁺)
  
  if bctype != 1
    atmos_boundary_flux_diffusive!(nf, NoFluxBC(), atmos, F,
                                   state⁺, diff⁺, hyperdiff⁺, aux⁺, n⁻,
                                   state⁻, diff⁻, hyperdiff⁻, aux⁻,
                                   bctype, t,
                                   state1⁻, diff1⁻, aux1⁻)
  else
    # Start with the noflux BC and then build custom flux from there
    atmos_boundary_state!(nf, NoFluxBC(), atmos,
                          state⁺, diff⁺, aux⁺, n⁻,
                          state⁻, diff⁻, aux⁻,
                          bctype, t)
    
    u₀ = state⁻.ρu / state⁻.ρ
    windspeed₀ = norm(u₀)
    
    _, τ⁻ = turbulence_tensors(atmos.turbulence, state⁻, diff⁻, aux⁻, t)
    
    u_τ = bc.u_τ
    
    @inbounds begin
      τ13⁺ = - u_τ^2 * windspeed₀[1] / norm(windspeed₀)
      τ23⁺ = - u_τ^2 * windspeed₀[2] / norm(windspeed₀)
      τ21⁺ = τ⁻[2,1]
    end
    
    FT = eltype(state⁺)
    τ⁺ = SHermitianCompact{3, FT, 6}(SVector(0   ,
                                             τ21⁺, τ13⁺,
                                             0   , τ23⁺, 0))
    
    d_q_tot⁺  = SVector(0, 
                        0, 
                        state⁺.ρ * bc.w′qt′)
    

    d_h_tot⁺ = SVector(0, 
                       0, 
                       (bc.w′θ′ * cp_m(TS) * state⁺.ρ + state⁺.ρ * bc.w′qt′ * LH_v0))
  
    # Set the flux using the now defined plus-side data
    flux_diffusive!(atmos, F, state⁺, τ⁺, d_h_tot⁺)
    flux_diffusive!(atmos.moisture, F, state⁺, d_q_tot⁺)
  end
end
# ------------------------ End Boundary Condition --------------------- # 


"""
  Bomex Sources
"""
struct BomexGeostrophic{FT} <: Source
  "Coriolis forcing coefficient [s⁻¹]"
  f_coriolis::FT
  "Eastward geostrophic velocity `[m/s]` (Base)"
  u_geostrophic::FT
  "Eastward geostrophic velocity `[m/s]` (Slope)"
  u_slope::FT
  "Northward geostrophic velocity `[m/s]`"
  v_geostrophic::FT
end
function atmos_source!(s::BomexGeostrophic, atmos::AtmosModel, source::Vars, state::Vars, diffusive::Vars, aux::Vars, t::Real)

  f_coriolis    = s.f_coriolis
  u_geostrophic = s.u_geostrophic
  u_slope       = s.u_slope
  v_geostrophic = s.v_geostrophic

  z          = altitude(atmos.orientation,aux)
  # Note z dependence of eastward geostrophic velocity
  u_geo      = SVector(u_geostrophic + u_slope * z, v_geostrophic, 0)
  ẑ          = vertical_unit_vector(atmos.orientation, aux)
  fkvector   = f_coriolis * ẑ
  # Accumulate sources
  source.ρu -= fkvector × (state.ρu .- state.ρ*u_geo)
end

struct BomexSponge{FT} <: Source
  "Maximum domain altitude (m)"
  z_max::FT
  "Altitude at with sponge starts (m)"
  z_sponge::FT
  "Sponge Strength 0 ⩽ α_max ⩽ 1"
  α_max::FT
  "Sponge exponent"
  γ::FT
  "Eastward geostrophic velocity `[m/s]` (Base)"
  u_geostrophic::FT
  "Eastward geostrophic velocity `[m/s]` (Slope)"
  u_slope::FT
  "Northward geostrophic velocity `[m/s]`"
  v_geostrophic::FT
end
function atmos_source!(s::BomexSponge, atmos::AtmosModel, source::Vars, state::Vars, diffusive::Vars, aux::Vars, t::Real)

  z_max = s.z_max
  z_sponge = s.z_sponge
  α_max = s.α_max
  γ = s.γ
  u_geostrophic = s.u_geostrophic
  u_slope       = s.u_slope
  v_geostrophic = s.v_geostrophic

  z          = altitude(atmos.orientation,aux)
  u_geo      = SVector(u_geostrophic + u_slope * z, v_geostrophic, 0)
  ẑ          = vertical_unit_vector(atmos.orientation, aux)
  # Accumulate sources
  if z_sponge <= z
    r = (z - z_sponge)/(z_max-z_sponge)
    β_sponge = α_max * sinpi(r/2)^s.γ
    source.ρu -= β_sponge * (state.ρu .- state.ρ * u_geo)
  end
end

struct BomexMoistureTendency{FT} <: Source
 "Advection tendency in total moisture `[s⁻¹]`"
  ∂qt∂t::FT
  "Lower extent of piecewise profile `[m]`"
  z_l::FT   
  "Upper extent of piecewise profile `[m]`"
  z_h::FT
end
function atmos_source!(s::BomexMoistureTendency, atmos::AtmosModel, source::Vars, state::Vars, diffusive::Vars, aux::Vars, t::Real)
  FT = eltype(state)
  
  z_l = s.z_l
  z_h = s.z_h
  ∂qt∂t = s.∂qt∂t
  
  ρ     = state.ρ
  z     = altitude(atmos.orientation,aux)
  # Piecewise profile for advective moisture forcing
  if z <= z_l
    source.moisture.ρq_tot += ρ * ∂qt∂t
  elseif z_l < z <= z_h
    source.moisture.ρq_tot += ρ * (∂qt∂t-∂qt∂t*(z-z_l)/(z_h-z_l))
  end
end

struct BomexLargeScaleSubsidence{FT} <: Source
 "Subsidence velocity `[m/s]`" 
  w_sub::FT
  "Lower extent of piecewise profile `[m]`"
  z_l::FT
  "Upper extent of piecewise profile `[m]`"
  z_h::FT
end
function atmos_source!(s::BomexLargeScaleSubsidence, atmos::AtmosModel, source::Vars, state::Vars, diffusive::Vars, aux::Vars, t::Real)
  FT = eltype(state)
  
  w_sub = s.w_sub
  z_l = s.z_l
  z_h = s.z_h
  
  #Set large scale subsidence
  z = altitude(atmos.orientation,aux)
  wₛ = FT(0)
  if z <= z_l
    wₛ = FT(0) + z*(w_sub)/(z_l)
  elseif z_l < z <= z_h
    wₛ = w_sub - (z - z_l)* (w_sub)/(z_h - z_l)
  end
  ρ = state.ρ
  k̂ = vertical_unit_vector(atmos.orientation, aux)
  source.ρe -= ρ * wₛ * dot(k̂, diffusive.∇h_tot)
  source.moisture.ρq_tot -= ρ * wₛ * dot(k̂, diffusive.moisture.∇q_tot)
end

struct BomexTemperatureTendency{FT} <: Source
  "Cooling rate `[K/s]`"
  Qᵣ::FT
  "Lower extent of piecewise profile `[m]`"
  z_l::FT
  "Upper extent of piecewise profile `[m]`"
  z_h::FT
end
function atmos_source!(s::BomexTemperatureTendency, atmos::AtmosModel, source::Vars, state::Vars, diffusive::Vars, aux::Vars, t::Real)
  FT    = eltype(state)
 
  Qᵣ    = s.Qᵣ
  z_l   = s.z_l
  z_h   = s.z_h
  
  z     = altitude(atmos.orientation,aux)
  TS    = thermo_state(atmos.moisture, atmos.orientation, state, aux)
  T     = air_temperature(TS)
  q_pt  = PhasePartition(TS)
  Qₑ    = internal_energy(T_0-Qᵣ, q_pt)
  if z <= z_l 
    source.ρe += state.ρ * Qₑ
  elseif z_l < z <= z_h
    source.ρe += state.ρ * (Qₑ - Qₑ * ((z-z_l) / (z_h-z_l)))
  end
end

"""
  Initial Condition for BOMEX LES
"""
seed = MersenneTwister(0)
function init_bomex!(bl, state, aux, (x,y,z), t)

  FT      = eltype(state)
  # Observed ground quantities
  pg::FT  = 1.015e6
  u::FT   = 0
  v::FT   = 0
  w::FT   = 0 
  # Prescribed heights for piece-wise profile construction
  
  zl1::FT = 520
  zl2::FT = 1480
  zl3::FT = 2000
  zl4::FT = 3000
  # Assign piecewise quantities to θ_liq and q_tot 
  
  θ_liq::FT = 0 
  q_tot::FT = 0 

  if FT(0) <= z <= zl1
    θ_liq = 298.7
    q_tot = 17.0 + (z/zl1)*(16.3-17.0)
  elseif z > zl1 && z <= zl2
    θ_liq = 298.7 + (z-zl1) * (302.4-298.7)/(zl2-zl1)
    q_tot = 16.3 + (z-zl1) * (10.7-16.3)/(zl2-zl1)
  elseif z > zl2 && z <= zl3
    θ_liq = 302.4 + (z-zl2) * (308.2-302.4)/(zl3-zl2)
    q_tot = 10.7 + (z-zl2) * (4.2-10.7)/(zl3-zl2)
  else 
    θ_liq = 308.2 + (z-zl3) * (311.85-308.2)/(zl4-zl3)
    q_tot = 4.2 + (z-zl3) * (3.0-4.2)/(zl4-zl3)
  end
  
  # Set velocity profiles - piecewise profile for u
  zlv::FT = 700
  
  if FT(0) < z <= zlv
    u = -8.75
  else
    u = -8.75 + (z - zlv) * (-4.61 + 8.75)/(zl4 - zlv)
  end
  
  q_tot *= 1e-3 # Convert total specific humidity to kg/kg
  
  Γ = grav / cv_d
  T_min = FT(255)
  T_surface = FT(299)

  T = max(T_surface - Γ*z, T_min)
  p = MSLP * (T/T_surface)^(grav/(R_d*Γ))
  
  if T == T_min
    z_top = (T_surface - T_min) / Γ
    H_min = R_d * T_min / grav
    p *= exp(-(z-z_top)/H_min)
  end

  ρ = air_density(T,p,PhasePartition(q_tot))
  q_pt = PhasePartition(q_tot)
  U           = ρ * u
  V           = ρ * v
  W           = ρ * w
  e_kin       = FT(1//2) * (u^2 + v^2 + w^2)
  e_pot       = FT(grav) * z
  E           = ρ * total_energy(e_kin, e_pot, T, q_pt)
  state.ρ     = ρ
  state.ρu    = SVector(U, V, W) 
  state.ρe    = E + rand(seed)*E/100
  state.moisture.ρq_tot = ρ * q_tot
end

function config_bomex(FT, N, resolution, xmax, ymax, zmax)
  # SGS Filter constants
  C_smag = FT(0.23) # Smagorinsky coefficient
  u_τ    = FT(0.28) # Friction velocity
  w′θ′   = FT(8e-3)  # Latent heat flux
  w′qt′  = FT(5.2e-5)   # Sensible heat flux

  bc = BOMEX_BC{FT}(u_τ, w′θ′, w′qt′) # Boundary conditions
  ics = init_bomex! # Initial conditions 

  source = (
            Gravity(),
            BomexMoistureTendency{FT}(-1.2e-8, 300, 500),
            BomexSponge{FT}(3000, 2400, 0.5, 2, -10, 1.8e-3, 0),
            BomexLargeScaleSubsidence{FT}(-0.65e-2, 1500, 2100),
            BomexGeostrophic{FT}(0.376e-4, -10, 1.8e-3, 0),
            BomexTemperatureTendency{FT}(2/86400, 1500, 3000)
           )

  ode_solver_type = CLIMA.DefaultSolverType()
  model = AtmosModel{FT}(AtmosLESConfiguration;
                         turbulence        = SmagorinskyLilly{FT}(C_smag),
                         moisture          = EquilMoist(20),
                         source            = source,
                         boundarycondition = bc,
                         init_state        = ics)

  config = CLIMA.Atmos_LES_Configuration("BOMEX", N, resolution,
                                         xmax, ymax, zmax,
                                         init_bomex!,
                                         solver_type=ode_solver_type,
                                         model=model)
    return config
end

function main()
  CLIMA.init()

  FT = Float64

  # DG polynomial order
  N = 4

  # Domain resolution and size
  Δh = FT(100)
  Δv = FT(40)

  resolution = (Δh, Δh, Δv)

  xmax = 6400
  ymax = 6400
  zmax = 3000

  t0 = FT(0)
  timeend = FT(3600*6)
  CFLmax  = FT(0.1)

  driver_config = config_bomex(FT, N, resolution, xmax, ymax, zmax)
  solver_config = CLIMA.setup_solver(t0, timeend, driver_config, forcecpu=true, Courant_number=CFLmax)

  cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(1) do (init=false)
      Filters.apply!(solver_config.Q, 6, solver_config.dg.grid, TMARFilter())
      nothing
  end
    
  result = CLIMA.invoke!(solver_config;
                        user_callbacks=(cbtmarfilter,),
                        check_euclidean_distance=true)
end

main()
