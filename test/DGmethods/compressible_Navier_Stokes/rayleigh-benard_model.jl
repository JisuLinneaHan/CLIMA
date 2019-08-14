# Load Packages 
using MPI
using CLIMA
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.DGmethods
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.MPIStateArrays
using CLIMA.LowStorageRungeKuttaMethod
using CLIMA.ODESolvers
using CLIMA.GenericCallbacks
using CLIMA.Atmos
using CLIMA.VariableTemplates
using CLIMA.MoistThermodynamics
using CLIMA.PlanetParameters
using LinearAlgebra
using StaticArrays
using Logging, Printf, Dates
using CLIMA.Vtk
using Random

@static if haspkg("CuArrays")
  using CUDAdrv
  using CUDAnative
  using CuArrays
  CuArrays.allowscalar(false)
  const ArrayType = CuArray
else
  const ArrayType = Array 
end

if !@isdefined integration_testing
  const integration_testing =
    parse(Bool, lowercase(get(ENV,"JULIA_CLIMA_INTEGRATION_TESTING","false")))
end

# -------------- Problem constants ------------------- # 
const xmin      = 0
const ymin      = 0
const zmin      = 0
const xmax      = 2000
const ymax      = 400
const zmax      = 2000
const Ne        = (10,2,10)
const polynomialorder = 4
const dim       = 3
const Δx        = (xmax-xmin)/(Ne[1]*polynomialorder+1)
const Δy        = (ymax-ymin)/(Ne[2]*polynomialorder+1)
const Δz        = (zmax-zmin)/(Ne[3]*polynomialorder+1)
const Δ         = cbrt(Δx * Δy * Δz) 
const dt        = 0.01
const timeend   = 10000dt
const T_bot     = 320
const T_lapse   = -0.04
const T_top     = T_bot + T_lapse*zmax
const C_smag    = 0.15
const seed      = MersenneTwister(0)
# ------------- Initial condition function ----------- # 
function initialise_rayleigh_benard!(state::Vars, aux::Vars, (x,y,z), t)
  DF            = eltype(state)
  R_gas::DF     = R_d
  c_p::DF       = cp_d
  c_v::DF       = cv_d
  γ::DF         = c_p / c_v
  p0::DF        = MSLP
  δT            = z != DF(0) ? rand(seed, DF)/100 : 0 
  δw            = z != DF(0) ? rand(seed, DF)/100 : 0
  ΔT            = T_lapse * z + δT
  T             = T_bot + ΔT 
  P             = p0*(T/T_bot)^(grav/R_gas/T_lapse)
  ρ             = P / (R_gas * T)
  ρu, ρv, ρw    = DF(0) , DF(0) , ρ * δw
  E_int         = ρ * c_v * (T-T_0)
  E_pot         = ρ * grav * z
  E_kin         = ρ * 0.5 * δw^2 
  ρe_tot        = E_int + E_pot + E_kin
  state.ρ       = ρ
  state.ρu      = SVector(ρu, ρv, ρw)
  state.ρe      = ρe_tot
end
# --------------- Gravity source --------------------- # 
function source_geopot!(source::Vars, state::Vars, aux::Vars, t::Real)
  DF = eltype(state)
  source.ρu -= SVector(DF(0),
                       DF(0),
                       state.ρ * DF(grav))
end
# --------------- Driver definition ------------------ # 
function run(mpicomm, ArrayType, 
             topl, dim, Ne, polynomialorder, 
             timeend, DF, dt)
  # -------------- Define grid ----------------------------------- # 
  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = DF,
                                          DeviceArray = ArrayType,
                                          polynomialorder = polynomialorder
                                         )
  # -------------- Define model ---------------------------------- # 
  model = AtmosModel(SmagorinskyLilly(DF(C_smag), DF(Δ)), 
                     DryModel(), 
                     NoRadiation(),
                     source_geopot!, RayleighBenardBC(), initialise_rayleigh_benard!)
  # -------------- Define dgbalancelaw --------------------------- # 
  dg = DGModel(model,
               grid,
               Rusanov(),
               DefaultGradNumericalFlux())

  param = init_ode_param(dg)

  Q = init_ode_state(dg, param, DF(0))

  lsrk = LSRK54CarpenterKennedy(dg, Q; dt = dt, t0 = 0)

  eng0 = norm(Q)
  @info @sprintf """Starting
  norm(Q₀) = %.16e""" eng0

  # Set up the information callback (output field dump is via vtk callback: see cbinfo)
  starttime = Ref(now())
  cbinfo = GenericCallbacks.EveryXWallTimeSeconds(60, mpicomm) do (s=false)
    if s
      starttime[] = now()
    else
      energy = norm(Q)
      @info @sprintf("""Update
                     simtime = %.16e
                     runtime = %s
                     norm(Q) = %.16e""", ODESolvers.gettime(lsrk),
                     Dates.format(convert(Dates.DateTime,
                                          Dates.now()-starttime[]),
                                  Dates.dateformat"HH:MM:SS"),
                     energy)
    end
  end

  solve!(Q, lsrk, param; timeend=timeend, callbacks=(cbinfo,))

  # End of the simulation information
  engf = norm(Q)
  Qe = init_ode_state(dg, param, DF(timeend))
  engfe = norm(Qe)
  errf = euclidean_distance(Q, Qe)
  @info @sprintf """Finished
  norm(Q)                 = %.16e
  norm(Q) / norm(Q₀)      = %.16e
  norm(Q) - norm(Q₀)      = %.16e
  norm(Q - Qe)            = %.16e
  norm(Q - Qe) / norm(Qe) = %.16e
  """ engf engf/eng0 engf-eng0 errf errf / engfe
  errf
end
# --------------- Test block / Loggers ------------------ # 
using Test
let
  MPI.Initialized() || MPI.Init()
  mpicomm = MPI.COMM_WORLD
  ll = uppercase(get(ENV, "JULIA_LOG_LEVEL", "INFO"))
  loglevel = ll == "DEBUG" ? Logging.Debug :
    ll == "WARN"  ? Logging.Warn  :
    ll == "ERROR" ? Logging.Error : Logging.Info
  logger_stream = MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull
  global_logger(ConsoleLogger(logger_stream, loglevel))
  @static if haspkg("CUDAnative")
      device!(MPI.Comm_rank(mpicomm) % length(devices()))
  end
  DF = Float64
  brickrange = (range(DF(xmin); length=Ne[1]+1, stop=xmax),
                range(DF(ymin); length=Ne[2]+1, stop=ymax),
                range(DF(zmin); length=Ne[3]+1, stop=zmax))
  topl = StackedBrickTopology(mpicomm, brickrange, periodicity = (true, true, false))
  engf_eng0 = run(mpicomm, ArrayType, 
                  topl, dim, Ne, polynomialorder, 
                  timeend, DF, dt)
end

#nothing