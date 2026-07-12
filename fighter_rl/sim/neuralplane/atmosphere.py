import torch

EARTH_RADIUS_FT = 6356766.0 / 0.3048
SL_TEMPERATURE_R = 518.67
SL_PRESSURE_PSF = 2116.228
R_DRY = 1716.56305
G_ISA = 9.80665 / 0.3048
GAMMA = 1.4


def standard_atmosphere(altitude_ft):
    """Return density, speed of sound and Mach denominator through 20 km.

    This ports the first two ISA-1976 layers used by the competition envelope,
    including JSBSim's geometric-to-geopotential altitude conversion.
    """
    # fmt: off
    h = altitude_ft*EARTH_RADIUS_FT/(EARTH_RADIUS_FT + altitude_ft)
    lapse = (389.97 - SL_TEMPERATURE_R)/36089.2388
    t_trop = SL_TEMPERATURE_R + lapse*h
    exponent = G_ISA/(R_DRY*lapse)
    p_trop = SL_PRESSURE_PSF*(SL_TEMPERATURE_R/t_trop).pow(exponent)
    # fmt: on
    h11 = altitude_ft.new_tensor(36089.2388)
    t11 = altitude_ft.new_tensor(389.97)
    # fmt: off
    p11 = altitude_ft.new_tensor(SL_PRESSURE_PSF)*(altitude_ft.new_tensor(SL_TEMPERATURE_R)/t11).pow(exponent)
    p_strat = p11*torch.exp(-G_ISA*(h - h11)/(R_DRY*t11))
    # fmt: on
    temperature = torch.where(h < h11, t_trop, t11)
    pressure = torch.where(h < h11, p_trop, p_strat)
    # fmt: off
    density = pressure/(R_DRY*temperature)
    sound_speed = torch.sqrt(GAMMA*R_DRY*temperature)
    # fmt: on

    return density, sound_speed


def standard_temperature(altitude_ft):
    """ISA temperature in Rankine for the competition altitude envelope."""
    # fmt: off
    h = altitude_ft*EARTH_RADIUS_FT/(EARTH_RADIUS_FT + altitude_ft)
    lapse = (389.97 - SL_TEMPERATURE_R)/36089.2388

    return torch.where(h < 36089.2388, SL_TEMPERATURE_R + lapse*h, altitude_ft.new_tensor(389.97))
    # fmt: on


def standard_pressure(altitude_ft):
    temperature = standard_temperature(altitude_ft)
    # fmt: off
    h = altitude_ft*EARTH_RADIUS_FT/(EARTH_RADIUS_FT + altitude_ft)
    lapse = (389.97 - SL_TEMPERATURE_R)/36089.2388
    exponent = G_ISA/(R_DRY*lapse)
    p_trop = SL_PRESSURE_PSF*(SL_TEMPERATURE_R/temperature).pow(exponent)
    # fmt: on
    h11 = altitude_ft.new_tensor(36089.2388)
    t11 = altitude_ft.new_tensor(389.97)
    # fmt: off
    p11 = altitude_ft.new_tensor(SL_PRESSURE_PSF)*(altitude_ft.new_tensor(SL_TEMPERATURE_R)/t11).pow(exponent)
    p_strat = p11*torch.exp(-G_ISA*(h - h11)/(R_DRY*t11))
    # fmt: on

    return torch.where(h < h11, p_trop, p_strat)


def calibrated_airspeed(true_speed_fps, altitude_ft):
    """Torch port of FGAuxiliary::VcalibratedFromMach, returned in ft/s."""
    _, sound = standard_atmosphere(altitude_ft)
    mach = true_speed_fps / sound
    pressure = standard_pressure(altitude_ft)
    gamma = 1.4
    # fmt: off
    a = (gamma - 1)/2
    b = gamma/(gamma - 1)
    c = 2*b
    d = 1/(gamma - 1)
    sub_total = pressure*(1 + a*mach.square()).pow(b)
    coeff = (0.5*(gamma + 1))**b*((gamma + 1)/(gamma - 1))**d
    sup_total = (
        pressure*coeff*mach.clamp_min(1).pow(c)/(c*mach.clamp_min(1).square() - 1).pow(d)
    )
    qc = torch.where(mach < 1, sub_total, sup_total) - pressure
    A = qc/SL_PRESSURE_PSF + 1
    ma = torch.sqrt((2/(gamma - 1))*(A.pow((gamma - 1)/gamma) - 1).clamp_min(0))
    iterate_coeff = (0.5*(gamma + 1))**(-0.25*(2/((gamma - 1)/gamma))) * (
        0.5*(gamma + 1)/gamma
    )**(-0.5*(0.5*2/(gamma - 1)))

    for _ in range(10):
        sup = iterate_coeff * torch.sqrt(
            A
            * (1 - 1/((2/((gamma - 1)/gamma))*ma.clamp_min(1e-3).square()))
            .clamp_min(0)
            .pow(0.5*2/(gamma - 1))
        )
        ma = torch.where(ma > 1, sup, ma)
    return (GAMMA*R_DRY*SL_TEMPERATURE_R)**0.5*ma
    # fmt: on


def spherical_gravity(altitude_ft):
    """Competition JSBSim's default WGS84 central gravity magnitude.

    The reset latitude is fixed in the AIP environment.  The effective local
    sea-level radius/gravity below reproduce the native core to <2e-5 ft/s²
    over the tested 3--10 km envelope while retaining the inverse-square law.
    """
    radius = altitude_ft.new_tensor(20_909_000.0)
    g_sea = altitude_ft.new_tensor(32.22148)

    return g_sea * (radius / (radius + altitude_ft)).square()
