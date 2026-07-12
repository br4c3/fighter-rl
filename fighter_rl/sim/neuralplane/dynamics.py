import torch

from .atmosphere import spherical_gravity


class CompetitionDynamics(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Supplied XML empty weight + pilot + full internal fuel.
        self.mass_slug = 764.6535255789535
        # JSBSim combines base inertia, pilot and full tank contents. Values
        # were read from the exact core after the DLL-equivalent reset.
        self.Jx, self.Jy, self.Jz, self.Jxz = 15963.7333, 57380.3321, 70914.3931, 1167.6793

        # Full configuration is 17,400 lb empty + 230 lb pilot + two equal
        # internal tanks.  Derive the dry-body inertia about its own CG from
        # the exact full-fuel mass properties, then reapply the parallel-axis
        # theorem for every batched fuel state.
        self.dry_weight = 17630.0
        self.dry_cgx, self.dry_cgz = -194.868149247237, -5.03345098112748

        self.tank_x, self.tank_y, self.tank_z = -174.4, 65.0, 5.0

        full_fuel = 6971.963548061171
        full_cgx, full_cgz = -189.06766929517926, -2.1900658483050157

        dry_mass = self.dry_weight / 32.174
        fuel_mass = full_fuel / 32.174

        ddx = (full_cgx - self.dry_cgx) / 12
        ddz = (full_cgz - self.dry_cgz) / 12

        tx = (full_cgx - self.tank_x) / 12
        ty = self.tank_y / 12
        tz = (full_cgz - self.tank_z) / 12

        self.dry_Jx = 15963.767196323115 - dry_mass * ddz**2 - fuel_mass * (ty**2 + tz**2)
        self.dry_Jy = (
            57380.332084958936 - dry_mass * (ddx**2 + ddz**2) - fuel_mass * (tx**2 + tz**2)
        )
        self.dry_Jz = 70914.42894831116 - dry_mass * ddx**2 - fuel_mass * (tx**2 + ty**2)

        # Store the conventional positive-cross-term form used by the closed
        # form rotational equations below (negative of JSBSim property Ixz).
        full_cross = 1167.6806108567112
        self.dry_Jxz = full_cross - dry_mass * ddx * ddz - fuel_mass * tx * tz

        # AIP reset origin.  Over an air-combat episode latitude drift is tiny,
        # but Earth-rate terms are not: 2*Omega*V is ~0.1 ft/s^2 at 250 m/s.
        self.latitude_rad = 37.91455691666667 * 3.141592653589793 / 180.0
        self.earth_rate = 7.292115e-5
        self.earth_radius_ft = 20_909_000.0

    def mass_properties(self, fuel_lbs, reference):
        if fuel_lbs is None:
            fuel_lbs = reference.new_full((reference.shape[0],), 6971.963548061171)
        else:
            fuel_lbs = fuel_lbs.to(device=reference.device, dtype=reference.dtype)

        total_weight = self.dry_weight + fuel_lbs

        cgx = (self.dry_weight * self.dry_cgx + fuel_lbs * self.tank_x) / total_weight
        cgz = (self.dry_weight * self.dry_cgz + fuel_lbs * self.tank_z) / total_weight

        dm = reference.new_tensor(self.dry_weight / 32.174)
        fm = fuel_lbs / 32.174

        ddx = (cgx - self.dry_cgx) / 12
        ddz = (cgz - self.dry_cgz) / 12

        tx = (cgx - self.tank_x) / 12
        ty = reference.new_tensor(self.tank_y / 12)
        tz = (cgz - self.tank_z) / 12

        jx = self.dry_Jx + dm * ddz.square() + fm * (ty.square() + tz.square())
        jy = self.dry_Jy + dm * (ddx.square() + ddz.square()) + fm * (tx.square() + tz.square())
        jz = self.dry_Jz + dm * ddx.square() + fm * (tx.square() + ty.square())
        jxz = self.dry_Jxz + dm * ddx * ddz + fm * tx * tz

        return total_weight / 32.174, cgx, cgz, jx, jy, jz, jxz

    def specific_force(self, state, aero, thrust_lbf, fuel_lbs=None):
        """Body force/mass, excluding gravity, Coriolis and centrifugal terms."""
        # fmt: off
        vt, alpha, beta = state[:,6], state[:,7], state[:,8]
        ca, sa, cb, sb = torch.cos(alpha), torch.sin(alpha), torch.cos(beta), torch.sin(beta)
        D, C, L = aero[:,0], aero[:,1], aero[:,2]
        # fmt: on
        # JSBSim transforms native wind-axis aero forces [-Drag, Side, -Lift]
        # with mTw2b.  The Side contribution is therefore negative in body X/Z.
        # fmt: off
        fx = -D*ca*cb - C*ca*sb + L*sa + thrust_lbf
        fy = C*cb - D*sb
        fz = -D*sa*cb - C*sa*sb - L*ca
        # fmt: on

        mass = self.mass_properties(fuel_lbs, state)[0]

        # fmt: off
        return torch.stack((fx/mass, fy/mass, fz/mass), 1)
        # fmt: on

    def forward(self, state, aero, thrust_lbf, fuel_lbs=None):
        """state12 derivative; aero columns are D,C,L,Lmom,Mmom,Nmom."""
        x = state
        # fmt: off
        phi, theta, psi = x[:,3], x[:,4], x[:,5]
        vt, alpha, beta = x[:,6], x[:,7], x[:,8]
        p, q, r = x[:,9], x[:,10], x[:,11]

        sa, ca, sb, cb = torch.sin(alpha), torch.cos(alpha), torch.sin(beta), torch.cos(beta)
        sp, cp, st, ct = torch.sin(phi), torch.cos(phi), torch.sin(theta), torch.cos(theta)
        ss, cs = torch.sin(psi), torch.cos(psi)
        D, C, L, lm, mm, nm = aero.unbind(1)

        u, v, w = vt*ca*cb, vt*sb, vt*sa*cb

        fx = -D*ca*cb - C*ca*sb + L*sa + thrust_lbf
        # fmt: on

        # JSBSim SIDE-axis function already carries the body-Y sign.
        # fmt: off
        fy = C*cb - D*sb
        fz = -D*sa*cb - C*sa*sb - L*ca
        # fmt: on

        # XML moments are evaluated at AERORP. JSBSim transfers the force to
        # the current full-fuel CG. Structural-coordinate convention gives
        # M_CG = M_RP - r_(CG->RP) x F_aero.
        aero_fx = fx - thrust_lbf
        mass, cgx, cgz, Jx, Jy, Jz, Jxz = self.mass_properties(fuel_lbs, x)
        rx = (-189.5 - cgx) / 12.0
        rz = (3.9 - cgz) / 12.0

        # fmt: off
        lm = lm + rz*fy
        mm = mm - rz*aero_fx + rx*fz
        nm = nm - rx*fy
        # fmt: on

        # The direct thruster is at structural (0,0,0), not at the CG.
        # Full-fuel CG z=-2.1901 in gives a nose-down thrust moment.
        thruster_rz = cgz / 12.0
        # fmt: off
        mm = mm + thruster_rz*thrust_lbf
        # fmt: on

        # fmt: off
        g = spherical_gravity(x[:,2])
        # fmt: on

        # fmt: off
        ud = r*v - q*w - g*st + fx/mass
        vd = p*w - r*u + g*ct*sp + fy/mass
        wd = q*u - p*v + g*ct*cp + fz/mass
        # fmt: on

        # FGAccelerations ECEF terms:
        # -2*(Ti2b*OmegaPlanet)xUVW - Ti2b*(Omega x (Omega x R)).
        lat = x.new_tensor(self.latitude_rad)
        om = x.new_tensor(self.earth_rate)
        # fmt: off
        omega_n = om*torch.cos(lat)
        omega_d = -om*torch.sin(lat)
        # fmt: on

        # NED-to-body transform (transpose of the body-to-NED Euler matrix).
        # fmt: off
        omega_bx = ct*cs*omega_n - st*omega_d
        omega_by = (sp*st*cs - cp*ss)*omega_n + sp*ct*omega_d
        omega_bz = (cp*st*cs + sp*ss)*omega_n + cp*ct*omega_d

        coriolis_x = -2*(omega_by*w - omega_bz*v)
        coriolis_y = -2*(omega_bz*u - omega_bx*w)
        coriolis_z = -2*(omega_bx*v - omega_by*u)
        # fmt: on

        # fmt: off
        radius = x.new_tensor(self.earth_radius_ft) + x[:,2]
        # fmt: on
        # JSBSim's local Down axis is itself constructed from
        # J2-gravity minus Omega x (Omega x sea-level-position).  Expressed in
        # that local frame, gravitational north and sea-level centrifugal
        # north cancel; only the altitude-dependent remainder is negligible
        # over the competition envelope.  Adding the full north term here
        # would double count ~0.054 ft/s^2.
        centrifugal_n = torch.zeros_like(radius)
        # fmt: off
        centrifugal_d = -om.square()*radius*torch.cos(lat).square()
        centrifugal_x = ct*cs*centrifugal_n - st*centrifugal_d
        centrifugal_y = (sp*st*cs - cp*ss)*centrifugal_n + sp*ct*centrifugal_d
        centrifugal_z = (cp*st*cs + sp*ss)*centrifugal_n + cp*ct*centrifugal_d
        # fmt: on

        ud = ud + coriolis_x + centrifugal_x
        vd = vd + coriolis_y + centrifugal_y
        wd = wd + coriolis_z + centrifugal_z
        safe_vt = vt.clamp_min(1.0)

        # fmt: off
        vt_d = (u*ud + v*vd + w*wd) / safe_vt
        alpha_d = (u*wd - w*ud) / (u.square() + w.square()).clamp_min(1.0)
        beta_d = (vd*safe_vt - v*vt_d) / (safe_vt.square()*cb).clamp_min(1.0)
        # fmt: on

        # FGAccelerations solves Euler's equation using angular velocity
        # relative to ECI, not the reported Earth-relative PQR:
        #   PQRi_dot = J^-1 (M - PQRi x J*PQRi)
        #   PQR_dot  = PQRi_dot + PQRi x omega_earth_body
        # The distinction is small per frame but measurable in long rolling
        # manoeuvres. JSBSim's inertia matrix stores -Jxz off diagonal.
        pqr = torch.stack((p, q, r), 1)
        omega_b = torch.stack((omega_bx, omega_by, omega_bz), 1)
        pqri = pqr + omega_b
        pi, qi, ri = pqri.unbind(1)
        # fmt: off
        angular_momentum = torch.stack((Jx*pi - Jxz*ri, Jy*qi, Jz*ri - Jxz*pi), 1)
        # fmt: on
        rhs = torch.stack((lm, mm, nm), 1) - torch.linalg.cross(pqri, angular_momentum)
        rxm, rym, rzm = rhs.unbind(1)
        denom = Jx * Jz - Jxz.square()
        # fmt: off
        pqri_dot = torch.stack((
            (Jz*rxm + Jxz*rzm) / denom,
            rym / Jy,
            (Jxz*rxm + Jx*rzm) / denom,
        ), 1)
        # fmt: on
        pqr_dot = pqri_dot + torch.linalg.cross(pqri, omega_b)
        pd, qd, rd = pqr_dot.unbind(1)
        safe_ct = torch.where(ct.abs() < 1e-4, ct.sign() * 1e-4, ct)

        # fmt: off
        north = u*ct*cs + v*(sp*st*cs - cp*ss) + w*(cp*st*cs + sp*ss)
        east = u*ct*ss + v*(sp*st*ss + cp*cs) + w*(cp*st*ss - sp*cs)
        altitude = u*st - v*sp*ct - w*cp*ct

        phi_d = p + torch.tan(theta)*(q*sp + r*cp)
        theta_d = q*cp - r*sp
        psi_d = (q*sp + r*cp) / safe_ct
        # fmt: on

        return torch.stack(
            (north, east, altitude, phi_d, theta_d, psi_d, vt_d, alpha_d, beta_d, pd, qd, rd), 1
        )
