"""Vectorized evaluator for all aerodynamic products/tables in competition f16.xml."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
import torch

from .atmosphere import standard_atmosphere


def _interp1(x, grid, values):
    i = torch.searchsorted(grid, x.contiguous()).clamp(1, len(grid) - 1)
    x0, x1 = grid[i - 1], grid[i]
    y0, y1 = values[i - 1], values[i]
    xc = x.clamp(grid[0], grid[-1])
    return y0 + (xc - x0) * (y1 - y0) / (x1 - x0)


def _interp2(row, col, rows, cols, values):
    ir = torch.searchsorted(rows, row.contiguous()).clamp(1, len(rows) - 1)
    ic = torch.searchsorted(cols, col.contiguous()).clamp(1, len(cols) - 1)
    r0, r1, c0, c1 = rows[ir-1], rows[ir], cols[ic-1], cols[ic]
    tr = ((row.clamp(rows[0], rows[-1]) - r0) / (r1-r0)).clamp(0, 1)
    tc = ((col.clamp(cols[0], cols[-1]) - c0) / (c1-c0)).clamp(0, 1)
    v00, v01 = values[ir-1, ic-1], values[ir-1, ic]
    v10, v11 = values[ir, ic-1], values[ir, ic]
    return (v00*(1-tc)+v01*tc)*(1-tr) + (v10*(1-tc)+v11*tc)*tr


class CompetitionXMLAero(torch.nn.Module):
    """Evaluate the XML exactly; tables can later be distilled into MLPs."""

    AXES = ("DRAG", "SIDE", "LIFT", "ROLL", "PITCH", "YAW")

    def __init__(self, xml_path: str | Path):
        super().__init__()
        aero = ET.parse(xml_path).getroot().find("aerodynamics")
        self.specs = {name: [] for name in self.AXES}
        table_number = 0
        for axis in aero.findall("axis"):
            for function in axis.findall("function"):
                product = function.find("product")
                factors = []
                for child in product:
                    if child.tag == "property":
                        factors.append(("property", (child.text or "").strip()))
                    elif child.tag == "value":
                        factors.append(("value", float(child.text)))
                    elif child.tag == "table":
                        independent = [(v.text or "").strip() for v in child.findall("independentVar")]
                        lines = [[float(x) for x in line.split()]
                                 for line in (child.findtext("tableData") or "").strip().splitlines()]
                        prefix = f"table_{table_number}"; table_number += 1
                        if len(independent) == 1:
                            array = torch.tensor(lines, dtype=torch.float32)
                            self.register_buffer(prefix + "_x", array[:, 0].contiguous())
                            self.register_buffer(prefix + "_v", array[:, 1].contiguous())
                            factors.append(("table1", (prefix, independent[0])))
                        elif len(independent) == 2:
                            cols = torch.tensor(lines[0], dtype=torch.float32)
                            body = torch.tensor(lines[1:], dtype=torch.float32)
                            self.register_buffer(prefix + "_r", body[:, 0].contiguous())
                            self.register_buffer(prefix + "_c", cols.contiguous())
                            self.register_buffer(prefix + "_v", body[:, 1:].contiguous())
                            factors.append(("table2", (prefix, independent[0], independent[1])))
                        else:
                            raise ValueError("only 1D/2D tables are present in competition XML")
                self.specs[axis.attrib["name"]].append((function.attrib.get("name", ""), factors))

    def forward(self, properties: dict[str, torch.Tensor]):
        reference = next(iter(properties.values()))
        outputs = []
        for axis in self.AXES:
            total = torch.zeros_like(reference)
            for _name, factors in self.specs[axis]:
                value = torch.ones_like(reference)
                for kind, item in factors:
                    if kind == "property": value = value * properties[item]
                    elif kind == "value": value = value * item
                    elif kind == "table1":
                        prefix, key = item
                        value = value * _interp1(properties[key], getattr(self, prefix+"_x"),
                                                 getattr(self, prefix+"_v"))
                    else:
                        prefix, row, col = item
                        value = value * _interp2(properties[row], properties[col],
                            getattr(self, prefix+"_r"), getattr(self, prefix+"_c"),
                            getattr(self, prefix+"_v"))
                total = total + value
            outputs.append(total)
        return torch.stack(outputs, dim=-1)

    def components(self, properties: dict[str, torch.Tensor]):
        """Named XML function outputs for module-level diagnostics."""
        reference = next(iter(properties.values())); result = {}
        for axis in self.AXES:
            for name, factors in self.specs[axis]:
                value = torch.ones_like(reference)
                for kind, item in factors:
                    if kind == "property": value = value * properties[item]
                    elif kind == "value": value = value * item
                    elif kind == "table1":
                        prefix,key=item;value=value*_interp1(properties[key],getattr(self,prefix+"_x"),getattr(self,prefix+"_v"))
                    else:
                        prefix,row,col=item;value=value*_interp2(properties[row],properties[col],getattr(self,prefix+"_r"),getattr(self,prefix+"_c"),getattr(self,prefix+"_v"))
                result[name] = value
        return result


def airborne_properties(state12, surfaces_deg, speedbrake_deg=None, gear_pos=None):
    """Build the 21 XML properties for airborne gear-up training."""
    alt, vt = state12[:, 2], state12[:, 6]
    alpha, beta = state12[:, 7], state12[:, 8]
    p, q, r = state12[:, 9], state12[:, 10], state12[:, 11]
    values = torch.deg2rad(surfaces_deg)
    elevator, aileron, rudder, lef, tef = values[:, :5].unbind(1)
    aero_aileron = values[:, 5] if values.shape[1] > 5 else aileron
    rho, sound_speed = standard_atmosphere(alt)
    mach = vt / sound_speed
    qbar = .5 * rho * vt.square()
    sb = torch.zeros_like(vt) if speedbrake_deg is None else torch.deg2rad(speedbrake_deg)
    gear = torch.zeros_like(vt) if gear_pos is None else gear_pos
    tef_control=tef/.349
    aileron_speed_comp=aileron/.375
    flaperon_mix=((-tef_control-aileron_speed_comp).clamp(-1,1)
                  +(tef_control-aileron_speed_comp).clamp(-1,1))*1.4324
    return {
        "aero/alpha-rad": alpha, "aero/beta-rad": beta,
        "aero/bi2vel": 30./(2*vt.clamp_min(1)), "aero/ci2vel": 11.32/(2*vt.clamp_min(1)),
        "aero/function/kCLge": torch.ones_like(vt), "aero/h_b-mac-ft": torch.full_like(vt, 2.),
        "aero/qbar-psf": qbar, "fcs/aileron-pos-rad": aero_aileron,
        "fcs/elevator-pos-rad": elevator,
        "fcs/flaperon-mix-rad": flaperon_mix,
        "fcs/lef-pos-rad": lef, "fcs/rudder-pos-rad": rudder,
        "fcs/speedbrake-pos-rad": sb, "gear/gear-pos-norm": gear,
        "metrics/Sw-sqft": torch.full_like(vt, 300.), "metrics/bw-ft": torch.full_like(vt, 30.),
        "metrics/cbarw-ft": torch.full_like(vt, 11.32), "velocities/mach": mach,
        "velocities/p-aero-rad_sec": p, "velocities/q-aero-rad_sec": q,
        "velocities/r-aero-rad_sec": r,
    }
