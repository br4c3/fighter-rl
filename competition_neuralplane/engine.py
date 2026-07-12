"""Batched F100-PW-229 thrust tables and spool state."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import NamedTuple
import torch

from .xml_aero import _interp2
from .atmosphere import standard_atmosphere, standard_temperature


class EngineState(NamedTuple):
    n1: torch.Tensor
    n2: torch.Tensor
    thrust_lbf: torch.Tensor
    fuel_flow_pph: torch.Tensor


class CompetitionF100(torch.nn.Module):
    """XML thrust maps are exact; spool time constants are exposed for fitting."""

    def __init__(self, xml_path: str | Path, hz=60, spool_up_s=1.0, spool_down_s=1.5):
        super().__init__(); self.dt = 1./float(hz)
        root = ET.parse(xml_path).getroot()
        self.mil = float(root.findtext("milthrust")); self.maximum = float(root.findtext("maxthrust"))
        self.idle_n1 = float(root.findtext("idlen1")); self.idle_n2 = float(root.findtext("idlen2"))
        self.spool_up_s, self.spool_down_s = float(spool_up_s), float(spool_down_s)
        for function in root.findall("function"):
            name = function.attrib["name"].lower().replace("thrust", "")
            lines = [[float(x) for x in line.split()]
                     for line in function.find("table/tableData").text.strip().splitlines()]
            cols = torch.tensor(lines[0]); body = torch.tensor(lines[1:])
            self.register_buffer(name+"_mach", body[:,0].contiguous())
            self.register_buffer(name+"_alt", cols.contiguous())
            self.register_buffer(name+"_map", body[:,1:].contiguous())

    def initial(self, batch, device=None):
        n1 = torch.full((batch,), self.idle_n1, device=device)
        n2 = torch.full((batch,), self.idle_n2, device=device)
        # InitRunning(-1) leaves the turbine near this flow before the one
        # full-throttle DLL warm-start frame (which adds 100 pph).
        return EngineState(n1, n2, torch.zeros_like(n1), torch.full_like(n1,6461.35))

    def _map(self, name, mach, altitude_ft):
        return _interp2(mach, altitude_ft, getattr(self,name+"_mach"),
                        getattr(self,name+"_alt"), getattr(self,name+"_map"))

    def steady_dry_fuel_flow(self, n2, mach, altitude_ft):
        """Fuel flow left by JSBSim InitRunning/GetSteadyState (lb/hour)."""
        idle=self._map("idle",mach,altitude_ft)*self.mil
        mil_component=self._map("mil",mach,altitude_ft)*(self.mil-idle)
        n2norm=((n2-self.idle_n2)/(100-self.idle_n2)).clamp(0,1)
        dry=idle+mil_component*n2norm.square()
        temperature=standard_temperature(altitude_ft)
        corrected_tsfc=.74*torch.sqrt(temperature/389.7)*(.84+(1-n2norm).square())
        return (dry*corrected_tsfc).clamp_min(self.mil**.2*107.)

    def thrust_from_n2(self, n2, throttle_cmd, mach, altitude_ft):
        """Recover current thrust from exposed release N2/throttle telemetry."""
        lever=(2.0*throttle_cmd.clamp(0,1))
        idle=self._map("idle",mach,altitude_ft)*self.mil
        mil_component=self._map("mil",mach,altitude_ft)*(self.mil-idle)
        aug=self._map("aug",mach,altitude_ft)*self.maximum
        n2norm=((n2-self.idle_n2)/(100-self.idle_n2)).clamp(0,1)
        dry=idle+mil_component*n2norm.square()
        return torch.where(lever>1,dry+(aug-dry)*(lever-1).clamp(0,1),dry)

    def forward(self, state: EngineState, throttle_cmd, mach, altitude_ft):
        # f16.xml doubles pilot throttle: [0,.5] dry range, [.5,1] augmentation.
        lever = (2.0*throttle_cmd.clamp(0,1))
        idle = self._map("idle",mach,altitude_ft)*self.mil
        mil_component = self._map("mil",mach,altitude_ft)*(self.mil-idle)
        aug = self._map("aug",mach,altitude_ft)*self.maximum
        dry_fraction = lever.clamp(0,1)
        target_n1 = self.idle_n1+(100-self.idle_n1)*dry_fraction
        target_n1 = torch.where(lever>1, target_n1.new_tensor(100.), target_n1)
        target_n2 = self.idle_n2+(100-self.idle_n2)*dry_fraction
        target_n2 = torch.where(lever>1, target_n2.new_tensor(100.), target_n2)
        n2norm0=((state.n2-self.idle_n2)/(100-self.idle_n2)).clamp(0,1)
        n=torch.minimum(torch.ones_like(n2norm0),n2norm0+.1)
        # Exact default FGSpoolUp rates for bypass ratio 0.4.
        density,_=standard_atmosphere(altitude_ft);density_ratio=density/2.3768924e-3
        denominator=1+3*(1-n).pow(3)+(1-density_ratio)
        up=90/3.4/denominator;down_n1=2.4*90/3.4/denominator;down_n2=3.0*90/3.4/denominator
        def seek(current,target,up_rate,down_rate):
            delta=torch.where(target>current,up_rate*self.dt,-down_rate*self.dt)
            candidate=current+delta
            return torch.where(target>current,torch.minimum(candidate,target),torch.maximum(candidate,target))
        n1=seek(state.n1,target_n1,up,down_n1);n2=seek(state.n2,target_n2,up,down_n2)
        n2norm=((n2-self.idle_n2)/(100-self.idle_n2)).clamp(0,1)
        dry=idle+mil_component*n2norm.square()
        thrust=torch.where(lever>1,dry+(aug-dry)*(lever-1).clamp(0,1),dry)
        # Exact FGTurbine fuel model: corrected dry TSFC, augmented TSFC and
        # asymmetric Seek rates in lb/hour per second.
        temperature=standard_temperature(altitude_ft)
        corrected_tsfc=.74*torch.sqrt(temperature/389.7)*(.84+(1-n2norm).square())
        target_dry=(dry*corrected_tsfc).clamp_min(self.mil**.2*107.)
        target_aug=thrust*2.05
        target_flow=torch.where(lever>1,target_aug,target_dry)
        accel=torch.where(lever>1,torch.full_like(lever,5000.),torch.full_like(lever,1000.))
        flow=seek(state.fuel_flow_pph,target_flow,accel,torch.full_like(lever,10000.))
        return EngineState(n1,n2,thrust,flow)
