"""Read-only adapter for the current AIP randomized-loiter kill curriculum."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import copy,os,yaml

ROOT=Path(__file__).resolve().parent
AIP_STAGE_DIR=ROOT.parent/"AIP_LIB"/"DogFightEnv"/"Release"/"experiments"/"loiter_stages"
FILES=(
    "00_safety_random.yaml",
    "01_acquire_front_easy.yaml",
    "02_acquire_front_mixed.yaml",
    "03_acquire_all_easy.yaml",
    "04_close_front.yaml",
    "05_close_all.yaml",
    "06_rear_entry_guided.yaml",
    "07_rear_entry_all.yaml",
    "08_rear_control_guided.yaml",
    "09_rear_control_all.yaml",
    "10_kill_wide_guided.yaml",
    "11_kill_final_loiter.yaml",
)
BRIDGE_SCHEDULE_NAMES={"kill_bridge","stage10_11_bridge","final_kill_bridge"}
GUN_CURRICULUM_SCHEDULE_NAMES={"gun_curriculum","gun","shooting","tight_wez"}
BUCKET_GUN_CURRICULUM_SCHEDULE_NAMES={"gun_bucket_curriculum","bucket_gun","axis_bucket","bucketized_gun"}
CONDITIONS=(
 {"timeout_rate_min":.95,"crash_rate_max":.03,"own_crash_rate_max":.02,"target_crash_rate_max":.03,"own_low_alt_rate_max":.02,"target_low_alt_rate_max":.03,"min_valid_samples":10},
 {"final_ata_deg_max":45.,"crash_rate_max":.06,"own_crash_rate_max":.03,"target_crash_rate_max":.05,"own_low_alt_rate_max":.05,"target_low_alt_rate_max":.05,"min_valid_samples":10},
 {"final_ata_deg_max":40.,"crash_rate_max":.06,"own_crash_rate_max":.03,"target_crash_rate_max":.05,"own_low_alt_rate_max":.05,"target_low_alt_rate_max":.05},
 {"final_ata_deg_max":55.,"crash_rate_max":.08,"own_crash_rate_max":.04,"target_crash_rate_max":.06,"own_low_alt_rate_max":.06,"target_low_alt_rate_max":.06,"min_valid_samples":10},
 {"final_ata_deg_max":45.,"ep_min_distance_max":3500.,"crash_rate_max":.08,"own_crash_rate_max":.04,"target_crash_rate_max":.06,"own_low_alt_rate_max":.06,"target_low_alt_rate_max":.06,"min_valid_samples":10},
 {"final_ata_deg_max":55.,"ep_min_distance_max":3500.,"crash_rate_max":.08,"own_crash_rate_max":.04,"target_crash_rate_max":.06,"own_low_alt_rate_max":.06,"target_low_alt_rate_max":.06,"min_valid_samples":10},
 {"final_ata_deg_max":50.,"final_aa_deg_max":85.,"ep_min_distance_max":3200.,"crash_rate_max":.08,"own_crash_rate_max":.04,"target_crash_rate_max":.06,"own_low_alt_rate_max":.06,"target_low_alt_rate_max":.06,"min_valid_samples":10},
 {"final_ata_deg_max":55.,"final_aa_deg_max":85.,"ep_min_distance_max":3200.,"crash_rate_max":.08,"own_crash_rate_max":.04,"target_crash_rate_max":.06,"own_low_alt_rate_max":.06,"target_low_alt_rate_max":.06,"min_valid_samples":10},
 {"final_ata_deg_max":45.,"final_aa_deg_max":70.,"ep_min_distance_max":2600.,"crash_rate_max":.08,"own_crash_rate_max":.04,"target_crash_rate_max":.06,"own_low_alt_rate_max":.06,"target_low_alt_rate_max":.06,"min_valid_samples":10},
 {"final_ata_deg_max":45.,"final_aa_deg_max":70.,"ep_min_distance_max":2600.,"crash_rate_max":.08,"own_crash_rate_max":.04,"target_crash_rate_max":.06,"own_low_alt_rate_max":.06,"target_low_alt_rate_max":.06,"min_valid_samples":10},
 {"win_rate_min":.35,"crash_rate_max":.08,"own_crash_rate_max":.03,"target_crash_rate_max":.05,"own_low_alt_rate_max":.05,"target_low_alt_rate_max":.05,"min_valid_samples":20},
 {"win_rate_min":.50,"crash_rate_max":.03,"own_crash_rate_max":.01,"target_crash_rate_max":.02,"own_low_alt_rate_max":.02,"target_low_alt_rate_max":.02,"min_valid_samples":20},
)

@dataclass(frozen=True)
class LoiterStage:
    index:int
    name:str
    decision_limit:int
    max_engage_time:float
    step_ratio:int
    ownship_randomization:dict
    target_randomization:dict
    wez:dict
    reward:dict
    advance_conditions:dict
    source:Path

def _is_number(value)->bool:
    return isinstance(value,(int,float)) and not isinstance(value,bool)

def _lerp(a,b,t:float):
    if _is_number(a) and _is_number(b):
        return float(a)+(float(b)-float(a))*t
    if isinstance(a,list) and isinstance(b,list) and len(a)==len(b) and all(_is_number(x) for x in a+b):
        return [_lerp(x,y,t) for x,y in zip(a,b)]
    if isinstance(a,dict) and isinstance(b,dict):
        out=copy.deepcopy(a)
        for key,value in b.items():
            out[key]=_lerp(a[key],value,t) if key in a else copy.deepcopy(value)
        return out
    return copy.deepcopy(b if t>=.5 else a)

def _stage_copy(stage:LoiterStage,*,index:int,name:str|None=None,decision_limit:int|None=None,
                max_engage_time:float|None=None,ownship_randomization:dict|None=None,
                target_randomization:dict|None=None,wez:dict|None=None,reward:dict|None=None,
                advance_conditions:dict|None=None)->LoiterStage:
    return LoiterStage(
        index=index,
        name=name or stage.name,
        decision_limit=int(decision_limit if decision_limit is not None else stage.decision_limit),
        max_engage_time=float(max_engage_time if max_engage_time is not None else stage.max_engage_time),
        step_ratio=stage.step_ratio,
        ownship_randomization=copy.deepcopy(ownship_randomization if ownship_randomization is not None else stage.ownship_randomization),
        target_randomization=copy.deepcopy(target_randomization if target_randomization is not None else stage.target_randomization),
        wez=copy.deepcopy(wez if wez is not None else stage.wez),
        reward=copy.deepcopy(reward if reward is not None else stage.reward),
        advance_conditions=copy.deepcopy(advance_conditions if advance_conditions is not None else stage.advance_conditions),
        source=stage.source,
    )

def _tight_safety_reward(reward:dict,penalty:float=2.0)->dict:
    out=copy.deepcopy(reward)
    out["low_altitude_m"]=max(float(out.get("low_altitude_m",800.0)),1500.0)
    out["low_altitude_penalty"]=max(float(out.get("low_altitude_penalty",1.0)),float(penalty))
    out["altitude_floor_m"]=max(float(out.get("altitude_floor_m",1500.0)),2000.0)
    out["altitude_margin_scale"]=max(float(out.get("altitude_margin_scale",0.0)),0.005)
    return out

def _with_numeric_safety_conditions(conditions:dict)->dict:
    out=copy.deepcopy(conditions)
    # These are not AIP competition rules; they are fast-simulator guardrails.
    # A stage should not advance if NaN/Inf isolation is being exercised or if
    # distance metrics are partly invalid.  Otherwise a policy can pass a gate
    # on stale/contaminated summaries.
    out.setdefault("nonfinite_rate_max",0.003)
    out.setdefault("distance_valid_rate_min",0.995)
    return out

def _with_final_kill_bridge(stages:list[LoiterStage])->list[LoiterStage]:
    """Add NeuralPlane-only bridge stages between AIP stage 10 and final stage 11.

    The original AIP YAML files remain unchanged.  This schedule separates the
    two hard changes that otherwise happen simultaneously at stage 11:

    * target/ownship randomization becomes final-width;
    * WEZ shrinks from 6 deg / 1500 m to 2 deg / 914.4 m.
    """
    if len(stages)<12:
        return stages
    wide=stages[10]
    final=stages[11]
    prefix=[_stage_copy(stage,index=i) for i,stage in enumerate(stages[:11])]
    final_random_own=copy.deepcopy(final.ownship_randomization)
    final_random_target=copy.deepcopy(final.target_randomization)
    bridge_conditions_1=_with_numeric_safety_conditions({"win_rate_min":.35,"crash_rate_max":.08,"own_crash_rate_max":.03,"target_crash_rate_max":.05,"own_low_alt_rate_max":.05,"target_low_alt_rate_max":.05,"min_valid_samples":20})
    bridge_conditions_2=_with_numeric_safety_conditions({"win_rate_min":.40,"crash_rate_max":.06,"own_crash_rate_max":.025,"target_crash_rate_max":.04,"own_low_alt_rate_max":.04,"target_low_alt_rate_max":.04,"min_valid_samples":20})
    bridge_conditions_3=_with_numeric_safety_conditions({"win_rate_min":.45,"crash_rate_max":.05,"own_crash_rate_max":.02,"target_crash_rate_max":.03,"own_low_alt_rate_max":.03,"target_low_alt_rate_max":.03,"min_valid_samples":20})
    final_conditions=_with_numeric_safety_conditions({"win_rate_min":.50,"crash_rate_max":.03,"own_crash_rate_max":.01,"target_crash_rate_max":.02,"own_low_alt_rate_max":.02,"target_low_alt_rate_max":.02,"min_valid_samples":20})
    bridge1_reward=_tight_safety_reward(wide.reward,2.0)
    bridge1_reward.update({"win_reward":150.0,"loss_reward":-150.0,"draw_reward":-48.0})
    bridge1=_stage_copy(
        wide,index=11,name="loiter_kill_bridge_final_random_wide_wez",
        decision_limit=2200,max_engage_time=220.0,
        ownship_randomization=final_random_own,
        target_randomization=final_random_target,
        wez={"angle_deg":6.0,"min_range_m":152.4,"max_range_m":1500.0},
        reward=bridge1_reward,
        advance_conditions=bridge_conditions_1,
    )
    bridge2=_stage_copy(
        final,index=12,name="loiter_kill_bridge_mid_wez",
        decision_limit=2300,max_engage_time=230.0,
        ownship_randomization=final_random_own,
        target_randomization=final_random_target,
        wez={"angle_deg":4.0,"min_range_m":152.4,"max_range_m":1200.0},
        reward=_tight_safety_reward(_lerp(wide.reward,final.reward,.50),2.0),
        advance_conditions=bridge_conditions_2,
    )
    bridge3=_stage_copy(
        final,index=13,name="loiter_kill_bridge_tight_wez",
        decision_limit=2400,max_engage_time=240.0,
        ownship_randomization=final_random_own,
        target_randomization=final_random_target,
        wez={"angle_deg":3.0,"min_range_m":152.4,"max_range_m":1050.0},
        reward=_tight_safety_reward(_lerp(wide.reward,final.reward,.75),2.2),
        advance_conditions=bridge_conditions_3,
    )
    final_stage=_stage_copy(final,index=14,reward=_tight_safety_reward(final.reward,2.5),advance_conditions=final_conditions)
    return prefix+[bridge1,bridge2,bridge3,final_stage]

def _gun_reward(*,phi_scale:float,inner_soft_m:float,win_reward:float=8.0,loss_reward:float=-10.0,
                damage_scale:float=12.0,own_damage_scale:float=18.0,draw_reward:float=-2.0,
                track_scale:float=0.0,track_trail_m:float=900.0,track_x_sigma_m:float=450.0,
                track_y_sigma_m:float=300.0,track_z_sigma_m:float=250.0,
                track_closure_sigma_mps:float=45.0,track_closure_limit_mps:float=80.0,
                track_overshoot_x_m:float=-80.0,track_too_close_m:float=260.0)->dict:
    return {
        "mode":"gun_curriculum",
        "survival_bonus":0.0,
        "step_penalty":-0.002,
        "damage_scale":float(damage_scale),
        "own_damage_scale":float(own_damage_scale),
        "dwell_scale":0.03,
        "dwell_cap_steps":20,
        "phi_scale":float(phi_scale),
        "phi_gamma":0.99,
        "phi_ata_deg":6.0,
        "phi_range_center_m":650.0,
        "phi_range_sigma_m":450.0,
        "aim_scale":0.0,
        "aim_sigma_deg":5.0,
        "aim_range_center_m":650.0,
        "aim_range_sigma_m":450.0,
        "inner_soft_m":float(inner_soft_m),
        "inner_penalty_scale":0.65,
        "hard_collision_m":130.0,
        "hard_collision_penalty":8.0,
        "hard_collision_terminate":True,
        "bad_3_9_penalty":0.55,
        "bad_3_9_range_m":1200.0,
        "bad_3_9_ata_deg":3.0,
        "ahead_no_aim_penalty":0.025,
        "ahead_no_aim_range_m":1500.0,
        "ahead_no_aim_x_m":100.0,
        "ahead_no_aim_ata_deg":5.0,
        "red_wez_penalty":0.08,
        "low_altitude_m":1000.0,
        "low_altitude_penalty":2.0,
        "altitude_floor_m":1800.0,
        "altitude_nominal_m":7000.0,
        "altitude_margin_scale":0.002,
        "target_crash_without_damage_penalty":6.0,
        "target_crash_valid_damage_window":0.05,
        "action_rate_penalty":0.001,
        "track_scale":float(track_scale),
        "track_trail_m":float(track_trail_m),
        "track_x_sigma_m":float(track_x_sigma_m),
        "track_y_sigma_m":float(track_y_sigma_m),
        "track_z_sigma_m":float(track_z_sigma_m),
        "track_closure_sigma_mps":float(track_closure_sigma_mps),
        "track_closure_limit_mps":float(track_closure_limit_mps),
        "track_closure_penalty":0.060,
        "track_overshoot_x_m":float(track_overshoot_x_m),
        "track_overshoot_penalty":0.080,
        "track_too_close_m":float(track_too_close_m),
        "track_too_close_penalty":0.20,
        "win_reward":float(win_reward),
        "loss_reward":float(loss_reward),
        "draw_reward":float(draw_reward),
    }

def _track_reward(*,trail_m:float,scale:float=0.10,phi_scale:float=0.015,inner_soft_m:float=260.0)->dict:
    reward=_gun_reward(
        phi_scale=phi_scale,
        inner_soft_m=inner_soft_m,
        win_reward=0.0,
        loss_reward=-8.0,
        draw_reward=0.0,
        damage_scale=0.0,
        own_damage_scale=10.0,
        track_scale=scale,
        track_trail_m=trail_m,
        track_x_sigma_m=max(250.0,trail_m*0.45),
        track_y_sigma_m=max(180.0,trail_m*0.30),
        track_z_sigma_m=220.0,
        track_closure_sigma_mps=40.0,
        track_closure_limit_mps=70.0,
        track_overshoot_x_m=-100.0,
        track_too_close_m=260.0,
    )
    reward["dwell_scale"]=0.0
    reward["target_crash_without_damage_penalty"]=8.0
    reward["aim_scale"]=max(float(phi_scale)*1.5,0.006)
    reward["aim_sigma_deg"]=8.0
    reward["aim_range_center_m"]=float(trail_m)
    reward["aim_range_sigma_m"]=max(250.0,float(trail_m)*0.45)
    return reward

def _nose_bridge_reward(*,trail_m:float=760.0,phi_scale:float=0.070,inner_soft_m:float=305.0)->dict:
    reward=_gun_reward(
        phi_scale=phi_scale,
        inner_soft_m=inner_soft_m,
        win_reward=1.0,
        loss_reward=-8.0,
        draw_reward=-0.5,
        damage_scale=6.0,
        own_damage_scale=12.0,
        track_scale=0.070,
        track_trail_m=trail_m,
        track_x_sigma_m=300.0,
        track_y_sigma_m=190.0,
        track_z_sigma_m=190.0,
        track_closure_sigma_mps=35.0,
        track_closure_limit_mps=65.0,
        track_overshoot_x_m=-70.0,
        track_too_close_m=250.0,
    )
    reward.update({
        "step_penalty":-0.0015,
        "dwell_scale":0.018,
        "dwell_cap_steps":12,
        "aim_scale":0.055,
        "aim_sigma_deg":3.0,
        "aim_range_center_m":trail_m,
        "aim_range_sigma_m":300.0,
        "track_closure_penalty":0.070,
    })
    return reward

def _gun_target_mix(*items:tuple[str,float])->list[dict]:
    total=sum(float(weight) for _,weight in items)
    return [{"policy":name,"weight":float(weight)/total} for name,weight in items]

def _bucket(name:str,weight:float,**kwargs)->dict:
    out={"name":name,"weight":float(weight)}
    out.update(kwargs)
    return out

def _gun_stage(
    index:int,
    name:str,
    seconds:float,
    distance:list[float],
    ata:list[float],
    aa_tail:list[float],
    target_mix:list[dict],
    *,
    phi_scale:float,
    inner_soft_m:float,
    own_speed:list[float]=[255.0,310.0],
    target_bank:list[float]=[0.0,0.0],
    target_speed:list[float]=[245.0,285.0],
    altitude_offset:list[float]=[-50.0,50.0],
    own_roll_jitter:float=1.0,
    own_pitch_jitter:float=0.5,
    easy_fraction:float=0.20,
    boundary_fraction:float=0.20,
    ensure_initial_feasible:bool=True,
    min_initial_closing_mps:float=8.0,
    max_time_to_wez_fraction:float=0.80,
    reward_override:dict|None=None,
    advance:dict|None=None,
    bucket_mix:list[dict]|None=None,
    sampling:dict|None=None,
)->LoiterStage:
    reward=copy.deepcopy(reward_override) if reward_override is not None else _gun_reward(phi_scale=phi_scale,inner_soft_m=inner_soft_m)
    target_randomization={
        "geometry_mode":"gun_curriculum",
        "distance_m":distance,
        "ata_deg":ata,
        "aa_tail_deg":aa_tail,
        "altitude_m":[6500.0,7600.0],
        "altitude_offset_m":altitude_offset,
        "own_speed_mps":own_speed,
        "speed_mps":target_speed,
        "own_heading_deg":[0.0,360.0],
        "roll_deg":[-4.0,4.0],
        "pitch_deg":[-2.0,2.0],
        "loiter_bank_abs_deg":target_bank,
        "randomize_loiter_direction":True,
        "target_policy_mix":target_mix,
        "boundary_fraction":float(boundary_fraction),
        "easy_fraction":float(easy_fraction),
        "ensure_initial_feasible":bool(ensure_initial_feasible),
        "min_initial_closing_mps":float(min_initial_closing_mps),
        "max_time_to_wez_fraction":float(max_time_to_wez_fraction),
        "feasible_resample_attempts":10,
    }
    if bucket_mix:
        total=sum(float(item.get("weight",1.0)) for item in bucket_mix)
        target_randomization["bucket_mix"]=[{**copy.deepcopy(item),"weight":float(item.get("weight",1.0))/max(total,1e-9)} for item in bucket_mix]
    if sampling:
        target_randomization["sampling"]=copy.deepcopy(sampling)
    conditions={
        "episodes_min":30,
        "own_crash_rate_max":0.02,
        "target_crash_rate_max":0.02,
        "inner_violation_rate_max":0.05,
        "bad_3_9_rate_max":0.05,
        "red_wez_rate_max":0.15,
        "distance_valid_rate_min":0.995,
        "nonfinite_rate_max":0.003,
        "init_feasible_rate_min":0.90,
    }
    conditions.update(advance or {})
    return LoiterStage(
        index=index,
        name=name,
        decision_limit=int(round(float(seconds)*10.0)),
        max_engage_time=float(seconds),
        step_ratio=6,
        ownship_randomization={"radius":0.0,"r_roll":float(own_roll_jitter),"r_pitch":float(own_pitch_jitter),"r_heading":0.0},
        target_randomization=target_randomization,
        wez={"angle_deg":2.0,"min_range_m":152.4,"max_range_m":914.4},
        reward=reward,
        advance_conditions=_with_numeric_safety_conditions(conditions),
        source=ROOT/"synthetic_gun_curriculum",
    )

def _with_gun_curriculum()->list[LoiterStage]:
    return [
        _gun_stage(
            0,"trail_straight_tiny",8.0,[750.0,950.0],[0.0,0.5],[0.0,4.0],
            _gun_target_mix(("straight",1.0)),
            phi_scale=0.018,inner_soft_m=280.0,
            own_speed=[255.0,285.0],target_speed=[245.0,270.0],
            altitude_offset=[-1.0,1.0],own_roll_jitter=0.05,own_pitch_jitter=0.05,
            easy_fraction=0.45,boundary_fraction=0.00,
            reward_override=_track_reward(trail_m=850.0,scale=0.12),
            advance={
                "episodes_min":12,
                "own_crash_rate_max":0.05,
                "track_score_min":0.32,
                "overshoot_rate_max":0.12,
                "closure_violation_rate_max":0.16,
                "inner_violation_rate_max":0.16,
                "bad_3_9_rate_max":0.15,
                "red_wez_rate_max":0.25,
                "init_feasible_rate_min":0.99,
            },
        ),
        _gun_stage(
            1,"trail_speed_wave_straight",12.0,[850.0,1150.0],[0.0,1.0],[0.0,8.0],
            _gun_target_mix(("straight",1.0)),
            phi_scale=0.016,inner_soft_m=280.0,
            own_speed=[250.0,290.0],target_speed=[245.0,275.0],
            target_bank=[0.0,6.0],altitude_offset=[-3.0,3.0],
            own_roll_jitter=0.10,own_pitch_jitter=0.08,
            easy_fraction=0.40,boundary_fraction=0.03,
            reward_override=_track_reward(trail_m=1000.0,scale=0.115),
            advance={"episodes_min":16,"own_crash_rate_max":0.04,"track_score_min":0.30,"overshoot_rate_max":0.12,"closure_violation_rate_max":0.16,"inner_violation_rate_max":0.14,"init_feasible_rate_min":0.98},
        ),
        _gun_stage(
            2,"trail_weak_turn_easy",16.0,[900.0,1350.0],[0.0,2.0],[0.0,14.0],
            _gun_target_mix(("straight",0.70),("weak_turn",0.30)),
            phi_scale=0.014,inner_soft_m=275.0,
            own_speed=[250.0,295.0],target_speed=[242.0,276.0],
            target_bank=[4.0,12.0],altitude_offset=[-8.0,8.0],
            own_roll_jitter=0.25,own_pitch_jitter=0.18,
            easy_fraction=0.36,boundary_fraction=0.05,
            reward_override=_track_reward(trail_m=1100.0,scale=0.11),
            advance={"own_crash_rate_max":0.035,"track_score_min":0.28,"overshoot_rate_max":0.11,"closure_violation_rate_max":0.15,"inner_violation_rate_max":0.12,"init_feasible_rate_min":0.975},
        ),
        _gun_stage(
            3,"trail_constant_loiter",20.0,[1000.0,1600.0],[0.0,4.0],[0.0,22.0],
            _gun_target_mix(("weak_turn",0.40),("constant_turn",0.45),("jink",0.15)),
            phi_scale=0.012,inner_soft_m=270.0,
            own_speed=[250.0,300.0],target_speed=[240.0,278.0],
            target_bank=[8.0,22.0],altitude_offset=[-20.0,20.0],
            own_roll_jitter=0.45,own_pitch_jitter=0.30,
            easy_fraction=0.32,boundary_fraction=0.07,
            reward_override=_track_reward(trail_m=1250.0,scale=0.105),
            advance={"own_crash_rate_max":0.03,"track_score_min":0.25,"overshoot_rate_max":0.10,"closure_violation_rate_max":0.14,"inner_violation_rate_max":0.10,"init_feasible_rate_min":0.965},
        ),
        _gun_stage(
            4,"trail_jink_mild",25.0,[1050.0,1900.0],[0.0,7.0],[0.0,35.0],
            _gun_target_mix(("weak_turn",0.25),("constant_turn",0.40),("jink",0.35)),
            phi_scale=0.010,inner_soft_m=265.0,
            own_speed=[252.0,305.0],target_speed=[238.0,280.0],
            target_bank=[10.0,30.0],altitude_offset=[-40.0,40.0],
            own_roll_jitter=0.70,own_pitch_jitter=0.50,
            easy_fraction=0.28,boundary_fraction=0.08,
            reward_override=_track_reward(trail_m=1400.0,scale=0.10),
            advance={"own_crash_rate_max":0.03,"track_score_min":0.23,"overshoot_rate_max":0.10,"closure_violation_rate_max":0.13,"inner_violation_rate_max":0.09,"init_feasible_rate_min":0.955},
        ),
        _gun_stage(
            5,"trail_high_bank",30.0,[1200.0,2300.0],[0.0,11.0],[0.0,50.0],
            _gun_target_mix(("constant_turn",0.45),("jink",0.35),("defensive",0.20)),
            phi_scale=0.009,inner_soft_m=260.0,
            own_speed=[255.0,310.0],target_speed=[236.0,282.0],
            target_bank=[16.0,42.0],altitude_offset=[-80.0,80.0],
            own_roll_jitter=1.0,own_pitch_jitter=0.75,
            easy_fraction=0.24,boundary_fraction=0.09,
            reward_override=_track_reward(trail_m=1550.0,scale=0.095),
            advance={"own_crash_rate_max":0.03,"track_score_min":0.21,"overshoot_rate_max":0.095,"closure_violation_rate_max":0.12,"inner_violation_rate_max":0.08,"init_feasible_rate_min":0.945},
        ),
        _gun_stage(
            6,"trail_reacquire_loiter",35.0,[1200.0,2700.0],[0.0,17.0],[0.0,70.0],
            _gun_target_mix(("constant_turn",0.35),("jink",0.35),("defensive",0.25),("shooter",0.05)),
            phi_scale=0.008,inner_soft_m=255.0,
            own_speed=[258.0,315.0],target_speed=[234.0,284.0],
            target_bank=[20.0,48.0],altitude_offset=[-130.0,130.0],
            own_roll_jitter=1.4,own_pitch_jitter=1.0,
            easy_fraction=0.20,boundary_fraction=0.09,
            reward_override=_track_reward(trail_m=1700.0,scale=0.090),
            advance={"own_crash_rate_max":0.028,"track_score_min":0.19,"overshoot_rate_max":0.09,"closure_violation_rate_max":0.11,"inner_violation_rate_max":0.07,"init_feasible_rate_min":0.935},
        ),
        _gun_stage(
            7,"trail_aggressive_loiter",40.0,[1300.0,3200.0],[0.0,25.0],[0.0,90.0],
            _gun_target_mix(("constant_turn",0.25),("jink",0.35),("defensive",0.30),("shooter",0.10)),
            phi_scale=0.007,inner_soft_m=250.0,
            own_speed=[260.0,320.0],target_speed=[232.0,286.0],
            target_bank=[24.0,55.0],altitude_offset=[-180.0,180.0],
            own_roll_jitter=1.8,own_pitch_jitter=1.3,
            easy_fraction=0.18,boundary_fraction=0.08,
            reward_override=_track_reward(trail_m=1900.0,scale=0.085),
            advance={"own_crash_rate_max":0.025,"track_score_min":0.17,"overshoot_rate_max":0.085,"closure_violation_rate_max":0.10,"inner_violation_rate_max":0.06,"init_feasible_rate_min":0.925},
        ),
        _gun_stage(
            8,"gun_trigger_static",4.0,[550.0,720.0],[0.0,0.4],[0.0,4.0],
            _gun_target_mix(("straight",1.0)),
            phi_scale=0.105,inner_soft_m=370.0,
            own_speed=[288.0,306.0],target_speed=[248.0,260.0],
            altitude_offset=[-1.0,1.0],own_roll_jitter=0.10,own_pitch_jitter=0.08,
            easy_fraction=0.40,boundary_fraction=0.03,
            advance={"episodes_min":16,"ep_wez_steps_min":5.0,"target_damage_min":0.08,"own_crash_rate_max":0.04,"inner_violation_rate_max":0.12,"init_feasible_rate_min":0.99},
        ),
        _gun_stage(
            9,"gun_wez_hold",6.0,[450.0,850.0],[0.0,0.8],[0.0,10.0],
            _gun_target_mix(("straight",1.0)),
            phi_scale=0.085,inner_soft_m=350.0,
            own_speed=[275.0,315.0],target_speed=[245.0,270.0],
            altitude_offset=[-5.0,5.0],own_roll_jitter=0.25,own_pitch_jitter=0.20,
            easy_fraction=0.30,boundary_fraction=0.10,
            advance={"ep_wez_steps_min":7.0,"target_damage_min":0.14,"own_crash_rate_max":0.03,"inner_violation_rate_max":0.08,"init_feasible_rate_min":0.98},
        ),
        _gun_stage(
            10,"gun_small_reacquire",8.0,[350.0,950.0],[0.0,2.5],[0.0,22.0],
            _gun_target_mix(("straight",0.80),("weak_turn",0.20)),
            phi_scale=0.073,inner_soft_m=345.0,
            own_speed=[270.0,320.0],target_speed=[240.0,275.0],
            target_bank=[6.0,16.0],altitude_offset=[-12.0,12.0],
            own_roll_jitter=0.55,own_pitch_jitter=0.45,
            easy_fraction=0.25,boundary_fraction=0.14,
            advance={"ep_wez_steps_min":6.0,"target_damage_min":0.16,"own_crash_rate_max":0.03,"inner_violation_rate_max":0.06,"init_feasible_rate_min":0.97},
        ),
        _gun_stage(
            11,"gun_near_track",12.0,[400.0,1250.0],[0.0,5.0],[0.0,40.0],
            _gun_target_mix(("straight",0.45),("weak_turn",0.35),("jink",0.20)),
            phi_scale=0.058,inner_soft_m=335.0,
            own_speed=[270.0,323.0],target_speed=[240.0,275.0],
            target_bank=[10.0,26.0],altitude_offset=[-30.0,30.0],
            own_roll_jitter=1.0,own_pitch_jitter=0.8,
            easy_fraction=0.22,boundary_fraction=0.13,
            advance={"target_damage_min":0.17,"inner_violation_rate_max":0.05,"init_feasible_rate_min":0.96},
        ),
        _gun_stage(
            12,"gun_outer_entry",16.0,[750.0,1800.0],[0.0,12.0],[5.0,70.0],
            _gun_target_mix(("straight",0.25),("weak_turn",0.15),("constant_turn",0.40),("jink",0.15),("defensive",0.05)),
            phi_scale=0.040,inner_soft_m=315.0,
            own_speed=[274.0,326.0],target_speed=[236.0,272.0],
            target_bank=[14.0,36.0],altitude_offset=[-80.0,80.0],
            own_roll_jitter=1.6,own_pitch_jitter=1.1,
            easy_fraction=0.19,boundary_fraction=0.12,
            advance={"target_damage_min":0.14,"bad_3_9_rate_max":0.05,"red_wez_rate_max":0.13,"init_feasible_rate_min":0.95},
        ),
        _gun_stage(
            13,"gun_side_rear_entry",20.0,[800.0,2600.0],[0.0,20.0],[15.0,115.0],
            _gun_target_mix(("straight",0.15),("constant_turn",0.35),("defensive",0.20),("jink",0.25),("shooter",0.05)),
            phi_scale=0.028,inner_soft_m=300.0,
            own_speed=[278.0,330.0],target_speed=[234.0,270.0],
            target_bank=[20.0,44.0],altitude_offset=[-140.0,140.0],
            own_roll_jitter=2.2,own_pitch_jitter=1.6,
            easy_fraction=0.17,boundary_fraction=0.11,
            advance={"target_damage_min":0.12,"bad_3_9_rate_max":0.05,"red_wez_rate_max":0.12,"init_feasible_rate_min":0.94},
        ),
        _gun_stage(
            14,"gun_three_nine_intro",28.0,[900.0,3300.0],[0.0,35.0],[25.0,155.0],
            _gun_target_mix(("constant_turn",0.30),("defensive",0.30),("shooter",0.15),("jink",0.25)),
            phi_scale=0.018,inner_soft_m=290.0,
            own_speed=[280.0,330.0],target_speed=[232.0,270.0],
            target_bank=[24.0,50.0],altitude_offset=[-220.0,220.0],
            own_roll_jitter=2.8,own_pitch_jitter=1.9,
            easy_fraction=0.14,boundary_fraction=0.10,
            advance={"win_rate_min":0.08,"target_damage_min":0.06,"bad_3_9_rate_max":0.05,"inner_violation_rate_max":0.038,"red_wez_rate_max":0.12,"init_feasible_rate_min":0.93},
        ),
        _gun_stage(
            15,"gun_three_nine_management",35.0,[900.0,3800.0],[0.0,45.0],[30.0,170.0],
            _gun_target_mix(("defensive",0.35),("shooter",0.20),("constant_turn",0.25),("jink",0.20)),
            phi_scale=0.014,inner_soft_m=285.0,
            own_speed=[280.0,332.0],target_speed=[232.0,270.0],
            target_bank=[26.0,52.0],altitude_offset=[-260.0,260.0],
            own_roll_jitter=3.0,own_pitch_jitter=2.0,
            easy_fraction=0.13,boundary_fraction=0.09,
            advance={"win_rate_min":0.12,"target_damage_min":0.05,"bad_3_9_rate_max":0.05,"inner_violation_rate_max":0.035,"red_wez_rate_max":0.12,"init_feasible_rate_min":0.925},
        ),
        _gun_stage(
            16,"gun_defensive_shooter_mixed",45.0,[1000.0,4500.0],[0.0,60.0],[35.0,180.0],
            _gun_target_mix(("defensive",0.35),("shooter",0.20),("constant_turn",0.20),("jink",0.20),("bt",0.05)),
            phi_scale=0.011,inner_soft_m=270.0,
            own_speed=[282.0,335.0],target_speed=[230.0,270.0],
            target_bank=[30.0,56.0],altitude_offset=[-330.0,330.0],
            own_roll_jitter=3.5,own_pitch_jitter=2.3,
            easy_fraction=0.12,boundary_fraction=0.09,
            advance={"win_rate_min":0.15,"bad_3_9_rate_max":0.05,"inner_violation_rate_max":0.032,"red_wez_rate_max":0.12,"init_feasible_rate_min":0.92},
        ),
        _gun_stage(
            17,"gun_short_bfm",60.0,[1200.0,5500.0],[0.0,90.0],[45.0,180.0],
            _gun_target_mix(("defensive",0.30),("shooter",0.25),("bt",0.20),("constant_turn",0.15),("jink",0.10)),
            phi_scale=0.007,inner_soft_m=252.0,
            own_speed=[282.0,335.0],target_speed=[230.0,275.0],
            target_bank=[32.0,60.0],altitude_offset=[-480.0,480.0],
            own_roll_jitter=4.0,own_pitch_jitter=2.5,
            easy_fraction=0.11,boundary_fraction=0.08,
            advance={"win_rate_min":0.22,"bad_3_9_rate_max":0.05,"inner_violation_rate_max":0.03,"red_wez_rate_max":0.13,"init_feasible_rate_min":0.91},
        ),
        _gun_stage(
            18,"gun_bt_mixed",75.0,[1000.0,6500.0],[0.0,110.0],[45.0,180.0],
            _gun_target_mix(("bt",0.35),("defensive",0.25),("shooter",0.15),("constant_turn",0.10),("jink",0.15)),
            phi_scale=0.0055,inner_soft_m=245.0,
            own_speed=[282.0,335.0],target_speed=[230.0,275.0],
            target_bank=[34.0,62.0],altitude_offset=[-620.0,620.0],
            own_roll_jitter=4.5,own_pitch_jitter=2.8,
            easy_fraction=0.10,boundary_fraction=0.07,
            advance={"win_rate_min":0.25,"bad_3_9_rate_max":0.055,"inner_violation_rate_max":0.028,"red_wez_rate_max":0.14,"init_feasible_rate_min":0.905},
        ),
        _gun_stage(
            19,"gun_league_feasible",90.0,[1000.0,7000.0],[0.0,130.0],[50.0,180.0],
            _gun_target_mix(("bt",0.45),("defensive",0.25),("shooter",0.15),("jink",0.15)),
            phi_scale=0.004,inner_soft_m=240.0,
            own_speed=[282.0,335.0],target_speed=[230.0,280.0],
            target_bank=[35.0,62.0],altitude_offset=[-800.0,800.0],
            own_roll_jitter=5.0,own_pitch_jitter=3.0,
            easy_fraction=0.10,boundary_fraction=0.06,
            advance={"win_rate_min":0.30,"bad_3_9_rate_max":0.06,"inner_violation_rate_max":0.025,"red_wez_rate_max":0.15,"init_feasible_rate_min":0.90},
        ),
    ]

def _with_bucket_gun_curriculum()->list[LoiterStage]:
    """Axis-separated gun curriculum with bucketized reset distributions.

    This schedule keeps the real tight WEZ fixed, but separates the hard parts
    of the reset distribution: range/closure, pointing angle, aspect, target
    maneuver, and finally BT/BFM pressure.  Each macro stage keeps an anchor
    bucket alive while adding one or two perturbation buckets.
    """
    stages=[
        _gun_stage(
            0,"T0_anchor_trail_hold",5.0,[820.0,880.0],[0.0,0.2],[0.0,1.0],
            _gun_target_mix(("straight",1.0)),
            phi_scale=0.010,inner_soft_m=300.0,
            own_speed=[268.0,276.0],target_speed=[268.0,276.0],
            target_bank=[0.0,0.0],altitude_offset=[-1.0,1.0],
            own_roll_jitter=0.02,own_pitch_jitter=0.02,
            easy_fraction=0.85,boundary_fraction=0.00,
            reward_override=_track_reward(trail_m=850.0,scale=0.13,phi_scale=0.004,inner_soft_m=300.0),
            bucket_mix=[
                _bucket("anchor",0.85,distance_m=[840.0,860.0],ata_deg=[0.0,0.10],aa_tail_deg=[0.0,0.50],speed_mps=[270.0,274.0],dv_mps=[-2.0,2.0],altitude_offset_m=[-1.0,1.0],loiter_bank_abs_deg=[0.0,0.0],target_policy="straight"),
                _bucket("range_only",0.05,distance_m=[780.0,940.0],ata_deg=[0.0,0.10],aa_tail_deg=[0.0,0.50],speed_mps=[270.0,274.0],dv_mps=[-2.0,2.0],target_policy="straight"),
                _bucket("speed_only",0.05,distance_m=[840.0,860.0],ata_deg=[0.0,0.10],aa_tail_deg=[0.0,0.50],speed_mps=[270.0,274.0],dv_mps=[-8.0,10.0],target_policy="straight"),
                _bucket("angle_only",0.05,distance_m=[840.0,860.0],ata_deg=[0.0,0.50],aa_tail_deg=[0.0,2.0],speed_mps=[270.0,274.0],dv_mps=[-2.0,2.0],target_policy="straight"),
            ],
            advance={"episodes_min":30,"track_score_min":0.50,"bucket_anchor_track_score_min":0.60,"bucket_worst_track_score_min":0.28,"overshoot_rate_max":0.04,"closure_violation_rate_max":0.08,"inner_violation_rate_max":0.04,"own_crash_rate_max":0.01,"init_feasible_rate_min":0.995},
        ),
        _gun_stage(
            1,"T1_range_closure_trail",8.0,[730.0,1050.0],[0.0,0.2],[0.0,1.0],
            _gun_target_mix(("straight",1.0)),
            phi_scale=0.010,inner_soft_m=295.0,
            own_speed=[260.0,288.0],target_speed=[268.0,276.0],
            target_bank=[0.0,0.0],altitude_offset=[-2.0,2.0],
            own_roll_jitter=0.04,own_pitch_jitter=0.04,
            easy_fraction=0.70,boundary_fraction=0.02,
            reward_override=_track_reward(trail_m=880.0,scale=0.125,phi_scale=0.004,inner_soft_m=295.0),
            bucket_mix=[
                _bucket("anchor",0.60,distance_m=[840.0,880.0],ata_deg=[0.0,0.15],aa_tail_deg=[0.0,1.0],speed_mps=[270.0,276.0],dv_mps=[-2.0,2.0],target_policy="straight"),
                _bucket("range_short",0.10,distance_m=[730.0,820.0],ata_deg=[0.0,0.15],aa_tail_deg=[0.0,1.0],speed_mps=[270.0,276.0],dv_mps=[-4.0,4.0],target_policy="straight"),
                _bucket("range_long",0.10,distance_m=[920.0,1050.0],ata_deg=[0.0,0.15],aa_tail_deg=[0.0,1.0],speed_mps=[270.0,276.0],dv_mps=[0.0,10.0],target_policy="straight"),
                _bucket("closing_fast",0.10,distance_m=[840.0,900.0],ata_deg=[0.0,0.15],aa_tail_deg=[0.0,1.0],speed_mps=[268.0,274.0],dv_mps=[10.0,18.0],target_policy="straight"),
                _bucket("opening_slow",0.10,distance_m=[840.0,900.0],ata_deg=[0.0,0.15],aa_tail_deg=[0.0,1.0],speed_mps=[270.0,278.0],dv_mps=[-10.0,-5.0],target_policy="straight"),
            ],
            advance={"track_score_min":0.45,"bucket_anchor_track_score_min":0.58,"bucket_worst_track_score_min":0.24,"overshoot_rate_max":0.06,"closure_violation_rate_max":0.10,"inner_violation_rate_max":0.05,"init_feasible_rate_min":0.99},
        ),
        _gun_stage(
            2,"T2_angular_trail",10.0,[820.0,950.0],[0.0,0.8],[0.0,4.0],
            _gun_target_mix(("straight",1.0)),
            phi_scale=0.011,inner_soft_m=292.0,
            own_speed=[265.0,282.0],target_speed=[268.0,276.0],
            target_bank=[0.0,0.0],altitude_offset=[-3.0,3.0],
            own_roll_jitter=0.08,own_pitch_jitter=0.06,
            easy_fraction=0.65,boundary_fraction=0.03,
            reward_override=_track_reward(trail_m=900.0,scale=0.12,phi_scale=0.006,inner_soft_m=292.0),
            bucket_mix=[
                _bucket("anchor",0.50,distance_m=[850.0,900.0],ata_deg=[0.0,0.15],aa_tail_deg=[0.0,1.0],speed_mps=[270.0,276.0],dv_mps=[-3.0,3.0],target_policy="straight"),
                _bucket("ata_left",0.10,distance_m=[850.0,920.0],ata_deg=[0.20,0.80],aa_tail_deg=[0.0,1.0],ata_sign=-1,target_policy="straight"),
                _bucket("ata_right",0.10,distance_m=[850.0,920.0],ata_deg=[0.20,0.80],aa_tail_deg=[0.0,1.0],ata_sign=1,target_policy="straight"),
                _bucket("aa_left",0.10,distance_m=[850.0,920.0],ata_deg=[0.0,0.20],aa_tail_deg=[1.0,4.0],aa_sign=-1,target_policy="straight"),
                _bucket("aa_right",0.10,distance_m=[850.0,920.0],ata_deg=[0.0,0.20],aa_tail_deg=[1.0,4.0],aa_sign=1,target_policy="straight"),
                _bucket("combined_small",0.10,distance_m=[830.0,950.0],ata_deg=[0.20,0.80],aa_tail_deg=[1.0,4.0],speed_mps=[268.0,276.0],dv_mps=[-5.0,5.0],target_policy="straight"),
            ],
            advance={"track_score_min":0.40,"bucket_anchor_track_score_min":0.55,"bucket_worst_track_score_min":0.20,"overshoot_rate_max":0.08,"closure_violation_rate_max":0.12,"inner_violation_rate_max":0.06,"init_feasible_rate_min":0.985},
        ),
        _gun_stage(
            3,"T3_weak_turn_trail",13.0,[800.0,1100.0],[0.0,1.5],[0.0,8.0],
            _gun_target_mix(("straight",0.60),("weak_turn",0.40)),
            phi_scale=0.012,inner_soft_m=288.0,
            own_speed=[262.0,292.0],target_speed=[265.0,278.0],
            target_bank=[2.0,8.0],altitude_offset=[-5.0,5.0],
            own_roll_jitter=0.15,own_pitch_jitter=0.10,
            easy_fraction=0.55,boundary_fraction=0.04,
            reward_override=_track_reward(trail_m=930.0,scale=0.115,phi_scale=0.008,inner_soft_m=288.0),
            bucket_mix=[
                _bucket("anchor",0.35,distance_m=[850.0,920.0],ata_deg=[0.0,0.2],aa_tail_deg=[0.0,1.5],loiter_bank_abs_deg=[0.0,0.0],target_policy="straight"),
                _bucket("range_closure",0.20,distance_m=[780.0,1120.0],ata_deg=[0.0,0.3],aa_tail_deg=[0.0,2.0],dv_mps=[-8.0,12.0],target_policy="straight"),
                _bucket("angle",0.20,distance_m=[850.0,980.0],ata_deg=[0.2,1.5],aa_tail_deg=[1.0,8.0],dv_mps=[-4.0,6.0],target_policy="straight"),
                _bucket("weak_turn_easy",0.20,distance_m=[850.0,1050.0],ata_deg=[0.0,0.8],aa_tail_deg=[0.0,5.0],loiter_bank_abs_deg=[2.0,8.0],target_policy="weak_turn"),
                _bucket("boundary",0.05,distance_m=[1030.0,1150.0],ata_deg=[1.0,1.8],aa_tail_deg=[5.0,10.0],loiter_bank_abs_deg=[6.0,10.0],target_policy="weak_turn"),
            ],
            advance={"track_score_min":0.35,"bucket_worst_track_score_min":0.18,"overshoot_rate_max":0.09,"closure_violation_rate_max":0.13,"inner_violation_rate_max":0.07,"init_feasible_rate_min":0.975},
        ),
        _gun_stage(
            4,"T4_maneuver_trail",18.0,[850.0,1300.0],[0.0,3.0],[0.0,15.0],
            _gun_target_mix(("straight",0.25),("weak_turn",0.35),("constant_turn",0.30),("jink",0.10)),
            phi_scale=0.012,inner_soft_m=282.0,
            own_speed=[260.0,302.0],target_speed=[260.0,280.0],
            target_bank=[5.0,18.0],altitude_offset=[-15.0,15.0],
            own_roll_jitter=0.30,own_pitch_jitter=0.20,
            easy_fraction=0.45,boundary_fraction=0.05,
            reward_override=_track_reward(trail_m=1050.0,scale=0.11,phi_scale=0.009,inner_soft_m=282.0),
            bucket_mix=[
                _bucket("anchor",0.30,distance_m=[900.0,1000.0],ata_deg=[0.0,0.4],aa_tail_deg=[0.0,3.0],loiter_bank_abs_deg=[0.0,4.0],target_policy_mix=_gun_target_mix(("straight",0.70),("weak_turn",0.30))),
                _bucket("weak_turn",0.25,distance_m=[850.0,1150.0],ata_deg=[0.0,1.5],aa_tail_deg=[0.0,8.0],loiter_bank_abs_deg=[5.0,12.0],target_policy="weak_turn"),
                _bucket("constant_turn",0.20,distance_m=[900.0,1250.0],ata_deg=[0.5,2.5],aa_tail_deg=[3.0,12.0],loiter_bank_abs_deg=[10.0,18.0],target_policy="constant_turn"),
                _bucket("mild_jink",0.15,distance_m=[900.0,1300.0],ata_deg=[0.5,3.0],aa_tail_deg=[3.0,15.0],loiter_bank_abs_deg=[8.0,16.0],target_policy="jink"),
                _bucket("boundary",0.10,distance_m=[1200.0,1400.0],ata_deg=[2.0,4.0],aa_tail_deg=[10.0,18.0],loiter_bank_abs_deg=[14.0,22.0],target_policy_mix=_gun_target_mix(("constant_turn",0.60),("jink",0.40))),
            ],
            advance={"track_score_min":0.30,"bucket_worst_track_score_min":0.15,"overshoot_rate_max":0.10,"closure_violation_rate_max":0.14,"inner_violation_rate_max":0.08,"init_feasible_rate_min":0.965},
        ),
        _gun_stage(
            5,"A0_nose_on_trail_bridge",6.0,[650.0,900.0],[0.0,1.2],[0.0,10.0],
            _gun_target_mix(("straight",0.80),("weak_turn",0.20)),
            phi_scale=0.070,inner_soft_m=305.0,
            own_speed=[266.0,288.0],target_speed=[264.0,276.0],
            target_bank=[0.0,8.0],altitude_offset=[-4.0,4.0],
            own_roll_jitter=0.08,own_pitch_jitter=0.06,
            easy_fraction=0.55,boundary_fraction=0.04,
            reward_override=_nose_bridge_reward(trail_m=760.0,phi_scale=0.070,inner_soft_m=305.0),
            bucket_mix=[
                _bucket("nose_center",0.40,distance_m=[650.0,800.0],ata_deg=[0.0,0.60],aa_tail_deg=[0.0,5.0],dv_mps=[-3.0,5.0],target_policy="straight"),
                _bucket("trail_to_gun",0.25,distance_m=[760.0,930.0],ata_deg=[0.0,1.00],aa_tail_deg=[0.0,8.0],dv_mps=[0.0,9.0],target_policy="straight"),
                _bucket("weak_turn_nose",0.20,distance_m=[720.0,920.0],ata_deg=[0.0,1.20],aa_tail_deg=[0.0,10.0],loiter_bank_abs_deg=[2.0,8.0],target_policy="weak_turn"),
                _bucket("near_boundary",0.10,distance_m=[500.0,650.0],ata_deg=[0.0,0.80],aa_tail_deg=[0.0,8.0],dv_mps=[-5.0,3.0],target_policy="straight"),
                _bucket("ata_fixup",0.05,distance_m=[650.0,850.0],ata_deg=[1.0,2.0],aa_tail_deg=[0.0,12.0],target_policy_mix=_gun_target_mix(("straight",0.70),("weak_turn",0.30))),
            ],
            advance={"ep_wez_steps_min":2.5,"bucket_worst_ep_wez_steps_min":0.5,"target_damage_min":0.025,"track_score_min":0.35,"bucket_worst_track_score_min":0.10,"inner_violation_rate_max":0.04,"red_wez_rate_max":0.07,"own_crash_rate_max":0.025,"init_feasible_rate_min":0.985},
        ),
        _gun_stage(
            5,"G0_in_wez_hold",4.0,[550.0,700.0],[0.0,0.2],[0.0,2.0],
            _gun_target_mix(("straight",1.0)),
            phi_scale=0.110,inner_soft_m=310.0,
            own_speed=[270.0,282.0],target_speed=[268.0,276.0],
            target_bank=[0.0,0.0],altitude_offset=[-2.0,2.0],
            own_roll_jitter=0.05,own_pitch_jitter=0.05,
            easy_fraction=0.65,boundary_fraction=0.02,
            bucket_mix=[
                _bucket("perfect_wez",0.70,distance_m=[560.0,690.0],ata_deg=[0.0,0.20],aa_tail_deg=[0.0,2.0],dv_mps=[-3.0,5.0],target_policy="straight"),
                _bucket("slightly_far",0.10,distance_m=[720.0,860.0],ata_deg=[0.0,0.20],aa_tail_deg=[0.0,2.0],dv_mps=[0.0,8.0],target_policy="straight"),
                _bucket("slightly_near",0.10,distance_m=[320.0,460.0],ata_deg=[0.0,0.20],aa_tail_deg=[0.0,2.0],dv_mps=[-5.0,3.0],target_policy="straight"),
                _bucket("slight_ata",0.10,distance_m=[560.0,760.0],ata_deg=[0.4,0.9],aa_tail_deg=[0.0,4.0],dv_mps=[-3.0,5.0],target_policy="straight"),
            ],
            advance={"ep_wez_steps_min":8.0,"bucket_worst_ep_wez_steps_min":3.0,"target_damage_min":0.08,"bucket_worst_target_damage_min":0.02,"inner_violation_rate_max":0.03,"red_wez_rate_max":0.07,"own_crash_rate_max":0.02,"init_feasible_rate_min":0.99},
        ),
        _gun_stage(
            6,"G1_wez_boundary_hold",6.0,[260.0,930.0],[0.0,1.2],[0.0,5.0],
            _gun_target_mix(("straight",0.90),("weak_turn",0.10)),
            phi_scale=0.090,inner_soft_m=285.0,
            own_speed=[268.0,292.0],target_speed=[264.0,278.0],
            target_bank=[0.0,6.0],altitude_offset=[-5.0,5.0],
            own_roll_jitter=0.12,own_pitch_jitter=0.08,
            easy_fraction=0.45,boundary_fraction=0.10,
            bucket_mix=[
                _bucket("center_wez",0.45,distance_m=[550.0,750.0],ata_deg=[0.0,0.30],aa_tail_deg=[0.0,5.0],target_policy="straight"),
                _bucket("far_edge",0.20,distance_m=[820.0,930.0],ata_deg=[0.0,0.30],aa_tail_deg=[0.0,5.0],dv_mps=[2.0,12.0],target_policy="straight"),
                _bucket("ata_boundary",0.20,distance_m=[550.0,800.0],ata_deg=[0.70,1.20],aa_tail_deg=[0.0,5.0],target_policy="straight"),
                _bucket("inner_safe_edge",0.10,distance_m=[260.0,380.0],ata_deg=[0.0,0.30],aa_tail_deg=[0.0,5.0],dv_mps=[-8.0,2.0],target_policy="straight"),
                _bucket("combined_edge",0.05,distance_m=[800.0,940.0],ata_deg=[0.70,1.30],aa_tail_deg=[2.0,6.0],target_policy_mix=_gun_target_mix(("straight",0.80),("weak_turn",0.20))),
            ],
            advance={"ep_wez_steps_min":7.0,"bucket_worst_ep_wez_steps_min":2.0,"target_damage_min":0.12,"bucket_worst_target_damage_min":0.015,"inner_violation_rate_max":0.04,"red_wez_rate_max":0.08,"init_feasible_rate_min":0.98},
        ),
        _gun_stage(
            7,"G2_small_reacquire",9.0,[450.0,1000.0],[0.0,2.5],[0.0,20.0],
            _gun_target_mix(("straight",0.70),("weak_turn",0.20),("jink",0.10)),
            phi_scale=0.075,inner_soft_m=285.0,
            own_speed=[265.0,305.0],target_speed=[258.0,278.0],
            target_bank=[0.0,16.0],altitude_offset=[-12.0,12.0],
            own_roll_jitter=0.35,own_pitch_jitter=0.22,
            easy_fraction=0.35,boundary_fraction=0.10,
            bucket_mix=[
                _bucket("already_wez",0.35,distance_m=[500.0,820.0],ata_deg=[0.0,0.5],aa_tail_deg=[0.0,5.0],target_policy="straight"),
                _bucket("far_reacquire",0.20,distance_m=[850.0,1050.0],ata_deg=[0.0,1.0],aa_tail_deg=[0.0,8.0],dv_mps=[4.0,15.0],target_policy="straight"),
                _bucket("ata_reacquire",0.20,distance_m=[520.0,900.0],ata_deg=[1.0,2.5],aa_tail_deg=[0.0,8.0],target_policy="straight"),
                _bucket("aa_reacquire",0.10,distance_m=[520.0,900.0],ata_deg=[0.0,1.2],aa_tail_deg=[8.0,22.0],target_policy="straight"),
                _bucket("weak_turn",0.10,distance_m=[520.0,950.0],ata_deg=[0.0,1.8],aa_tail_deg=[0.0,16.0],loiter_bank_abs_deg=[6.0,16.0],target_policy="weak_turn"),
                _bucket("boundary_combo",0.05,distance_m=[900.0,1100.0],ata_deg=[1.8,3.0],aa_tail_deg=[15.0,25.0],loiter_bank_abs_deg=[10.0,20.0],target_policy_mix=_gun_target_mix(("weak_turn",0.70),("jink",0.30))),
            ],
            advance={"ep_wez_steps_min":5.0,"target_damage_min":0.12,"bucket_worst_target_damage_min":0.005,"inner_violation_rate_max":0.05,"red_wez_rate_max":0.10,"init_feasible_rate_min":0.965},
        ),
        _gun_stage(
            8,"G3_moving_gun_track",13.0,[450.0,1200.0],[0.0,5.0],[0.0,35.0],
            _gun_target_mix(("straight",0.40),("weak_turn",0.35),("jink",0.15),("constant_turn",0.10)),
            phi_scale=0.058,inner_soft_m=275.0,
            own_speed=[265.0,315.0],target_speed=[252.0,280.0],
            target_bank=[5.0,22.0],altitude_offset=[-30.0,30.0],
            own_roll_jitter=0.65,own_pitch_jitter=0.40,
            easy_fraction=0.28,boundary_fraction=0.10,
            bucket_mix=[
                _bucket("straight_wez",0.25,distance_m=[520.0,850.0],ata_deg=[0.0,1.0],aa_tail_deg=[0.0,8.0],target_policy="straight"),
                _bucket("weak_turn_center",0.25,distance_m=[560.0,950.0],ata_deg=[0.0,2.0],aa_tail_deg=[0.0,16.0],loiter_bank_abs_deg=[5.0,14.0],target_policy="weak_turn"),
                _bucket("angle_reacquire",0.20,distance_m=[600.0,1050.0],ata_deg=[2.0,5.0],aa_tail_deg=[8.0,28.0],target_policy_mix=_gun_target_mix(("straight",0.50),("weak_turn",0.50))),
                _bucket("range_reacquire",0.15,distance_m=[950.0,1250.0],ata_deg=[0.0,2.0],aa_tail_deg=[0.0,20.0],dv_mps=[5.0,18.0],target_policy_mix=_gun_target_mix(("straight",0.50),("weak_turn",0.50))),
                _bucket("mild_jink",0.10,distance_m=[650.0,1150.0],ata_deg=[0.5,4.0],aa_tail_deg=[8.0,30.0],loiter_bank_abs_deg=[10.0,22.0],target_policy="jink"),
                _bucket("boundary_combo",0.05,distance_m=[1000.0,1300.0],ata_deg=[4.0,6.0],aa_tail_deg=[25.0,38.0],loiter_bank_abs_deg=[16.0,26.0],target_policy_mix=_gun_target_mix(("jink",0.50),("constant_turn",0.50))),
            ],
            advance={"target_damage_min":0.14,"bucket_worst_target_damage_min":0.005,"inner_violation_rate_max":0.05,"bad_3_9_rate_max":0.04,"red_wez_rate_max":0.11,"init_feasible_rate_min":0.95},
        ),
        _gun_stage(
            9,"E0_outer_rear_entry",18.0,[800.0,1800.0],[0.0,10.0],[5.0,70.0],
            _gun_target_mix(("straight",0.25),("weak_turn",0.25),("constant_turn",0.35),("jink",0.15)),
            phi_scale=0.040,inner_soft_m=265.0,
            own_speed=[270.0,322.0],target_speed=[245.0,278.0],
            target_bank=[10.0,35.0],altitude_offset=[-80.0,80.0],
            own_roll_jitter=1.0,own_pitch_jitter=0.75,
            easy_fraction=0.22,boundary_fraction=0.09,
            bucket_mix=[
                _bucket("easy_rear",0.30,distance_m=[800.0,1300.0],ata_deg=[0.0,4.0],aa_tail_deg=[5.0,35.0],target_policy_mix=_gun_target_mix(("straight",0.50),("weak_turn",0.50))),
                _bucket("range_outer",0.20,distance_m=[1300.0,1900.0],ata_deg=[0.0,6.0],aa_tail_deg=[10.0,45.0],dv_mps=[5.0,18.0],target_policy_mix=_gun_target_mix(("weak_turn",0.50),("constant_turn",0.50))),
                _bucket("ata_outer",0.15,distance_m=[900.0,1600.0],ata_deg=[5.0,11.0],aa_tail_deg=[10.0,50.0],target_policy_mix=_gun_target_mix(("weak_turn",0.40),("constant_turn",0.60))),
                _bucket("aa_outer",0.15,distance_m=[900.0,1700.0],ata_deg=[0.0,7.0],aa_tail_deg=[45.0,75.0],target_policy_mix=_gun_target_mix(("constant_turn",0.60),("jink",0.40))),
                _bucket("constant_turn",0.15,distance_m=[900.0,1800.0],ata_deg=[0.0,8.0],aa_tail_deg=[20.0,65.0],loiter_bank_abs_deg=[18.0,35.0],target_policy="constant_turn"),
                _bucket("boundary_combo",0.05,distance_m=[1700.0,2000.0],ata_deg=[8.0,12.0],aa_tail_deg=[60.0,80.0],loiter_bank_abs_deg=[28.0,40.0],target_policy="jink"),
            ],
            advance={"target_damage_min":0.10,"inner_violation_rate_max":0.045,"bad_3_9_rate_max":0.045,"red_wez_rate_max":0.12,"init_feasible_rate_min":0.94},
        ),
        _gun_stage(
            10,"E1_side_rear_entry",24.0,[900.0,2600.0],[0.0,20.0],[15.0,115.0],
            _gun_target_mix(("straight",0.10),("weak_turn",0.10),("constant_turn",0.40),("jink",0.25),("defensive",0.15)),
            phi_scale=0.028,inner_soft_m=258.0,
            own_speed=[275.0,328.0],target_speed=[238.0,276.0],
            target_bank=[15.0,45.0],altitude_offset=[-150.0,150.0],
            own_roll_jitter=1.7,own_pitch_jitter=1.2,
            easy_fraction=0.18,boundary_fraction=0.09,
            bucket_mix=[
                _bucket("rear_quarter",0.30,distance_m=[900.0,1700.0],ata_deg=[0.0,10.0],aa_tail_deg=[15.0,65.0],target_policy_mix=_gun_target_mix(("weak_turn",0.30),("constant_turn",0.70))),
                _bucket("beam_soft",0.20,distance_m=[1000.0,2300.0],ata_deg=[5.0,20.0],aa_tail_deg=[65.0,115.0],target_policy_mix=_gun_target_mix(("constant_turn",0.50),("jink",0.50))),
                _bucket("outer_range",0.20,distance_m=[1800.0,2800.0],ata_deg=[0.0,14.0],aa_tail_deg=[25.0,90.0],dv_mps=[8.0,22.0],target_policy_mix=_gun_target_mix(("constant_turn",0.60),("jink",0.40))),
                _bucket("mild_defensive",0.15,distance_m=[1000.0,2300.0],ata_deg=[6.0,20.0],aa_tail_deg=[35.0,110.0],loiter_bank_abs_deg=[24.0,45.0],target_policy="defensive"),
                _bucket("jink",0.10,distance_m=[1000.0,2500.0],ata_deg=[5.0,18.0],aa_tail_deg=[35.0,110.0],loiter_bank_abs_deg=[20.0,45.0],target_policy="jink"),
                _bucket("boundary_combo",0.05,distance_m=[2300.0,2900.0],ata_deg=[16.0,24.0],aa_tail_deg=[100.0,125.0],loiter_bank_abs_deg=[35.0,50.0],target_policy_mix=_gun_target_mix(("jink",0.50),("defensive",0.50))),
            ],
            advance={"target_damage_min":0.08,"inner_violation_rate_max":0.04,"bad_3_9_rate_max":0.05,"red_wez_rate_max":0.12,"init_feasible_rate_min":0.925},
        ),
        _gun_stage(
            11,"E2_three_nine_management",32.0,[900.0,3300.0],[0.0,35.0],[25.0,155.0],
            _gun_target_mix(("constant_turn",0.30),("defensive",0.30),("jink",0.25),("shooter",0.15)),
            phi_scale=0.018,inner_soft_m=248.0,
            own_speed=[278.0,332.0],target_speed=[232.0,274.0],
            target_bank=[24.0,52.0],altitude_offset=[-250.0,250.0],
            own_roll_jitter=2.4,own_pitch_jitter=1.8,
            easy_fraction=0.14,boundary_fraction=0.08,
            bucket_mix=[
                _bucket("rear_entry",0.25,distance_m=[900.0,1900.0],ata_deg=[0.0,15.0],aa_tail_deg=[25.0,90.0],target_policy_mix=_gun_target_mix(("constant_turn",0.60),("jink",0.40))),
                _bucket("beam_entry",0.20,distance_m=[1200.0,2600.0],ata_deg=[10.0,30.0],aa_tail_deg=[75.0,135.0],target_policy_mix=_gun_target_mix(("constant_turn",0.40),("jink",0.60))),
                _bucket("near_3_9",0.20,distance_m=[1000.0,3000.0],ata_deg=[15.0,35.0],aa_tail_deg=[110.0,155.0],target_policy_mix=_gun_target_mix(("jink",0.50),("defensive",0.50))),
                _bucket("defensive_break",0.20,distance_m=[1000.0,3000.0],ata_deg=[5.0,30.0],aa_tail_deg=[45.0,150.0],target_policy="defensive"),
                _bucket("shooter_threat",0.10,distance_m=[1000.0,3200.0],ata_deg=[10.0,35.0],aa_tail_deg=[60.0,155.0],target_policy="shooter"),
                _bucket("hard_boundary",0.05,distance_m=[2800.0,3600.0],ata_deg=[30.0,42.0],aa_tail_deg=[145.0,180.0],target_policy_mix=_gun_target_mix(("defensive",0.50),("shooter",0.50))),
            ],
            advance={"win_rate_min":0.05,"target_damage_min":0.04,"bad_3_9_rate_max":0.05,"inner_violation_rate_max":0.035,"red_wez_rate_max":0.12,"target_crash_without_damage_rate_max":0.02,"init_feasible_rate_min":0.90},
        ),
        _gun_stage(
            12,"B0_bt_intro",45.0,[1000.0,4200.0],[0.0,60.0],[35.0,180.0],
            _gun_target_mix(("constant_turn",0.25),("jink",0.25),("defensive",0.25),("shooter",0.10),("bt",0.15)),
            phi_scale=0.010,inner_soft_m=240.0,
            own_speed=[280.0,335.0],target_speed=[230.0,276.0],
            target_bank=[28.0,58.0],altitude_offset=[-300.0,300.0],
            own_roll_jitter=3.2,own_pitch_jitter=2.2,
            easy_fraction=0.12,boundary_fraction=0.08,
            bucket_mix=[
                _bucket("known_easy",0.20,distance_m=[1000.0,2200.0],ata_deg=[0.0,25.0],aa_tail_deg=[35.0,120.0],target_policy_mix=_gun_target_mix(("constant_turn",0.50),("jink",0.50))),
                _bucket("constant_turn",0.20,distance_m=[1200.0,3400.0],ata_deg=[0.0,45.0],aa_tail_deg=[45.0,160.0],target_policy="constant_turn"),
                _bucket("defensive",0.20,distance_m=[1200.0,3800.0],ata_deg=[10.0,60.0],aa_tail_deg=[45.0,180.0],target_policy="defensive"),
                _bucket("shooter",0.10,distance_m=[1200.0,3800.0],ata_deg=[10.0,60.0],aa_tail_deg=[55.0,180.0],target_policy="shooter"),
                _bucket("bt_easy",0.20,distance_m=[1000.0,3500.0],ata_deg=[0.0,50.0],aa_tail_deg=[45.0,170.0],target_policy="bt"),
                _bucket("raw_opening",0.10,distance_m=[2500.0,4500.0],ata_deg=[35.0,70.0],aa_tail_deg=[120.0,180.0],target_policy_mix=_gun_target_mix(("bt",0.50),("defensive",0.50)),ensure_initial_feasible=False),
            ],
            advance={"win_rate_min":0.12,"target_damage_min":0.035,"bad_3_9_rate_max":0.06,"inner_violation_rate_max":0.03,"red_wez_rate_max":0.14,"init_feasible_rate_min":0.75},
        ),
        _gun_stage(
            13,"B1_short_bfm",70.0,[1200.0,5500.0],[0.0,90.0],[45.0,180.0],
            _gun_target_mix(("bt",0.40),("defensive",0.20),("shooter",0.15),("jink",0.15),("constant_turn",0.10)),
            phi_scale=0.006,inner_soft_m=235.0,
            own_speed=[280.0,338.0],target_speed=[228.0,280.0],
            target_bank=[32.0,62.0],altitude_offset=[-500.0,500.0],
            own_roll_jitter=4.2,own_pitch_jitter=2.8,
            easy_fraction=0.10,boundary_fraction=0.07,
            bucket_mix=[
                _bucket("bt_feasible",0.35,distance_m=[1200.0,4200.0],ata_deg=[0.0,75.0],aa_tail_deg=[45.0,180.0],target_policy="bt"),
                _bucket("defensive",0.20,distance_m=[1400.0,4800.0],ata_deg=[10.0,90.0],aa_tail_deg=[55.0,180.0],target_policy="defensive"),
                _bucket("shooter",0.15,distance_m=[1400.0,5000.0],ata_deg=[10.0,90.0],aa_tail_deg=[60.0,180.0],target_policy="shooter"),
                _bucket("jink",0.15,distance_m=[1300.0,4800.0],ata_deg=[5.0,80.0],aa_tail_deg=[45.0,180.0],target_policy="jink"),
                _bucket("constant_turn",0.10,distance_m=[1300.0,4600.0],ata_deg=[5.0,75.0],aa_tail_deg=[45.0,180.0],target_policy="constant_turn"),
                _bucket("raw_opening",0.05,distance_m=[3800.0,6000.0],ata_deg=[70.0,110.0],aa_tail_deg=[130.0,180.0],target_policy_mix=_gun_target_mix(("bt",0.60),("shooter",0.40)),ensure_initial_feasible=False),
            ],
            advance={"win_rate_min":0.20,"target_damage_min":0.025,"bad_3_9_rate_max":0.065,"inner_violation_rate_max":0.028,"red_wez_rate_max":0.15,"init_feasible_rate_min":0.65},
        ),
    ]
    return [_stage_copy(stage,index=i) for i,stage in enumerate(stages)]

def load_stages(stage_dir:Path|None=None,schedule:str|None=None)->list[LoiterStage]:
    directory=Path(stage_dir) if stage_dir else AIP_STAGE_DIR
    if not directory.is_dir():raise FileNotFoundError(f"AIP loiter stage directory not found: {directory}")
    stages=[]
    for i,filename in enumerate(FILES):
        path=directory/filename;raw=yaml.safe_load(path.read_text(encoding="utf-8"));env=raw["env"];cfg=raw["env_config"]
        stages.append(LoiterStage(i,str(raw["name"]),int(env["episode_step_limit"]),float(env["max_engage_time"]),int(cfg.get("step_ratio",6)),copy.deepcopy(cfg.get("ownship_randomization",{})),copy.deepcopy(cfg["target_randomization"]),copy.deepcopy(cfg["wez"]),copy.deepcopy(cfg["reward"]),_with_numeric_safety_conditions(CONDITIONS[i]),path))
    selected=(schedule or os.environ.get("LOITER_STAGE_SCHEDULE") or os.environ.get("STAGE_SCHEDULE") or "aip").strip().lower()
    if selected in {"","aip","default","original"}:
        return stages
    if selected in BRIDGE_SCHEDULE_NAMES:
        return _with_final_kill_bridge(stages)
    if selected in GUN_CURRICULUM_SCHEDULE_NAMES:
        return _with_gun_curriculum()
    if selected in BUCKET_GUN_CURRICULUM_SCHEDULE_NAMES:
        return _with_bucket_gun_curriculum()
    raise ValueError(f"Unknown loiter stage schedule: {selected!r}. Use 'aip', 'kill_bridge', 'gun_curriculum', or 'gun_bucket_curriculum'.")

def advancement_satisfied(stage:LoiterStage,metrics:dict[str,float])->tuple[bool,str]:
    reasons=[]
    for condition,threshold in stage.advance_conditions.items():
        if condition=="min_valid_samples":
            value=float(metrics.get("episodes",0.))
            if value<threshold:return False,f"episodes={value:.4g} < {threshold}"
            reasons.append(f"episodes={value:.4g}")
            continue
        key=condition.removesuffix("_min").removesuffix("_max");value=float(metrics.get(key,float("nan")))
        if condition.endswith("_min") and not value>=threshold:return False,f"{key}={value:.4g} < {threshold}"
        if condition.endswith("_max") and not value<=threshold:return False,f"{key}={value:.4g} > {threshold}"
        reasons.append(f"{key}={value:.4g}")
    return True,", ".join(reasons)

__all__=["LoiterStage","load_stages","advancement_satisfied"]
