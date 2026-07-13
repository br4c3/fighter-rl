#!/usr/bin/env python3
"""Render an offline HTML dashboard from Fighter RL training artifacts.

This tool is deliberately independent from the trainers and fighter_rl package.
It only reads JSON/JSONL files from a completed or running experiment directory.
"""

import argparse
import json
import math
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

DASHBOARD_FIELDS = {
    "stage",
    "stage_name",
    "update",
    "status",
    "valid_steps",
    "valid_count",
    "decision",
    "decision_limit",
    "reward_mean",
    "loss",
    "q_loss",
    "actor_loss",
    "alpha",
    "episodes",
    "gate",
    "pass_streak",
    "replay_valid",
    "replay_slots",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clip_fraction",
    "explained_variance",
    "gradient_norm",
    "action_abs_mean",
    "action_std",
    "action_saturation_rate",
    "mean_delta_action",
    "roll_command_mean",
    "pitch_command_mean",
    "rudder_command_mean",
    "throttle_command_mean",
    "win_rate",
    "timeout_rate",
    "crash_rate",
    "own_crash_rate",
    "target_crash_rate",
    "target_damage",
    "own_damage",
    "ep_wez_steps",
    "ep_wez_streak_max",
    "track_score",
    "bucket_worst_track_score",
    "closure_violation_rate",
    "closure_violation_step_fraction",
    "mean_abs_closing_mps",
    "trail_range_error_m",
    "opening_away_step_fraction",
    "final_ata_deg",
    "final_aa_deg",
    "inner_violation_rate",
    "bad_3_9_rate",
    "red_wez_rate",
    "overshoot_rate",
    "nonfinite_rate",
    "init_feasible_rate",
}


def read_json(path, default):
    if not path.is_file():
        return default

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def read_jsonl(path):
    records = []
    skipped = 0

    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            text = line.strip()

            if not text:
                continue

            try:
                item = json.loads(text)
            except json.JSONDecodeError:
                skipped += 1
                continue

            if isinstance(item, dict):
                item["_line"] = line_number
                records.append(item)
    return records, skipped


def finite_or_none(value):
    if isinstance(value, bool) or value is None:
        return value

    if isinstance(value, (int, float)):
        number = float(value)
        return value if math.isfinite(number) else None

    if isinstance(value, dict):
        return {str(key): finite_or_none(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [finite_or_none(item) for item in value]
    return value


def flatten_record(record, sequence):
    merged = {
        key: value for key, value in record.items() if key not in {"metrics", "ppo", "actions"}
    }

    for section in ("metrics", "ppo", "actions"):
        nested = record.get(section)

        if isinstance(nested, dict):
            merged.update(nested)
    flat = {key: merged[key] for key in DASHBOARD_FIELDS if key in merged}
    flat["sequence"] = sequence
    flat["stage"] = int(flat.get("stage", 0) or 0)
    flat["update"] = int(flat.get("update", 0) or 0)
    return finite_or_none(flat)


def evenly_spaced(items, limit):
    if len(items) <= limit:
        return items

    if limit <= 2:
        return [items[0], items[-1]][:limit]

    indices = {round(i * (len(items) - 1) / (limit - 1)) for i in range(limit)}
    return [item for index, item in enumerate(items) if index in indices]


def downsample(records, max_points):
    if len(records) <= max_points:
        return records

    groups = defaultdict(list)

    for record in records:
        groups[record["stage"]].append(record)

    sampled = []
    total = len(records)

    for stage in sorted(groups):
        group = groups[stage]
        allocation = max(2, round(max_points * len(group) / total))
        sampled.extend(evenly_spaced(group, allocation))
    sampled.sort(key=lambda item: item["sequence"])
    return evenly_spaced(sampled, max_points)


def resolve_run(path):
    candidate = Path(path).expanduser().resolve()

    if candidate.is_file():
        if candidate.name != "metrics.jsonl":
            raise ValueError(f"Expected metrics.jsonl, got: {candidate}")
        return candidate.parent

    direct = candidate / "metrics.jsonl"

    if direct.is_file():
        return candidate

    matches = list(candidate.glob("**/metrics.jsonl")) if candidate.is_dir() else []

    if not matches:
        raise FileNotFoundError(f"No metrics.jsonl found under: {candidate}")
    return max(matches, key=lambda item: item.stat().st_mtime).parent


def demo_payload():
    records = []
    sequence = 0

    for stage, name, updates, base_track, base_closure in (
        (0, "T0_anchor_trail_hold", 45, 0.70, 0.10),
        (1, "T1_range_closure_trail", 60, 0.62, 0.16),
        (2, "T2_angular_trail", 75, 0.56, 0.19),
        (3, "T3_weak_turn_trail", 110, 0.49, 0.24),
        (4, "T35_mild_maneuver_trail", 85, 0.40, 0.42),
    ):
        for update in range(1, updates + 1):
            sequence += 1
            progress = update / updates
            records.append(
                {
                    "sequence": sequence,
                    "stage": stage,
                    "stage_name": name,
                    "update": update,
                    "valid_steps": sequence * 65536,
                    "reward_mean": -0.08 + 0.18 * progress - stage * 0.012,
                    "track_score": base_track + 0.13 * progress,
                    "closure_violation_rate": max(0.05, base_closure - 0.12 * progress),
                    "trail_range_error_m": 190 - 80 * progress + stage * 8,
                    "final_ata_deg": 78 - 25 * progress + stage * 2,
                    "loss": 3.0 + stage + (1 - progress) * 4,
                    "policy_loss": -0.02 + 0.015 * progress,
                    "value_loss": 5.0 + stage * 1.5 - progress * 2,
                    "entropy": 1.6 - 0.25 * progress,
                    "approx_kl": 0.004 + 0.003 * progress,
                    "clip_fraction": 0.04 + 0.05 * progress,
                    "explained_variance": -0.1 + 0.65 * progress,
                    "win_rate": max(0.0, 0.12 * progress - stage * 0.01),
                    "target_damage": max(0.0, 0.18 * progress - stage * 0.012),
                    "own_damage": 0.02 + stage * 0.008,
                    "red_wez_rate": 0.01 + stage * 0.012,
                    "own_crash_rate": 0.005,
                    "action_saturation_rate": 0.04 + stage * 0.018,
                    "mean_delta_action": 0.20 + stage * 0.025,
                    "action_abs_mean": 0.28 + stage * 0.02,
                    "gate": "pass:demo" if update == updates and stage < 4 else "block:demo",
                }
            )

    stages = [
        {"index": stage, "name": name, "advance_conditions": {}}
        for stage, name in sorted({(item["stage"], item["stage_name"]) for item in records})
    ]
    curriculum = [
        {"stage": stage, "name": stages[stage]["name"], "status": "advanced"} for stage in range(4)
    ]
    return records, stages, curriculum, {"variant": "ppo_lstm", "seed": 7}, 0


def load_payload(run_dir, max_points):
    raw_records, skipped = read_jsonl(run_dir / "metrics.jsonl")
    records = [flatten_record(record, i) for i, record in enumerate(raw_records)]
    records = downsample(records, max_points)
    stages = read_json(run_dir / "stage_snapshot.json", [])
    curriculum = read_json(run_dir / "curriculum_state.json", [])
    config = read_json(run_dir / "config.json", {})

    stages = stages if isinstance(stages, list) else []
    curriculum = curriculum if isinstance(curriculum, list) else []
    config = config if isinstance(config, dict) else {}

    if not stages:
        names = {}

        for record in records:
            names[record["stage"]] = record.get("stage_name", f"Stage {record['stage']}")
        stages = [
            {"index": index, "name": names[index], "advance_conditions": {}}
            for index in sorted(names)
        ]
    return records, stages, curriculum, config, skipped, len(raw_records)


def build_report_data(run_dir, records, stages, curriculum, config, skipped, source_count):
    variant = str(config.get("variant", "")).lower()

    if not variant and records:
        variant = "sac" if "q_loss" in records[-1] else "ppo"

    return finite_or_none(
        {
            "run": str(run_dir),
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "variant": variant or "unknown",
            "config": config,
            "records": records,
            "source_record_count": source_count,
            "rendered_record_count": len(records),
            "skipped_lines": skipped,
            "stages": stages,
            "curriculum": curriculum,
        }
    )


HTML_TEMPLATE = r"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  __AUTO_REFRESH__
  <title>Fighter RL Training Report</title>
  <style>
    :root { --line:#cbd5e1; --grid:#e5e7eb; --text:#111827; --muted:#64748b; --green:#067647; --amber:#92400e; --red:#b42318; --blue:#075985; }
    * { box-sizing:border-box; }
    body { margin:0; background:#fff; color:var(--text); font:13px/1.4 Arial,Helvetica,sans-serif; font-variant-numeric:tabular-nums; }
    .shell { max-width:1700px; margin:auto; padding:18px; }
    header { display:flex; justify-content:space-between; align-items:flex-start; gap:16px; padding-bottom:12px; margin-bottom:12px; border-bottom:2px solid #334155; }
    h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:14px; margin:0 0 8px; } .muted { color:var(--muted); }
    .toolbar { display:flex; gap:8px; align-items:center; flex-wrap:wrap; } select { color:var(--text); background:#fff; border:1px solid #94a3b8; padding:5px 7px; }
    .cards { display:grid; grid-template-columns:repeat(6,minmax(130px,1fr)); gap:0; margin-bottom:12px; border:1px solid var(--line); }
    .card { padding:9px 11px; min-height:70px; border-right:1px solid var(--line); } .card:last-child { border-right:0; }
    .label { color:var(--muted); font-size:10px; font-weight:700; text-transform:uppercase; } .value { font-size:19px; font-weight:700; margin-top:4px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; } .sub { color:var(--muted); font-size:11px; margin-top:1px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; } .panel { padding:10px; min-width:0; border:1px solid var(--line); } .span2 { grid-column:span 2; }
    .chart { width:100%; height:220px; display:block; } .axis { stroke:#94a3b8; stroke-width:1; } .gridline { stroke:var(--grid); stroke-width:1; } .chart text { fill:#64748b; font-size:10px; } .legend { display:flex; flex-wrap:wrap; gap:10px; min-height:19px; } .key { display:flex; gap:5px; align-items:center; color:#475569; font-size:11px; } .dot { width:10px; height:3px; }
    table { width:100%; border-collapse:collapse; font-size:11px; } th { background:#f1f5f9; color:#334155; text-align:left; font-weight:700; padding:6px; border:1px solid var(--line); } td { padding:6px; border:1px solid #e2e8f0; vertical-align:top; } tr.current { background:#eff6ff; }
    .badge { display:inline-block; padding:1px 5px; border:1px solid currentColor; font-size:10px; font-weight:700; } .advanced { color:var(--green); } .running { color:var(--blue); } .stalled,.blocked { color:var(--red); } .pending { color:var(--amber); }
    .gate { max-width:430px; color:#475569; word-break:break-word; } .ok { color:var(--green); } .warn { color:var(--amber); } .bad { color:var(--red); }
    footer { margin:12px 0 0; padding-top:8px; border-top:1px solid var(--line); color:var(--muted); font-size:10px; }
    @media(max-width:1050px){ .cards{grid-template-columns:repeat(3,1fr)} .card{border-bottom:1px solid var(--line)} .grid{grid-template-columns:1fr}.span2{grid-column:span 1} }
    @media(max-width:650px){ .shell{padding:10px}.cards{grid-template-columns:repeat(2,1fr)} header{display:block}.toolbar{margin-top:8px} }
  </style>
</head>
<body><main class="shell">
  <header><div><h1>Fighter RL Training Report</h1><div id="run" class="muted"></div></div><div class="toolbar"><span id="variant" class="badge running"></span><label class="muted">Stage <select id="stageFilter"></select></label></div></header>
  <section id="cards" class="cards"></section>
  <section class="grid">
    <article class="panel span2"><h2>Curriculum progress</h2><div id="stageTable"></div></article>
    <article class="panel"><h2>Reward</h2><div id="rewardLegend" class="legend"></div><svg id="rewardChart" class="chart"></svg></article>
    <article class="panel"><h2>Tracking & closure</h2><div id="trackLegend" class="legend"></div><svg id="trackChart" class="chart"></svg></article>
    <article class="panel"><h2>Angles</h2><div id="geometryLegend" class="legend"></div><svg id="geometryChart" class="chart"></svg></article>
    <article class="panel"><h2>Range control</h2><div id="rangeLegend" class="legend"></div><svg id="rangeChart" class="chart"></svg></article>
    <article class="panel"><h2 id="optimizerTitle">Optimizer loss</h2><div id="optimizerLegend" class="legend"></div><svg id="optimizerChart" class="chart"></svg></article>
    <article class="panel"><h2 id="healthTitle">PPO trust region</h2><div id="healthLegend" class="legend"></div><svg id="healthChart" class="chart"></svg></article>
    <article class="panel"><h2>Combat & safety</h2><div id="combatLegend" class="legend"></div><svg id="combatChart" class="chart"></svg></article>
    <article class="panel"><h2>Policy actions</h2><div id="actionLegend" class="legend"></div><svg id="actionChart" class="chart"></svg></article>
    <article class="panel"><h2>Latest gate diagnosis</h2><div id="diagnosis"></div></article>
  </section>
  <footer id="footer"></footer>
</main>
<script id="reportData" type="application/json">__REPORT_DATA__</script>
<script>
const DATA=JSON.parse(document.getElementById('reportData').textContent); const COLORS=['#0072B2','#E69F00','#009E73','#D55E00','#CC79A7','#475569'];
const fmt=(v,d=3)=>v==null||!Number.isFinite(Number(v))?'—':Number(v).toLocaleString(undefined,{maximumFractionDigits:d});
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const latest=arr=>arr.length?arr[arr.length-1]:{}; const byStage=new Map(); DATA.records.forEach(r=>{if(!byStage.has(r.stage))byStage.set(r.stage,[]);byStage.get(r.stage).push(r)});
const curriculum=new Map((DATA.curriculum||[]).map(x=>[Number(x.stage),x])); const maxStage=DATA.records.length?Math.max(...DATA.records.map(r=>r.stage)):0;
document.getElementById('run').textContent=DATA.run+' · generated '+DATA.generated_at; document.getElementById('variant').textContent=DATA.variant.toUpperCase();
const filter=document.getElementById('stageFilter'); filter.innerHTML='<option value="all">All stages</option>'+DATA.stages.map(s=>`<option value="${s.index}">${s.index} · ${esc(s.name)}</option>`).join(''); filter.value=DATA.records.length?String(maxStage):'all';
function selectedRecords(){return filter.value==='all'||!filter.value?DATA.records:DATA.records.filter(r=>r.stage===Number(filter.value));}
function statusFor(stage){const known=curriculum.get(Number(stage.index));if(known)return known.status||'advanced';if(Number(stage.index)<maxStage)return'advanced';if(Number(stage.index)===maxStage)return latest(byStage.get(Number(stage.index))||[]).status==='stalled'?'stalled':'running';return'pending'}
function card(label,value,sub=''){return `<div class="card"><div class="label">${label}</div><div class="value">${value}</div><div class="sub">${sub}</div></div>`}
function renderCards(records){const last=latest(records), advanced=DATA.stages.filter(s=>statusFor(s)==='advanced').length, gate=String(last.gate||'pending');document.getElementById('cards').innerHTML=card('Current stage',`${fmt(last.stage,0)} · ${esc(last.stage_name||'—')}`,`${advanced}/${DATA.stages.length} advanced`)+card('Valid steps',fmt(last.valid_steps,0),`update ${fmt(last.update,0)}`)+card('Reward',fmt(last.reward_mean,4),`win ${fmt(last.win_rate)} · dmg ${fmt(last.target_damage)}`)+card('Track score',fmt(last.track_score),`worst ${fmt(last.bucket_worst_track_score)}`)+card('Closure violation',fmt(last.closure_violation_rate),`step frac ${fmt(last.closure_violation_step_fraction)}`)+card('Gate',gate.startsWith('pass:')?'<span class="ok">PASS</span>':'<span class="bad">BLOCK</span>',esc(gate.replace(/^(pass|block):/,'')));}
function renderStages(){let rows=DATA.stages.map(stage=>{const rs=byStage.get(Number(stage.index))||[],last=latest(rs),status=statusFor(stage),gate=last.gate||'',thresholds=stage.advance_conditions||{};return `<tr class="${Number(stage.index)===maxStage?'current':''}"><td>${stage.index}</td><td><b>${esc(stage.name)}</b></td><td><span class="badge ${esc(status)}">${esc(status)}</span></td><td>${fmt(last.update,0)}</td><td>${fmt(last.reward_mean,4)}</td><td>${fmt(last.track_score)}</td><td>${fmt(last.closure_violation_rate)}</td><td class="gate">${esc(gate)}</td><td class="gate">${esc(Object.entries(thresholds).map(([k,v])=>k+'='+v).join(', '))}</td></tr>`}).join('');document.getElementById('stageTable').innerHTML=`<div style="overflow:auto"><table><thead><tr><th>#</th><th>Stage</th><th>Status</th><th>Update</th><th>Reward</th><th>Track</th><th>Closure V.</th><th>Latest gate</th><th>Thresholds</th></tr></thead><tbody>${rows}</tbody></table></div>`;}
function renderLegend(id,series){document.getElementById(id).innerHTML=series.map((s,i)=>`<span class="key"><i class="dot" style="background:${s.color||COLORS[i]}"></i>${esc(s.label)}</span>`).join('')}
function chart(svgId,legendId,records,series,{min=null,max=null}={}){const svg=document.getElementById(svgId);renderLegend(legendId,series);const W=720,H=245,p={l:48,r:16,t:12,b:27};svg.setAttribute('viewBox',`0 0 ${W} ${H}`);const values=[];series.forEach(s=>records.forEach(r=>{const v=Number(r[s.key]);if(Number.isFinite(v))values.push(v)}));if(!values.length){svg.innerHTML='<text x="360" y="125" text-anchor="middle">No data for this metric</text>';return}let lo=min??Math.min(...values),hi=max??Math.max(...values);if(lo===hi){lo-=1;hi+=1}const pad=(hi-lo)*.08; if(min==null)lo-=pad;if(max==null)hi+=pad;const x=i=>p.l+(records.length<=1?0:i/(records.length-1))*(W-p.l-p.r),y=v=>p.t+(hi-v)/(hi-lo)*(H-p.t-p.b);let out='';for(let i=0;i<5;i++){const yy=p.t+i*(H-p.t-p.b)/4,v=hi-i*(hi-lo)/4;out+=`<line class="gridline" x1="${p.l}" y1="${yy}" x2="${W-p.r}" y2="${yy}"/><text x="${p.l-7}" y="${yy+3}" text-anchor="end">${fmt(v,2)}</text>`}out+=`<line class="axis" x1="${p.l}" y1="${H-p.b}" x2="${W-p.r}" y2="${H-p.b}"/>`;series.forEach((s,si)=>{let segments=[],current=[];records.forEach((r,i)=>{const v=Number(r[s.key]);if(Number.isFinite(v))current.push(`${x(i).toFixed(1)},${y(v).toFixed(1)}`);else if(current.length){segments.push(current);current=[]}});if(current.length)segments.push(current);segments.forEach(points=>{out+=`<polyline fill="none" stroke="${s.color||COLORS[si]}" stroke-width="2" stroke-linejoin="round" points="${points.join(' ')}"/>`})});out+=`<text x="${p.l}" y="${H-7}">${records.length?`stage ${records[0].stage} / update ${records[0].update}`:''}</text><text x="${W-p.r}" y="${H-7}" text-anchor="end">${records.length?`stage ${latest(records).stage} / update ${latest(records).update}`:''}</text>`;svg.innerHTML=out;}
function renderDiagnosis(records){const last=latest(records),gate=String(last.gate||'No completed gate window yet'),parts=gate.replace(/^(pass|block):/,'').split(',').filter(Boolean),klass=gate.startsWith('pass:')?'ok':'bad';document.getElementById('diagnosis').innerHTML=`<p class="${klass}"><b>${gate.startsWith('pass:')?'Gate passed':'Gate blocked'}</b></p>`+(parts.length?`<ul>${parts.map(x=>`<li>${esc(x)}</li>`).join('')}</ul>`:`<p class="muted">${esc(gate)}</p>`)+`<table><tbody><tr><td>explained variance</td><td>${fmt(last.explained_variance)}</td></tr><tr><td>entropy</td><td>${fmt(last.entropy)}</td></tr><tr><td>approx KL / clip fraction</td><td>${fmt(last.approx_kl)} / ${fmt(last.clip_fraction)}</td></tr><tr><td>action saturation / delta</td><td>${fmt(last.action_saturation_rate)} / ${fmt(last.mean_delta_action)}</td></tr></tbody></table><p class="muted">open=${fmt(last.opening_away_step_fraction)} is the fraction of steps moving away outside trail range; it is not action magnitude.</p>`;}
function render(){const r=selectedRecords();renderCards(r);chart('rewardChart','rewardLegend',r,[{key:'reward_mean',label:'reward'}]);chart('trackChart','trackLegend',r,[{key:'track_score',label:'track score'},{key:'closure_violation_rate',label:'closure violation'},{key:'opening_away_step_fraction',label:'opening away'}],{min:0,max:1});chart('geometryChart','geometryLegend',r,[{key:'final_ata_deg',label:'ATA deg'},{key:'final_aa_deg',label:'AA deg'}],{min:0});chart('rangeChart','rangeLegend',r,[{key:'trail_range_error_m',label:'range error m'},{key:'mean_abs_closing_mps',label:'abs closure m/s'}],{min:0});const sac=DATA.variant.includes('sac');document.getElementById('optimizerTitle').textContent=sac?'SAC optimizer loss':'PPO optimizer loss';document.getElementById('healthTitle').textContent=sac?'SAC temperature':'PPO trust region';chart('optimizerChart','optimizerLegend',r,sac?[{key:'q_loss',label:'Q loss'},{key:'actor_loss',label:'actor loss'}]:[{key:'loss',label:'total loss'},{key:'value_loss',label:'value loss'},{key:'policy_loss',label:'policy loss'}]);chart('healthChart','healthLegend',r,sac?[{key:'alpha',label:'alpha'}]:[{key:'approx_kl',label:'approx KL'},{key:'clip_fraction',label:'clip fraction'}],{min:0});chart('combatChart','combatLegend',r,[{key:'win_rate',label:'win rate'},{key:'target_damage',label:'target damage'},{key:'own_damage',label:'own damage'},{key:'red_wez_rate',label:'red WEZ'},{key:'own_crash_rate',label:'own crash'}],{min:0});chart('actionChart','actionLegend',r,[{key:'action_abs_mean',label:'abs mean'},{key:'action_std',label:'std'},{key:'action_saturation_rate',label:'saturation'},{key:'mean_delta_action',label:'delta action'}],{min:0});renderDiagnosis(r)}
filter.addEventListener('change',render);renderStages();render();document.getElementById('footer').textContent=`Source records: ${DATA.source_record_count.toLocaleString()} · rendered: ${DATA.rendered_record_count.toLocaleString()} · skipped malformed lines: ${DATA.skipped_lines}`;
</script></body></html>"""


def render_html(data, refresh_seconds=0):
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    refresh = (
        f'<meta http-equiv="refresh" content="{max(1, int(refresh_seconds))}">'
        if refresh_seconds > 0
        else ""
    )
    return HTML_TEMPLATE.replace("__REPORT_DATA__", payload).replace("__AUTO_REFRESH__", refresh)


def render_once(args):
    if args.demo:
        records, stages, curriculum, config, skipped = demo_payload()
        run_dir = Path("demo")
        source_count = len(records)
    else:
        run_dir = resolve_run(args.run)
        records, stages, curriculum, config, skipped, source_count = load_payload(
            run_dir, args.max_points
        )

    data = build_report_data(run_dir, records, stages, curriculum, config, skipped, source_count)
    if args.output:
        output = Path(args.output).expanduser().resolve()
    elif args.demo:
        output = Path.cwd() / "training_report_demo.html"
    else:
        output = run_dir / "training_report.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_html(data, args.watch), encoding="utf-8")
    print(f"report: {output}")
    print(f"source: {run_dir} ({source_count} records, {len(records)} rendered)")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?", help="Run directory, metrics.jsonl, or parent directory")
    parser.add_argument(
        "-o", "--output", help="Output HTML path (default: RUN/training_report.html)"
    )
    parser.add_argument("--max-points", type=int, default=6000, help="Maximum embedded points")
    parser.add_argument(
        "--watch", type=float, default=0, metavar="SECONDS", help="Regenerate continuously"
    )
    parser.add_argument("--demo", action="store_true", help="Generate a demo report without a run")
    args = parser.parse_args()

    if not args.demo and not args.run:
        parser.error("run is required unless --demo is used")
    if args.max_points < 100:
        parser.error("--max-points must be at least 100")
    if args.watch < 0:
        parser.error("--watch cannot be negative")
    return args


def main():
    args = parse_args()

    try:
        while True:
            render_once(args)

            if args.watch <= 0:
                break
            time.sleep(max(1.0, args.watch))
    except KeyboardInterrupt:
        print("stopped")
    except (OSError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
