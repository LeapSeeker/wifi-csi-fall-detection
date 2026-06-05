"""onset 검수 도구 HTML 생성기 v2 — 인터랙티브 차트 (read-only).

priority_review_queue.csv(98 train/val) 를 진규가 브라우저에서 빠르게 검수하는 단일 HTML
도구를 생성한다. 결정은 localStorage 자동저장 + Export(csv/json) → Claude Code 가 그 파일을
읽어 manifest_v2_manual_augmented 생성.

v2 변경: PNG 고정 이미지 대신 sparse-energy 곡선을 **인터랙티브 canvas 차트**로 재구현.
  곡선 데이터는 export_energy_curves.py 가 만든 energy_curves.json 을 빌드시 임베드한다.
  (검수자 피드백 반영: x축 눈금 차등 / hover frame 읽기 / 클릭=수정입력 / 실시간 '내 선택'
   세로선 / 색약 접근성-텍스트라벨·빗금·파랑주황 / 에너지 곡선 강조 / rise_not_found 안내)

- 자기완결 HTML (queue + 곡선 데이터 임베드). 설치/서버 불필요(브라우저로 열면 동작).
- 곡선 데이터가 없는 세션은 PNG(../plots_priority/) 로 폴백.
- 판정은 진규. 추천은 참고만.

제약: read-only(원본 데이터·plot png·동결 파일·manifest 무수정). 도구 HTML 만 생성.
산출: debug/modeling/diag_out/onset_detector/finalization/review_tool/onset_review.html
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

FINAL = ROOT / "debug/modeling/diag_out/onset_detector/finalization"
QUEUE = FINAL / "priority_review_queue.csv"
TOOL_DIR = FINAL / "review_tool"
CURVES_JSON = TOOL_DIR / "energy_curves.json"


def to_clean(o):
    if o is None:
        return None
    if 50 <= o < 150:
        return o - 50
    if 200 <= o < 400:
        return o - 100
    if 450 <= o < 550:
        return o - 150
    return None


def build_sessions():
    rows = list(csv.DictReader(open(QUEUE, encoding="utf-8-sig")))
    rows = [r for r in rows if r["split_assignment"] in ("train", "val")]
    out = []
    for r in rows:
        hard = r["hard_reasons"] or ""
        rise = None
        if r["rise_frame_original"] not in ("", "None"):
            rise = int(float(r["rise_frame_original"]))
        clean_cands = [int(float(x)) for x in r["topk_cand_frames_clean"].split("|")
                       if x not in ("", "None")]
        # in-window 후보 (search 200..349 → clean 100..249 → orig = clean+100)
        orig_cands = [c + 100 for c in clean_cands]
        if "rise_not_found" in hard:
            rec_o = None
        elif "beep" in hard or "too_early" in hard:
            rec_o = orig_cands[0] if orig_cands else None
        else:
            rec_o = rise
        pob = r["peak_over_baseline"]
        walk_contam = r["walk_baseline_contamination"] == "True"
        hint = f"search 상승 {pob}x baseline; " + (
            f"in-window onset 후보 {orig_cands}" if orig_cands else "in-window 후보 없음(약한 transient)")
        if walk_contam:
            hint += " | ⚠ WALK baseline 오염(걷기 peak가 baseline[50:150] 구간)"
        out.append({
            "rank": int(r["rank"]), "filename": r["filename"], "subtype": r["subtype"],
            "split": r["split_assignment"], "priority": r["review_priority"],
            "hard": hard, "soft": r["soft_reasons"], "soft_count": int(r["soft_count"] or 0),
            "noise": r["baseline_noise_ratio"], "param_sens": r["param_sensitivity"],
            "pob": pob, "topk_count": r["topk_count"], "walk_contam": walk_contam,
            "rise_orig": rise, "topk_clean": clean_cands, "topk_orig": orig_cands,
            "rec_orig": rec_o, "rec_clean": to_clean(rec_o),
            "reco_text": r["recommendation"], "hint": hint,
            "plot": "review_" + r["filename"].replace(".csv", "") + ".png",
        })
    return out


HTML = r"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8"><title>SafeSignal onset 검수 v2</title>
<style>
*{box-sizing:border-box} body{font-family:system-ui,'Malgun Gothic',sans-serif;margin:0;background:#f4f5f7;color:#1a1a1a}
#bar{position:sticky;top:0;background:#222;color:#fff;padding:8px 14px;display:flex;gap:14px;align-items:center;flex-wrap:wrap;z-index:10}
#bar b{font-size:15px} #bar .sp{flex:1}
button{cursor:pointer;border:0;border-radius:6px;padding:7px 12px;font-size:14px;font-weight:600}
.bA{background:#1b7a2e;color:#fff}.bC{background:#b34700;color:#fff}.bM{background:#1565c0;color:#fff}
.bnav{background:#555;color:#fff}.bx{background:#37474f;color:#fff}
select{padding:6px;border-radius:6px;border:1px solid #ccc;font-size:14px}
#wrap{max-width:1180px;margin:14px auto;padding:0 14px}
#meta{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap;margin-bottom:6px}
#meta .fn{font-size:18px;font-weight:700} .tag{font-size:12px;padding:2px 8px;border-radius:10px;background:#e0e0e0;border:1px solid #c4c4c4}
.tag.walk{background:#fff;border:2px solid #b34700;font-weight:700} .tag.hard{background:#fff;border:2px solid #c62828;font-weight:700;color:#c62828} .tag.soft{background:#fff;border:1.5px dashed #8a6d00;color:#7a5f00}
#chartwrap{position:relative;background:#fff;border:1px solid #c4c4c4;border-radius:8px;padding:6px 6px 2px}
#chart{width:100%;height:340px;display:block;cursor:crosshair}
#mchart{width:100%;height:300px;display:block;cursor:crosshair}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;margin:6px 4px 2px;align-items:center}
.legend span{display:inline-flex;align-items:center;gap:6px}
.legend i{display:inline-block;width:28px;height:0;border-top:3px solid #000}
.guide{font-size:12.5px;color:#222;margin:6px 4px 2px;line-height:1.55;background:#eef3fb;border-left:4px solid #1565c0;padding:7px 10px;border-radius:0 6px 6px 0}
#tip{position:fixed;display:none;background:#111;color:#fff;padding:5px 9px;border-radius:5px;font-size:12px;pointer-events:none;z-index:99;line-height:1.45;white-space:nowrap}
#info{background:#fff;border:1px solid #c4c4c4;border-radius:8px;padding:12px;margin-top:10px;font-size:14px;line-height:1.6}
#info .rec{font-size:16px;font-weight:700;color:#0d47a1}
#info .note{color:#777;font-size:12px}
#info .rnf{background:#fff3e0;border:2px solid #b34700;border-radius:6px;padding:8px 10px;margin:8px 0;color:#7a3300;font-weight:600}
.cand{display:inline-block;margin:3px;padding:5px 10px;background:#fff;border:1.5px solid #1565c0;border-radius:6px;cursor:pointer;font-size:13px;color:#0d47a1;font-weight:600}
#picked{font-weight:700;color:#1b7a2e}
#actions{margin-top:12px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
#strip{display:flex;gap:2px;margin-top:10px;flex-wrap:wrap}
#strip span{width:16px;height:16px;border-radius:3px;background:#e6e6e6;border:1px solid #bbb;cursor:pointer;font-size:9px;line-height:14px;text-align:center;font-weight:700;color:#fff}
#strip span.approve{background:#1b7a2e;border-color:#155a22}#strip span.exclude{background:#b34700;border-color:#8a3700}#strip span.modify{background:#1565c0;border-color:#0d47a1}
#strip span.undone{background:#e6e6e6;color:#999;border-color:#bbb}
#strip span.cur{outline:3px solid #111;outline-offset:1px}
#modal{display:none;position:fixed;inset:0;background:rgba(0,0,0,.6);z-index:50;align-items:center;justify-content:center}
#modalbox{background:#fff;border-radius:10px;padding:16px;max-width:1040px;width:96%;max-height:94vh;overflow:auto}
#onsetIn{font-size:18px;padding:8px;width:130px;border:3px solid #1b7a2e;border-radius:6px}
#mwarn{display:none;margin:8px 0;padding:7px 10px;background:#fff3e0;border:2px solid #b34700;border-radius:6px;color:#7a3300;font-weight:600;font-size:13px}
.kbd{font-family:monospace;background:#eee;border:1px solid #bbb;border-radius:4px;padding:0 5px}
#exrow{display:none;gap:8px;align-items:center}
</style></head><body>
<div id="bar">
  <b id="prog">0 / 0</b>
  <span>남은: <b id="remain">0</b></span>
  <span>필터:</span>
  <select id="filter"><option value="">전체</option></select>
  <span class="sp"></span>
  <button class="bx" onclick="exp('csv')">Export CSV</button>
  <button class="bx" onclick="exp('json')">Export JSON</button>
  <button class="bnav" onclick="if(confirm('모든 판정 초기화?'))reset()">초기화</button>
</div>
<div id="wrap">
  <div id="meta"></div>
  <div class="guide">판단 기준 = <b>에너지 곡선(파란 굵은 선)</b>이 낙상처럼 <b>확 솟는 지점</b>이 onset.
    thr(회색 점선)·auto rise(주황)는 <b>참고용</b>. 그래프에 마우스를 올리면 frame·energy 표시,
    클릭하면 그 frame이 '선택 후보'(초록선)가 됩니다 — <span class="kbd">M</span> 로 그 값을 수정 저장.</div>
  <div id="chartwrap">
    <canvas id="chart"></canvas>
    <img id="fallimg" style="display:none;width:100%;border:1px solid #ddd;border-radius:6px" alt="plot fallback">
  </div>
  <div class="legend">
    <span><i style="border-top:3px solid #1565c0"></i>sparse energy (판단 기준)</span>
    <span><i style="border-top:2px dashed #888"></i>thr (참고)</span>
    <span><i style="border-top:2px dashed #e8830c"></i>auto rise (참고)</span>
    <span><i style="border-top:3px solid #1b7a2e"></i>내 선택 / 확정 onset</span>
    <span>구간: <b>BASELINE</b> 빗금 / <b>SEARCH</b> 파란빗금 / <b>BEEP</b> 주황교차빗금 (글자라벨 병기)</span>
  </div>
  <div style="margin:4px 4px 0;font-size:13px">클릭한 frame: <span id="picked">—</span></div>
  <div id="info"></div>
  <div id="actions">
    <button class="bA" onclick="decide('approve')">승인 <span class="kbd">A</span></button>
    <button class="bC" onclick="askExclude()">제외 <span class="kbd">C</span></button>
    <button class="bM" onclick="openModify()">수정 <span class="kbd">M</span></button>
    <span id="exrow">
      <select id="exreason">
        <option value="no_clear_transient">no_clear_transient</option>
        <option value="beep_misfire">beep_misfire</option>
        <option value="walking_residual">walking_residual</option>
        <option value="other">other</option>
      </select>
      <button class="bC" onclick="decide('exclude')">제외 확정</button>
    </span>
    <span class="sp" style="flex:1"></span>
    <button class="bnav" onclick="go(-1)">← 이전</button>
    <button class="bnav" onclick="go(1)">다음 →</button>
  </div>
  <div id="strip"></div>
</div>
<div id="modal"><div id="modalbox">
  <h3 id="mtitle"></h3>
  <div id="chartwrap" style="position:relative"><canvas id="mchart"></canvas></div>
  <p style="margin:8px 0 4px">onset frame (original 기준). 그래프 클릭/후보 클릭/직접 입력 → <b>초록 세로선이 실시간 이동</b>:</p>
  <div id="mcands"></div>
  <p><input id="onsetIn" type="number"> <span style="color:#1b7a2e;font-weight:700">clean <span id="cleanOut">—</span></span>
     <button class="bA" onclick="saveModify()">저장</button>
     <button class="bnav" onclick="closeModal()">취소</button></p>
  <div id="mwarn"></div>
</div></div>
<div id="tip"></div>
<script>
const SESSIONS = __DATA__;
const CDATA = __CURVES__;
const CURVES = CDATA.curves || {};
const RG = CDATA.meta || {baseline:[50,150],search:[190,350],fall_window:[200,400],beep_regions:[[0,50],[150,200],[400,450]],n_frames:550};
const N = RG.n_frames || 550;
const LS = "onset_review_decisions_v1";
let dec = JSON.parse(localStorage.getItem(LS) || "{}");
let filt = "", view = [], idx = 0, picked = null;
const tip = document.getElementById('tip');

function toClean(o){o=Number(o);if(isNaN(o))return null;if(o>=50&&o<150)return o-50;if(o>=200&&o<400)return o-100;if(o>=450&&o<550)return o-150;return null;}
function save(){localStorage.setItem(LS, JSON.stringify(dec));}
function rebuild(){view = SESSIONS.filter(s=>!filt||s.subtype===filt);if(idx>=view.length)idx=Math.max(0,view.length-1);}
function nowISO(){return new Date().toISOString();}
function inSearch(f){return f>=RG.search[0] && f<RG.search[1];}

/* ── 인터랙티브 차트 (canvas) ──────────────────────────────────────────────── */
const PAD={l:52,r:16,t:30,b:50};
function box(cv){return {x:PAD.l,y:PAD.t,w:cv.clientWidth-PAD.l-PAD.r,h:cv.clientHeight-PAD.t-PAD.b};}
function xPx(f,b){return b.x + (f/(N-1))*b.w;}
function yPx(v,b,ymax){return b.y + (1-v/ymax)*b.h;}
function frameAt(px,b){let f=Math.round((px-b.x)/b.w*(N-1));return Math.max(0,Math.min(N-1,f));}

function hatch(ctx,x0,x1,b,color,dir,gap,lw,alpha){
  ctx.save();ctx.beginPath();ctx.rect(x0,b.y,x1-x0,b.h);ctx.clip();
  ctx.strokeStyle=color;ctx.lineWidth=lw||1;ctx.globalAlpha=alpha||0.5;
  const span=(x1-x0)+b.h;
  for(let p=-b.h;p<span;p+=gap){ctx.beginPath();
    if(dir>0){ctx.moveTo(x0+p,b.y+b.h);ctx.lineTo(x0+p+b.h,b.y);}
    else{ctx.moveTo(x0+p,b.y);ctx.lineTo(x0+p+b.h,b.y+b.h);}
    ctx.stroke();}
  ctx.restore();
}
function vline(ctx,f,b,color,lw,dash,y0frac,y1frac){
  const x=xPx(f,b);ctx.save();ctx.strokeStyle=color;ctx.lineWidth=lw;ctx.setLineDash(dash||[]);
  ctx.beginPath();ctx.moveTo(x,b.y+(y0frac||0)*b.h);ctx.lineTo(x,b.y+(y1frac||1)*b.h);ctx.stroke();ctx.restore();
}
function rlabel(ctx,x,txt,color){ctx.save();ctx.fillStyle=color;ctx.font='bold 11px system-ui';ctx.fillText(txt,x,PAD.t-16);ctx.restore();}

function drawChart(cv,s,st){
  const dpr=window.devicePixelRatio||1;
  const cssW=cv.clientWidth||1100, cssH=cv.clientHeight||340;
  cv.width=cssW*dpr;cv.height=cssH*dpr;
  const ctx=cv.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,cssW,cssH);
  ctx.font='11px system-ui';
  const c=CURVES[s.filename];
  const b=box(cv);
  // 외곽
  ctx.strokeStyle='#ddd';ctx.lineWidth=1;ctx.strokeRect(b.x,b.y,b.w,b.h);
  const im=document.getElementById('fallimg');
  if(!c){ctx.fillStyle='#b34700';ctx.font='14px system-ui';
    ctx.fillText('곡선 데이터 없음 — PNG 폴백', b.x+10, b.y+24);
    if(im && cv.id==='chart'){im.src='../plots_priority/'+s.plot;im.style.display='block';}
    return;}
  if(im)im.style.display='none';
  const e=c.e; let ymax=Math.max(c.ymax||0, c.thr||0)*1.14; if(!(ymax>0))ymax=1;

  // 구간 빗금 + 라벨 (색약: 패턴+텍스트)
  hatch(ctx, xPx(RG.baseline[0],b), xPx(RG.baseline[1],b), b, '#666', -1, 8, 1, 0.45);
  hatch(ctx, xPx(RG.search[0],b), xPx(RG.search[1],b), b, '#1565c0', 1, 9, 1, 0.30);
  RG.beep_regions.forEach(br=>{hatch(ctx,xPx(br[0],b),xPx(br[1],b),b,'#e8830c',1,8,1,0.40);
    hatch(ctx,xPx(br[0],b),xPx(br[1],b),b,'#e8830c',-1,8,1,0.40);});
  // 구간 경계 점선 + 라벨
  rlabel(ctx, xPx(RG.baseline[0],b)+2, 'BASELINE '+RG.baseline[0]+'–'+RG.baseline[1], '#444');
  rlabel(ctx, xPx(RG.search[0],b)+2, 'SEARCH '+RG.search[0]+'–'+RG.search[1]+' (유효 onset 구간)', '#0d47a1');
  RG.beep_regions.forEach(br=>{vline(ctx,br[0],b,'#e8830c',1,[3,3]);vline(ctx,br[1],b,'#e8830c',1,[3,3]);
    ctx.save();ctx.fillStyle='#8a3700';ctx.font='bold 10px system-ui';ctx.fillText('BEEP',xPx(br[0],b)+1,PAD.t-3);ctx.restore();});
  vline(ctx,RG.baseline[0],b,'#666',1,[4,3]);vline(ctx,RG.baseline[1],b,'#666',1,[4,3]);
  vline(ctx,RG.search[0],b,'#1565c0',1.2,[5,3]);vline(ctx,RG.search[1],b,'#1565c0',1.2,[5,3]);

  // x 눈금 3단계 (1/10/100)
  ctx.textAlign='center';ctx.textBaseline='top';
  for(let f=0;f<N;f++){const x=xPx(f,b);
    if(f%100===0){ctx.strokeStyle='#333';ctx.lineWidth=1.6;ctx.beginPath();ctx.moveTo(x,b.y+b.h);ctx.lineTo(x,b.y+b.h+11);ctx.stroke();
      ctx.fillStyle='#222';ctx.font='bold 12px system-ui';ctx.fillText(f,x,b.y+b.h+13);}
    else if(f%10===0){ctx.strokeStyle='#888';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x,b.y+b.h);ctx.lineTo(x,b.y+b.h+7);ctx.stroke();
      ctx.fillStyle='#777';ctx.font='9px system-ui';ctx.fillText(f,x,b.y+b.h+8);}
    else{ctx.strokeStyle='#d2d2d2';ctx.lineWidth=0.5;ctx.beginPath();ctx.moveTo(x,b.y+b.h);ctx.lineTo(x,b.y+b.h+3);ctx.stroke();}
  }
  ctx.textAlign='left';
  // y 눈금 + 라벨
  ctx.textBaseline='middle';ctx.fillStyle='#777';ctx.font='10px system-ui';
  for(let k=0;k<=4;k++){const v=ymax*k/4;const y=yPx(v,b,ymax);
    ctx.strokeStyle='#eee';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(b.x,y);ctx.lineTo(b.x+b.w,y);ctx.stroke();
    ctx.fillText(v.toFixed(3),4,y);}
  ctx.textBaseline='alphabetic';

  // thr (참고: 흐리게/얇게)
  if(c.thr!=null){const y=yPx(c.thr,b,ymax);ctx.save();ctx.strokeStyle='#888';ctx.lineWidth=1;ctx.setLineDash([4,4]);ctx.globalAlpha=0.8;
    ctx.beginPath();ctx.moveTo(b.x,y);ctx.lineTo(b.x+b.w,y);ctx.stroke();ctx.restore();
    ctx.fillStyle='#888';ctx.font='10px system-ui';ctx.fillText('thr '+c.thr.toFixed(3),b.x+b.w-66,y-3);}

  // auto rise (참고: 주황 점선)
  if(s.rise_orig!=null){vline(ctx,s.rise_orig,b,'#e8830c',1.6,[6,4]);
    ctx.save();ctx.fillStyle='#b34700';ctx.font='bold 10px system-ui';ctx.fillText('auto '+s.rise_orig,xPx(s.rise_orig,b)+2,b.y+12);ctx.restore();}
  // topk (보조: 짧은 점선)
  (s.topk_orig||[]).forEach(o=>vline(ctx,o,b,'#e8830c',1,[2,3],0.66,1));

  // sparse energy 곡선 (주인공: 굵은 파랑)
  ctx.save();ctx.strokeStyle='#1565c0';ctx.lineWidth=3;ctx.lineJoin='round';ctx.beginPath();
  for(let f=0;f<e.length;f++){const x=xPx(f,b),y=yPx(e[f],b,ymax);if(f===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);}
  ctx.stroke();ctx.restore();

  // 내 선택 / 확정 (주인공급 초록 굵은 실선)
  if(st && st.sel!=null && !isNaN(st.sel)){const f=st.sel;const x=xPx(f,b);
    vline(ctx,f,b,'#1b7a2e',3,[]);
    ctx.save();ctx.fillStyle='#1b7a2e';ctx.beginPath();ctx.moveTo(x-6,b.y-2);ctx.lineTo(x+6,b.y-2);ctx.lineTo(x,b.y+8);ctx.closePath();ctx.fill();
    const lbl=(st.selLabel||'내 선택')+' @'+f+' (clean '+toClean(f)+')';
    ctx.font='bold 12px system-ui';const tw=ctx.measureText(lbl).width;
    let lx=x+8; if(lx+tw>b.x+b.w)lx=x-8-tw;
    ctx.fillStyle='#1b7a2e';ctx.fillRect(lx-3,b.y+10,tw+6,16);ctx.fillStyle='#fff';ctx.fillText(lbl,lx,b.y+22);ctx.restore();}

  // hover guide
  if(st && st.hover!=null){const f=st.hover;const x=xPx(f,b);
    vline(ctx,f,b,'#333',1,[2,2]);
    const y=yPx(e[f],b,ymax);ctx.save();ctx.fillStyle='#333';ctx.beginPath();ctx.arc(x,y,3.5,0,7);ctx.fill();ctx.restore();}
  ctx.fillStyle='#444';ctx.font='11px system-ui';ctx.fillText('original frame',b.x+b.w/2-32,b.y+b.h+34);
}

function attachChart(cv,s,opts){
  cv._hover=null;
  function redraw(extra){const st={hover:(cv._hover!=null?cv._hover:null),sel:opts.getSel(),selLabel:opts.selLabel};
    if(extra)Object.assign(st,extra);drawChart(cv,s,st);}
  cv._redraw=redraw;
  cv.onmousemove=(ev)=>{const r=cv.getBoundingClientRect();const px=ev.clientX-r.left;const b=box(cv);
    if(px<b.x||px>b.x+b.w){cv._hover=null;tip.style.display='none';redraw();return;}
    const f=frameAt(px,b);cv._hover=f;redraw();
    const c=CURVES[s.filename];const en=c?c.e[f]:null;
    tip.style.display='block';tip.style.left=(ev.clientX+14)+'px';tip.style.top=(ev.clientY+14)+'px';
    tip.innerHTML='frame <b>'+f+'</b> (clean '+toClean(f)+')<br>energy '+(en!=null?en.toFixed(4):'-');};
  cv.onmouseleave=()=>{cv._hover=null;tip.style.display='none';redraw();};
  cv.onclick=(ev)=>{const r=cv.getBoundingClientRect();const px=ev.clientX-r.left;const b=box(cv);
    if(px<b.x||px>b.x+b.w)return;opts.onPick(frameAt(px,b));};
  redraw();return redraw;
}

/* ── 판정 ──────────────────────────────────────────────────────────────────── */
function decide(kind){const s=view[idx];if(!s)return;
  if(kind==='approve'){ if(s.rec_orig==null){alert('추천 onset 없음(rise_not_found 등) → 수정(M) 또는 제외(C)');return;}
    dec[s.filename]={decision:'approve',onset:s.rec_orig,reviewed_at:nowISO()};}
  else if(kind==='exclude'){const r=document.getElementById('exreason').value;
    dec[s.filename]={decision:'exclude',onset:null,exclude_reason:r,reviewed_at:nowISO()};}
  save();document.getElementById('exrow').style.display='none';next();}
function askExclude(){const e=document.getElementById('exrow');e.style.display=e.style.display==='flex'?'none':'flex';}

let mchart=null;
function openModify(prefill){const s=view[idx];if(!s)return;
  document.getElementById('mtitle').textContent=s.filename+'  ('+s.subtype+')';
  const mc=document.getElementById('mcands');mc.innerHTML='';
  const cands=[];if(s.rise_orig!=null)cands.push(['auto rise',s.rise_orig]);
  s.topk_orig.forEach((o,i)=>cands.push(['topk'+(i+1),o]));
  if(cands.length===0)mc.innerHTML='<i>후보 없음 — 그래프 클릭 또는 직접 입력</i>';
  cands.forEach(([lbl,o])=>{const bn=document.createElement('span');bn.className='cand';
    bn.textContent=lbl+': '+o+' (clean '+toClean(o)+')';bn.onclick=()=>setOnset(o);mc.appendChild(bn);});
  const cur=dec[s.filename];
  const init=(prefill!=null)?prefill:((cur&&cur.onset!=null)?cur.onset:(s.rec_orig!=null?s.rec_orig:''));
  document.getElementById('onsetIn').value=init;
  document.getElementById('modal').style.display='flex';
  const mcv=document.getElementById('mchart');
  mchart=attachChart(mcv,s,{getSel:()=>{const v=parseInt(document.getElementById('onsetIn').value,10);return isNaN(v)?null:v;},
    selLabel:'내 선택', onPick:setOnset});
  onsetChanged();
}
function setOnset(o){document.getElementById('onsetIn').value=o;onsetChanged();}
function onsetChanged(){const s=view[idx];const v=parseInt(document.getElementById('onsetIn').value,10);
  document.getElementById('cleanOut').textContent=(isNaN(v)?'—':(toClean(v)==null?'(매핑밖)':toClean(v)));
  const w=document.getElementById('mwarn');
  if(!isNaN(v) && !inSearch(v)){w.style.display='block';
    w.innerHTML='⚠ frame '+v+' 는 SEARCH['+RG.search[0]+'–'+RG.search[1]+'] 밖입니다. '
      +(s&&s.hard.includes('rise_not_found')?'(rise_not_found 세션: BASELINE 등 SEARCH 밖 onset 허용 — 곡선 확인 후 저장)':'(저장 시 한번 더 확인)');}
  else w.style.display='none';
  if(mchart)mchart();}
function saveModify(){const s=view[idx];const v=parseInt(document.getElementById('onsetIn').value,10);
  if(isNaN(v)){alert('frame 숫자 입력');return;}
  if(v<0||v>=N){alert('frame 범위 0–'+(N-1));return;}
  if(!inSearch(v)){if(!confirm('⚠ frame '+v+' 가 SEARCH['+RG.search[0]+'–'+RG.search[1]+'] 밖입니다.\n그래도 onset 으로 저장할까요?'))return;}
  dec[s.filename]={decision:'modify',onset:v,reviewed_at:nowISO()};save();closeModal();next();}
function closeModal(){document.getElementById('modal').style.display='none';mchart=null;tip.style.display='none';}
function go(d){idx=Math.min(view.length-1,Math.max(0,idx+d));render();}
function next(){if(idx<view.length-1){idx++;}render();}

function curSel(s){const d=dec[s.filename];
  if(d&&d.decision==='modify')return {sel:d.onset,label:'확정(수정)'};
  if(d&&d.decision==='approve')return {sel:d.onset,label:'확정(승인)'};
  if(picked!=null)return {sel:picked,label:'선택 후보'};
  return {sel:null,label:''};}

function render(){rebuild();const s=view[idx];const total=SESSIONS.length;
  const done=Object.keys(dec).length;
  document.getElementById('prog').textContent=(idx+1)+' / '+view.length+(filt?' ('+filt+')':'')+'  · 판정 '+done+'/'+total;
  document.getElementById('remain').textContent=(total-done);
  if(!s){document.getElementById('meta').innerHTML='세션 없음';document.getElementById('info').innerHTML='';return;}
  picked=null;document.getElementById('picked').textContent='—';
  const d=dec[s.filename];
  const dl=d?('<span class="tag" style="background:'+({approve:'#a5d6a7',exclude:'#ffcc99',modify:'#90caf9'}[d.decision])+'">'+d.decision+(d.onset!=null?' @'+d.onset:'')+(d.exclude_reason?' ('+d.exclude_reason+')':'')+'</span>'):'<span class="tag">미검수</span>';
  document.getElementById('meta').innerHTML=
    '<span class="fn">#'+s.rank+' '+s.filename+'</span>'+
    '<span class="tag '+(s.subtype.includes('WALK')?'walk':'')+'">'+s.subtype+'</span>'+
    '<span class="tag">'+s.split+'</span>'+'<span class="tag">prio '+s.priority+'</span>'+
    (s.hard?'<span class="tag hard">hard: '+s.hard+'</span>':'')+
    (s.soft?'<span class="tag soft">soft: '+s.soft+'</span>':'')+dl;
  // chart
  const cv=document.getElementById('chart');
  attachChart(cv,s,{getSel:()=>curSel(s).sel, selLabel:curSel(s).label,
    onPick:(f)=>{picked=f;document.getElementById('picked').textContent=f+' (clean '+toClean(f)+') — M 키로 수정 저장';cv._redraw({sel:f,selLabel:'선택 후보'});}});
  // info
  let cand='';s.topk_orig.forEach((o,i)=>{cand+='<span class="cand" onclick="quickModify('+o+')">topk'+(i+1)+': '+o+' (clean '+toClean(o)+')</span>';});
  if(s.rise_orig!=null)cand='<span class="cand" onclick="quickModify('+s.rise_orig+')">auto rise: '+s.rise_orig+' (clean '+toClean(s.rise_orig)+')</span>'+cand;
  const rnf=s.hard.includes('rise_not_found');
  document.getElementById('info').innerHTML=
    '<div class="rec">추천 onset: '+(s.rec_orig!=null?(s.rec_orig+' (clean '+s.rec_clean+')'):'없음 — 수정/제외 권장')+'</div>'+
    (rnf?'<div class="rnf">⚠ detector가 SEARCH['+RG.search[0]+'–'+RG.search[1]+'] 구간에서 onset 미검출(rise_not_found). '
      +'큰 모션이 BASELINE 구간에 있을 수 있음(WALK 패턴). <b>곡선을 직접 확인</b>해 솟는 지점을 onset 으로 지정하세요. '
      +'SEARCH 밖 frame도 지정 가능(경고만, 입력 막지 않음).</div>':'')+
    '<div>'+s.reco_text+'</div>'+
    '<div style="margin-top:6px">힌트: '+s.hint+'</div>'+
    '<div style="margin-top:6px">후보(클릭=수정창 입력): '+(cand||'<i>없음</i>')+'</div>'+
    '<div class="note">Recommendation only — 최종 판정은 진규. 곡선 '+(CURVES[s.filename]?'OK':'없음(PNG 폴백)')+'.</div>';
  // strip (색 + 글자)
  const st=document.getElementById('strip');st.innerHTML='';
  const LET={approve:'A',exclude:'C',modify:'M'};
  view.forEach((v,i)=>{const sp=document.createElement('span');const dd=dec[v.filename];
    if(dd){sp.className=dd.decision;sp.textContent=LET[dd.decision]||'';}else{sp.className='undone';sp.textContent='·';}
    if(i===idx)sp.className+=' cur';sp.title=v.filename;sp.onclick=()=>{idx=i;render();};st.appendChild(sp);});
}
function quickModify(o){openModify(o);}
function reset(){dec={};save();render();}
function exp(fmt){const rows=[];
  SESSIONS.forEach(s=>{const d=dec[s.filename];if(!d)return;
    const fo=d.onset!=null?d.onset:'';const fc=d.onset!=null?toClean(d.onset):'';
    rows.push({filename:s.filename,subtype:s.subtype,split_assignment:s.split,decision:d.decision,
      final_onset_frame_original:fo,final_onset_frame_clean:(fc==null?'':fc),
      exclude_reason:d.exclude_reason||'',recommended_onset:(s.rec_orig!=null?s.rec_orig:''),
      reviewer:'jinkyu',reviewed_at:d.reviewed_at});});
  let blob,name;
  if(fmt==='json'){blob=new Blob([JSON.stringify(rows,null,2)],{type:'application/json'});name='review_decisions.json';}
  else{const cols=Object.keys(rows[0]||{filename:''});
    const csv=[cols.join(',')].concat(rows.map(r=>cols.map(c=>('"'+String(r[c]).replace(/"/g,'""')+'"')).join(','))).join('\n');
    blob=new Blob(['﻿'+csv],{type:'text/csv'});name='review_decisions.csv';}
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=name;a.click();
  alert(rows.length+'건 export ('+name+'). finalization/review_tool/ 에 두면 Claude Code가 v2 생성.');}
document.addEventListener('keydown',e=>{if(document.getElementById('modal').style.display==='flex'){if(e.key==='Escape')closeModal();return;}
  if(e.target&&(e.target.tagName==='INPUT'||e.target.tagName==='SELECT'))return;
  if(e.key==='a'||e.key==='A')decide('approve');else if(e.key==='c'||e.key==='C')askExclude();
  else if(e.key==='m'||e.key==='M')openModify();else if(e.key==='ArrowLeft')go(-1);else if(e.key==='ArrowRight')go(1);});
document.getElementById('onsetIn').addEventListener('input',onsetChanged);
window.addEventListener('resize',()=>{const cv=document.getElementById('chart');if(cv&&cv._redraw)cv._redraw();if(mchart)mchart();});
// filter init
const subs=[...new Set(SESSIONS.map(s=>s.subtype))].sort();
const fsel=document.getElementById('filter');subs.forEach(x=>{const o=document.createElement('option');o.value=x;o.textContent=x+' ('+SESSIONS.filter(s=>s.subtype===x).length+')';fsel.appendChild(o);});
fsel.onchange=()=>{filt=fsel.value;idx=0;render();};
render();
</script></body></html>
"""

README = """# onset 검수 도구 v2 사용법 (인터랙티브 차트)

1. `onset_review.html` 을 브라우저(Chrome/Edge/Firefox)로 더블클릭해 연다.
   - v2 는 곡선을 JS 로 직접 그리므로(PNG 아님) **곡선은 file:// 에서도 보인다.**
   - 곡선 데이터는 HTML 안에 임베드됨(energy_curves.json 빌드시 포함).
2. 차트 보는 법:
   - **파란 굵은 선 = sparse energy(판단 기준).** 낙상처럼 확 솟는 지점이 onset.
   - 회색 점선 thr / 주황 점선 auto rise = 참고용.
   - 구간: BASELINE(빗금)·SEARCH(파란빗금, 유효 onset 구간)·BEEP(주황 교차빗금) — 글자 라벨 병기.
   - x축 눈금: 1(작은선)·10(중간+숫자)·100(굵은+숫자) 3단계.
   - 마우스 올리면 frame·energy 표시. 클릭하면 '선택 후보'(초록선) → M 키로 그 값 수정 저장.
3. 키보드: A=승인 / C=제외 / M=수정 / ←→=이동. 또는 버튼.
   - 승인: 추천 onset 그대로 확정 → 자동 다음. (rise_not_found 세션은 추천 없음 → 수정/제외)
   - 제외: 사유 선택 → 제외 확정.
   - 수정: 모달에서 그래프 클릭/후보 클릭/직접 입력 → **초록 세로선 실시간 이동** → 저장.
     SEARCH 밖 frame 도 지정 가능(경고만, 입력 막지 않음).
4. 매 판정마다 localStorage 자동저장(닫았다 열어도 복구).
5. 끝나면 상단 **Export CSV** → `review_decisions.csv` 다운로드.
   그 파일을 `finalization/review_tool/` 에 두고 Claude Code 에게 "v2 생성" 요청.

## 곡선이 안 보이면
- 곡선은 HTML 임베드라 보통 file:// 더블클릭으로 충분.
- 만약 특정 세션이 'PNG 폴백'으로 뜨면 energy_curves.json 에 그 세션 곡선이 없는 것 →
  PNG(../plots_priority/) 참조. 로컬 서버가 필요하면:
    cd debug/modeling/diag_out/onset_detector/finalization
    python -m http.server 8000
  브라우저에서 http://localhost:8000/review_tool/onset_review.html

판정은 진규. 추천은 참고용. 원본/동결 파일 무수정.
"""


def main():
    TOOL_DIR.mkdir(parents=True, exist_ok=True)
    sessions = build_sessions()
    if CURVES_JSON.exists():
        curves_text = CURVES_JSON.read_text(encoding="utf-8")
    else:
        curves_text = '{"meta":{},"curves":{}}'
        print(f"[경고] {CURVES_JSON} 없음 → 곡선 미임베드(전 세션 PNG 폴백). "
              f"먼저 export_energy_curves.py 실행 필요.")
    try:
        cobj = json.loads(curves_text)
        n_curves = len(cobj.get("curves", {}))
    except Exception:
        n_curves = 0
    html = (HTML
            .replace("__DATA__", json.dumps(sessions, ensure_ascii=False))
            .replace("__CURVES__", curves_text))
    (TOOL_DIR / "onset_review.html").write_text(html, encoding="utf-8")
    (TOOL_DIR / "README.md").write_text(README, encoding="utf-8")
    n_rec = sum(1 for s in sessions if s["rec_orig"] is not None)
    matched = sum(1 for s in sessions if s["filename"] in (json.loads(curves_text).get("curves", {})))
    print(f"[생성] {TOOL_DIR/'onset_review.html'}")
    print(f"  세션 {len(sessions)} (train/val high) | 추천 onset 있음 {n_rec} / 없음(manual) {len(sessions)-n_rec}")
    print(f"  곡선 임베드 {n_curves}개 | 세션 매칭 {matched}/{len(sessions)} | 미매칭은 PNG 폴백")
    from collections import Counter
    print(f"  subtype: {dict(Counter(s['subtype'] for s in sessions))}")
    print(f"[안내] 브라우저로 onset_review.html 열기. 곡선은 임베드라 file:// 에서도 표시.")


if __name__ == "__main__":
    main()
