"""onset 검수 도구 HTML 생성기 (read-only).

priority_review_queue.csv(98 train/val) + plots_priority/ 를 진규가 브라우저에서 빠르게
검수하는 단일 HTML 도구를 생성한다. 결정은 localStorage 자동저장 + Export(csv/json) →
Claude Code 가 그 파일 읽어 manifest_v2_manual_augmented 생성.

- 자기완결 HTML (queue 데이터 임베드, plot 은 상대경로 ../plots_priority/ 참조).
- 설치/서버 불필요(브라우저로 열면 동작). img 가 file:// 에서 안 뜨면 README 의 http.server 안내.
- 판정은 진규. 추천은 참고만.

제약: read-only(원본/동결 파일 무수정). 산출:
  debug/modeling/diag_out/onset_detector/finalization/review_tool/onset_review.html
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
<html lang="ko"><head><meta charset="utf-8"><title>SafeSignal onset 검수</title>
<style>
*{box-sizing:border-box} body{font-family:system-ui,'Malgun Gothic',sans-serif;margin:0;background:#f4f5f7;color:#1a1a1a}
#bar{position:sticky;top:0;background:#222;color:#fff;padding:8px 14px;display:flex;gap:14px;align-items:center;flex-wrap:wrap;z-index:10}
#bar b{font-size:15px} #bar .sp{flex:1}
button{cursor:pointer;border:0;border-radius:6px;padding:7px 12px;font-size:14px;font-weight:600}
.bA{background:#2e7d32;color:#fff}.bC{background:#c62828;color:#fff}.bM{background:#f9a825;color:#222}
.bnav{background:#555;color:#fff}.bx{background:#1565c0;color:#fff}
select{padding:6px;border-radius:6px;border:1px solid #ccc;font-size:14px}
#wrap{max-width:1180px;margin:14px auto;padding:0 14px}
#meta{display:flex;gap:16px;align-items:baseline;flex-wrap:wrap;margin-bottom:6px}
#meta .fn{font-size:18px;font-weight:700} .tag{font-size:12px;padding:2px 8px;border-radius:10px;background:#e0e0e0}
.tag.walk{background:#ffe0b2} .tag.hard{background:#ffcdd2} .tag.soft{background:#fff9c4}
#plot{width:100%;border:1px solid #ddd;background:#fff;border-radius:8px}
#info{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px;margin-top:10px;font-size:14px;line-height:1.6}
#info .rec{font-size:16px;font-weight:700;color:#1565c0}
#info .note{color:#888;font-size:12px}
.cand{display:inline-block;margin:3px;padding:5px 10px;background:#e3f2fd;border:1px solid #90caf9;border-radius:6px;cursor:pointer;font-size:13px}
#actions{margin-top:12px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
#strip{display:flex;gap:1px;margin-top:10px;flex-wrap:wrap}
#strip span{width:11px;height:11px;border-radius:2px;background:#ccc;cursor:pointer}
#strip span.approve{background:#2e7d32}#strip span.exclude{background:#c62828}#strip span.modify{background:#f9a825}
#strip span.cur{outline:2px solid #1565c0;outline-offset:1px}
#modal{display:none;position:fixed;inset:0;background:rgba(0,0,0,.6);z-index:50;align-items:center;justify-content:center}
#modalbox{background:#fff;border-radius:10px;padding:16px;max-width:1000px;width:94%;max-height:92vh;overflow:auto}
#modalbox img{width:100%;border:1px solid #ddd;border-radius:6px}
#onsetIn{font-size:18px;padding:8px;width:130px;border:2px solid #1565c0;border-radius:6px}
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
  <img id="plot" alt="plot">
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
  <img id="mplot" alt="plot">
  <p>onset frame (original 기준). 후보 클릭 시 자동입력:</p>
  <div id="mcands"></div>
  <p><input id="onsetIn" type="number"> <button class="bA" onclick="saveModify()">저장</button>
     <button class="bnav" onclick="closeModal()">취소</button></p>
</div></div>
<script>
const SESSIONS = __DATA__;
const LS = "onset_review_decisions_v1";
let dec = JSON.parse(localStorage.getItem(LS) || "{}");
let filt = "", view = [], idx = 0;
function toClean(o){o=Number(o);if(isNaN(o))return null;if(o>=50&&o<150)return o-50;if(o>=200&&o<400)return o-100;if(o>=450&&o<550)return o-150;return null;}
function save(){localStorage.setItem(LS, JSON.stringify(dec));}
function rebuild(){view = SESSIONS.filter(s=>!filt||s.subtype===filt);if(idx>=view.length)idx=Math.max(0,view.length-1);}
function nowISO(){return new Date().toISOString();}
function decide(kind){const s=view[idx];if(!s)return;
  if(kind==='approve'){ if(s.rec_orig==null){alert('추천 onset 없음 → 수정(M) 또는 제외(C)');return;}
    dec[s.filename]={decision:'approve',onset:s.rec_orig,reviewed_at:nowISO()};}
  else if(kind==='exclude'){const r=document.getElementById('exreason').value;
    dec[s.filename]={decision:'exclude',onset:null,exclude_reason:r,reviewed_at:nowISO()};}
  save();document.getElementById('exrow').style.display='none';next();}
function askExclude(){const e=document.getElementById('exrow');e.style.display=e.style.display==='flex'?'none':'flex';}
function openModify(){const s=view[idx];if(!s)return;
  document.getElementById('mtitle').textContent=s.filename+'  ('+s.subtype+')';
  document.getElementById('mplot').src='../plots_priority/'+s.plot;
  const mc=document.getElementById('mcands');mc.innerHTML='';
  const cands=[];if(s.rise_orig!=null)cands.push(['auto rise',s.rise_orig]);
  s.topk_orig.forEach((o,i)=>cands.push(['topk'+(i+1),o]));
  if(cands.length===0)mc.innerHTML='<i>후보 없음 — 직접 입력</i>';
  cands.forEach(([lbl,o])=>{const b=document.createElement('span');b.className='cand';
    b.textContent=lbl+': '+o+' (clean '+toClean(o)+')';b.onclick=()=>document.getElementById('onsetIn').value=o;mc.appendChild(b);});
  const cur=dec[s.filename];document.getElementById('onsetIn').value=(cur&&cur.onset!=null)?cur.onset:(s.rec_orig!=null?s.rec_orig:'');
  document.getElementById('modal').style.display='flex';}
function saveModify(){const s=view[idx];const v=parseInt(document.getElementById('onsetIn').value,10);
  if(isNaN(v)){alert('frame 숫자 입력');return;}
  dec[s.filename]={decision:'modify',onset:v,reviewed_at:nowISO()};save();closeModal();next();}
function closeModal(){document.getElementById('modal').style.display='none';}
function go(d){idx=Math.min(view.length-1,Math.max(0,idx+d));render();}
function next(){if(idx<view.length-1){idx++;}render();}
function render(){rebuild();const s=view[idx];const total=SESSIONS.length;
  const done=Object.keys(dec).length;
  document.getElementById('prog').textContent=(idx+1)+' / '+view.length+(filt?' ('+filt+')':'')+'  · 판정 '+done+'/'+total;
  document.getElementById('remain').textContent=(total-done);
  if(!s){document.getElementById('meta').innerHTML='세션 없음';document.getElementById('plot').src='';document.getElementById('info').innerHTML='';return;}
  const d=dec[s.filename];
  const dl=d?('<span class="tag" style="background:'+({approve:'#a5d6a7',exclude:'#ef9a9a',modify:'#fff59d'}[d.decision])+'">'+d.decision+(d.onset!=null?' @'+d.onset:'')+(d.exclude_reason?' ('+d.exclude_reason+')':'')+'</span>'):'<span class="tag">미검수</span>';
  document.getElementById('meta').innerHTML=
    '<span class="fn">#'+s.rank+' '+s.filename+'</span>'+
    '<span class="tag '+(s.subtype.includes('WALK')?'walk':'')+'">'+s.subtype+'</span>'+
    '<span class="tag">'+s.split+'</span>'+'<span class="tag">prio '+s.priority+'</span>'+
    (s.hard?'<span class="tag hard">hard: '+s.hard+'</span>':'')+
    (s.soft?'<span class="tag soft">soft: '+s.soft+'</span>':'')+dl;
  document.getElementById('plot').src='../plots_priority/'+s.plot;
  let cand='';s.topk_orig.forEach((o,i)=>{cand+='<span class="cand" onclick="quickModify('+o+')">topk'+(i+1)+': '+o+' (clean '+toClean(o)+')</span>';});
  if(s.rise_orig!=null)cand='<span class="cand" onclick="quickModify('+s.rise_orig+')">auto rise: '+s.rise_orig+' (clean '+toClean(s.rise_orig)+')</span>'+cand;
  document.getElementById('info').innerHTML=
    '<div class="rec">추천 onset: '+(s.rec_orig!=null?(s.rec_orig+' (clean '+s.rec_clean+')'):'없음 — 수정/제외 권장')+'</div>'+
    '<div>'+s.reco_text+'</div>'+
    '<div style="margin-top:6px">힌트: '+s.hint+'</div>'+
    '<div style="margin-top:6px">후보(클릭=수정창 입력): '+(cand||'<i>없음</i>')+'</div>'+
    '<div class="note">Recommendation only — 최종 판정은 진규.</div>';
  // strip
  const st=document.getElementById('strip');st.innerHTML='';
  view.forEach((v,i)=>{const sp=document.createElement('span');const dd=dec[v.filename];
    if(dd)sp.className=dd.decision;if(i===idx)sp.className+=' cur';sp.title=v.filename;sp.onclick=()=>{idx=i;render();};st.appendChild(sp);});
}
function quickModify(o){openModify();document.getElementById('onsetIn').value=o;}
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
  if(e.key==='a'||e.key==='A')decide('approve');else if(e.key==='c'||e.key==='C')askExclude();
  else if(e.key==='m'||e.key==='M')openModify();else if(e.key==='ArrowLeft')go(-1);else if(e.key==='ArrowRight')go(1);});
// filter init
const subs=[...new Set(SESSIONS.map(s=>s.subtype))].sort();
const fsel=document.getElementById('filter');subs.forEach(x=>{const o=document.createElement('option');o.value=x;o.textContent=x+' ('+SESSIONS.filter(s=>s.subtype===x).length+')';fsel.appendChild(o);});
fsel.onchange=()=>{filt=fsel.value;idx=0;render();};
render();
</script></body></html>
"""

README = """# onset 검수 도구 사용법

1. `onset_review.html` 을 브라우저(Chrome/Edge/Firefox)로 더블클릭해 연다.
2. 키보드: A=승인 / C=제외 / M=수정 / ←→=이동. 또는 버튼 클릭.
   - 승인: 추천 onset 그대로 확정 → 자동 다음.
   - 제외: 사유 선택(no_clear_transient/beep_misfire/walking_residual/other) → 제외 확정.
   - 수정: 모달에서 onset frame 직접 입력(후보 클릭=자동입력) → 저장.
3. 매 판정마다 브라우저 localStorage 에 자동저장(닫았다 열어도 복구).
4. 다 끝나면 상단 **Export CSV** → `review_decisions.csv` 다운로드.
   그 파일을 `finalization/review_tool/` 에 두고 Claude Code 에게 "v2 생성" 요청.

## plot 이 안 보이면 (file:// 보안 차단 시)
같은 폴더 상위(finalization/)에서 로컬 서버를 띄우고 접속:
    cd debug/modeling/diag_out/onset_detector/finalization
    python -m http.server 8000
브라우저에서 http://localhost:8000/review_tool/onset_review.html

판정은 진규. 추천은 참고용. 원본/동결 파일 무수정.
"""


def main():
    TOOL_DIR.mkdir(parents=True, exist_ok=True)
    sessions = build_sessions()
    html = HTML.replace("__DATA__", json.dumps(sessions, ensure_ascii=False))
    (TOOL_DIR / "onset_review.html").write_text(html, encoding="utf-8")
    (TOOL_DIR / "README.md").write_text(README, encoding="utf-8")
    n_rec = sum(1 for s in sessions if s["rec_orig"] is not None)
    print(f"[생성] {TOOL_DIR/'onset_review.html'}")
    print(f"  세션 {len(sessions)} (train/val high) | 추천 onset 있음 {n_rec} / 없음(manual) {len(sessions)-n_rec}")
    from collections import Counter
    print(f"  subtype: {dict(Counter(s['subtype'] for s in sessions))}")
    print(f"  plot 참조: ../plots_priority/ (상대경로)")
    print(f"[안내] 브라우저로 onset_review.html 열기. plot 안 뜨면 README의 http.server 사용.")


if __name__ == "__main__":
    main()
