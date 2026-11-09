"""Build a portable local HTML report comparing raw Dinosaur motion and policy.

Example: python tools/build_amp_comparison.py --checkpoint CKPT --out results/blinkamp/seed1_105000
Evaluation records must exist at OUT/evaluation.json and OUT/assets/model_command_*.npz;
use blinkamp.evaluate --record-all --record OUT/assets/model.npz --out OUT/evaluation.json.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from blinkmimic.robot import get_robot
from blinkmimic.rotations import quat_to_mat

ROOT = Path(__file__).resolve().parents[1]


def export_experts(out):
    info = json.loads((ROOT/'configs/blinkamp/motions.json').read_text())
    robot = get_robot('g1')
    order = [info['joint_names'].index(name) for name in robot.joint_names]
    entries = []
    for entry in info['motions']:
        if entry['style'] != 'dinosaur':
            continue
        path = ROOT/entry['path']
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry['sha256']:
            raise ValueError(f'Raw motion checksum mismatch: {path}')
        raw = torch.load(path,map_location='cpu',weights_only=True)
        # Preserve raw qpos values: no RSI floor offset, clipping, interpolation,
        # reference-state initialization or simulated control rollout.
        states = torch.cat((raw['base_position'],raw['base_quat'][:,[3,0,1,2]],
            raw['base_linear_velocity'],raw['base_angular_velocity'],
            raw['joint_position'][:,order],raw['joint_velocity'][:,order]),-1).numpy()
        direction = entry['direction']
        target = out/'assets'/f'expert_{direction}.npz'
        np.savez_compressed(target,states=states,dt=1/60,source=entry['path'],source_sha256=digest)
        R = quat_to_mat(raw['base_quat'][:,[3,0,1,2]])
        v = (raw['base_position'][1:]-raw['base_position'][:-1])*60
        local = (R[:-1].transpose(-1,-2) @ v[...,None]).squeeze(-1).mean(0).tolist()
        command_id = {'FW':1,'BW':2,'SW':3 if local[1]>=0 else 4}[direction]
        entries.append(dict(direction=direction,frames=len(states),fps=60,duration=len(states)/60,
            source=entry['path'],sha256=digest,mean_root_velocity=local,command_id=command_id))
    return sorted(entries,key=lambda x:['FW','BW','SW'].index(x['direction'])),info


def audit_html(out):
    path = out/'command_support_audit.json'
    if not path.exists():
        return ''
    audit = json.loads(path.read_text())
    rows = ''
    for direction in ('FW','BW','SW'):
        values = audit['clips'][direction]['frame_velocity']['mean']
        rows += '<tr><th>'+direction+'</th>'+''.join(f'<td>{v:.3f}</td>' for v in values)+'</tr>'
    coverage = audit['coverage']['planar_0.2_yaw_0.2']['command_fraction']
    return f"""<section class="section"><h2>采样诊断与待执行实验</h2><p>训练与自动守护已按要求停止。当前 command 均匀覆盖 vx/vy ±0.8 m/s、yaw ±0.5 rad/s，专家三段动作平均速度如下。</p><div class="table-wrap"><table><thead><tr><th>原始动作</th><th>平均 root vx · m/s</th><th>平均 root vy · m/s</th><th>源处理的平均 wz · rad/s</th></tr></thead><tbody>{rows}</tbody></table></div><p>抽样覆盖诊断：在平面速度差≤0.2 m/s、yaw差≤0.2 rad/s的条件下，约{coverage:.1%}的随机command能找到邻近专家窗口。该数值基于4096个专家窗口，不是严格支持域边界。慢速SW还涉及原版0.2死区问题。</p><p>计划四组对照：A 原采样＋完整AMP；B motion-derived command＋完整AMP；C 原采样＋屏蔽AMP速度；D 两者结合。奖励系数和其余训练预算固定，先各10k Actor更新筛选，再用配对种子确认。新实验尚未启动。</p><p><a href="experiment_plan.md">完整实验设计</a> · <a href="command_support_audit.json">离线诊断JSON</a></p></section>"""


def build_html(out,report,entries,provenance):
    esc = html.escape
    rows = ''
    labels = ['停止','前进','后退','侧移 y+','侧移 y−','转向 +','转向 −']
    for label,r in zip(labels,report['results']):
        def number(key):
            value = r.get(key)
            return '—' if value is None else f'{value:.3f}'
        rows += f'<tr><th>{label}</th><td>{esc(str(r["command"]))}</td><td>{r["survival_rate"]:.0%}</td><td class="{("warm" if (r.get("velocity_rmse_m_s") or 0)>.2 else "")}">{number("velocity_rmse_m_s")}</td><td class="{("warm" if (r.get("yaw_rmse_rad_s") or 0)>.2 else "")}">{number("yaw_rmse_rad_s")}</td><td>{number("foot_slip_rms_m_s")}</td></tr>'
    data = []
    for e in entries:
        j = e['command_id']
        r = report['results'][j]
        data.append(dict(**e,label={'FW':'前进','BW':'后退','SW':'侧移'}[e['direction']],
            expert=f'assets/expert_{e["direction"]}.mp4',model=f'assets/model_command_{j:02d}.mp4',
            expertPoster=f'assets/expert_{e["direction"]}.jpg',modelPoster=f'assets/model_command_{j:02d}.jpg',
            command=r['command'],velocity=r['velocity_rmse_m_s'],yaw=r['yaw_rmse_rad_s']))
    payload = json.dumps(data,ensure_ascii=False).replace('<','\\u003c')
    survival = sum(r['survival_rate']*report['replicas_per_command'] for r in report['results'])
    total = len(report['results'])*report['replicas_per_command']
    text = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dinosaur · 原始动作与 BlinkAMP 对照</title><style>
:root{color-scheme:light;--ink:#152c3a;--muted:#5e7180;--line:#d9e3e9;--accent:#175c88}*{box-sizing:border-box}body{margin:0;background:#f4f7f9;color:var(--ink);font:15px/1.7 system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1280px;margin:auto;padding:42px 26px 64px}.eyebrow{font-size:12px;letter-spacing:2px;color:var(--accent);font-weight:700}h1{font-size:clamp(27px,4vw,42px);line-height:1.2;letter-spacing:-1px;margin:12px 0}h2{font-size:22px;margin:0 0 16px}.intro,.muted{color:var(--muted)}.facts{display:flex;gap:10px;flex-wrap:wrap;margin:22px 0 32px}.pill{background:white;border:1px solid var(--line);padding:7px 13px;border-radius:30px}.toolbar{display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap;align-items:center;margin:18px 0}.tabs,.controls{display:flex;gap:8px;flex-wrap:wrap}button,select{font:inherit;padding:7px 16px;border:1px solid var(--line);border-radius:8px;background:white;color:var(--ink);cursor:pointer}button[aria-selected=true],button.primary{background:var(--accent);border-color:var(--accent);color:white}button:focus-visible,select:focus-visible{outline:3px solid #77b4df;outline-offset:2px}.players{display:grid;grid-template-columns:1fr 1fr;gap:20px}.card{border:1px solid var(--line);border-radius:14px;background:white;overflow:hidden}.card-header{padding:16px 20px;display:flex;justify-content:space-between;gap:10px;align-items:center}.tag{font-size:12px;color:var(--muted)}video{width:100%;aspect-ratio:16/9;display:block;background:#e5edf1}.caption{padding:16px 20px;min-height:95px;font-size:14px}.caption strong{display:block}.note{padding:16px 20px;border-left:3px solid #8eafbF;background:#eaf0f4;color:#4e6675;border-radius:0 8px 8px 0;margin:22px 0 34px}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:12px;background:white}table{width:100%;border-collapse:collapse;white-space:nowrap}th,td{padding:13px 17px;text-align:left;border-bottom:1px solid #e8edf1}thead{background:#edf3f6;font-size:13px}tbody th{font-weight:500}tr:last-child td,tr:last-child th{border-bottom:0}.warm{background:#fff1d9;color:#87500a;font-weight:650}.section{margin-top:38px}.protocol{display:grid;grid-template-columns:1fr 1fr;gap:20px}.protocol>div{background:white;border:1px solid var(--line);padding:20px;border-radius:12px}p{margin:8px 0}a{color:var(--accent);text-underline-offset:3px}code{overflow-wrap:anywhere;font-size:12px}details{margin-top:20px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eaf0f4;padding:20px;border-radius:12px;font-size:12px}footer{margin-top:36px;color:var(--muted);font-size:13px}@media(max-width:760px){main{padding:26px 16px}.players,.protocol{grid-template-columns:1fr}.caption{min-height:0}th,td{padding:10px 12px}}
</style><main><header><div class="eyebrow">BLINKAMP / MOTION REVIEW</div><h1>Dinosaur：原始动作与策略对照</h1><p class="intro">把训练数据与真实仿真放在一起，检查身体前倾、手臂姿态、关节扭转和步态差异。</p><div class="facts"><span class="pill">Seed __SEED__ · __ITER__ iteration</span><span class="pill">EMA 策略</span><span class="pill">__SURVIVAL__ / __TOTAL__ 实例存活 20 秒</span><span class="pill">Pelvis / root 坐标</span></div></header>
<section aria-label="视频对照"><div class="toolbar"><div class="tabs" role="tablist" aria-label="动作方向"><button role="tab" aria-selected="true" data-motion="0">前进</button><button role="tab" aria-selected="false" data-motion="1">后退</button><button role="tab" aria-selected="false" data-motion="2">侧移</button></div><div class="controls"><button class="primary" id="play">同时播放</button><button id="reset">回到起点</button><select id="speed" aria-label="播放速度"><option value="0.25">0.25× 慢放</option><option value="0.5">0.5× 慢放</option><option value="1" selected>1× 原速</option><option value="1.5">1.5×</option></select></div></div>
<div class="players"><article class="card"><div class="card-header"><strong>原始数据集</strong><span class="tag">PhysHSI · 60 Hz</span></div><video id="expert" controls playsinline muted loop preload="metadata"></video><div class="caption"><strong id="expertTitle"></strong><span id="expertMeta" class="muted"></span></div></article><article class="card"><div class="card-header"><strong>BlinkAMP 策略</strong><span class="tag">真实仿真 · 50 Hz</span></div><video id="model" controls playsinline muted loop preload="metadata"></video><div class="caption"><strong id="modelTitle"></strong><span id="modelMeta" class="muted"></span></div></article></div><p id="state" class="muted" role="status" aria-live="polite"></p>
<div class="note">原始动作直接回放 .pt 中的根位姿与关节位置，不进行 RSI 高度偏移、关节限幅或物理控制；原始世界高度可能显示为离地，不能据此判断物理稳定性。策略视频来自独立的真实物理仿真。两者使用同一 G1 模型和相对 root 朝向的相机，视点跟随骨盆高度，视频帧率一致；速度、动作周期和起始相位并未匹配，“同时播放”仅用于同屏观察，不是逐帧误差对齐。视频各自循环。</div></section>
<section class="section"><h2>七类指令评估</h2><p class="muted">每类 __REPLICAS__ 个独立初始化实例，持续 __SECONDS__ 秒。RMSE 汇总全部实例，视频展示每组第 1 个实例。侧移使用 root 坐标 y 方向。</p><div class="table-wrap"><table><thead><tr><th>指令</th><th>[vx, vy, yaw]</th><th>20 秒存活率</th><th>平面速度 RMSE<br>m/s</th><th>转向 RMSE<br>rad/s</th><th>足部滑移 RMS<br>m/s</th></tr></thead><tbody>__ROWS__</tbody></table></div><p class="muted">存活率说明稳定性，不代表风格相似度。AMP 判别分数不作为本页风格质量结论。</p></section>
__AUDIT__<section class="section"><h2>对照说明与数据来源</h2><div class="protocol"><div><strong>原始动作</strong><p>三段 Dinosaur FW / BW / SW，完整时长约 24.95 秒。输入为原始 60 Hz 轨迹，视频以 25 fps 取帧显示，不改变播放速度。源关节顺序按名称映射至本项目模型。</p><p>报告中的原始视频是运动学回放，不是原始 PhysHSI 训练器的评估结果。</p></div><div><strong>策略评估</strong><p>检查点：seed __SEED__ / __ITER__。固定随机种子，源默认姿态及初始化随机性；观测噪声与 episode 外部扰动关闭，测试指令保持固定。</p><p>所有录像来自同一次七指令评估，完整原始指标和校验信息随报告保存。</p><a href="evaluation.json">完整评估 JSON</a> · <a href="provenance.json">来源与 SHA-256</a></div></div><details><summary>展开复现信息</summary><pre>__PROVENANCE__</pre></details></section><footer>本报告仅在本机生成，无外部脚本或网络依赖。分享时请保留整个文件夹。数据来源 PhysHSI（源仓库声明 CC-BY-NC-SA-4.0，保留底层数据条款）；源版本与各文件校验和见来源记录。</footer></main>
<script>const motions=__DATA__;const expert=document.getElementById('expert'),model=document.getElementById('model');let selected=0;const videos=[expert,model],play=document.getElementById('play'),speed=document.getElementById('speed');function show(i){selected=i;const m=motions[i];videos.forEach(v=>v.pause());expert.src=m.expert;model.src=m.model;expert.poster=m.expertPoster;model.poster=m.modelPoster;videos.forEach(v=>{v.load();v.playbackRate=Number(speed.value)});document.querySelectorAll('[data-motion]').forEach(b=>b.setAttribute('aria-selected',Number(b.dataset.motion)===i));document.getElementById('expertTitle').textContent='Dinosaur_'+m.direction+'.pt';document.getElementById('expertMeta').textContent=m.frames+' 帧 · '+m.duration.toFixed(2)+' 秒 · 原始位姿';document.getElementById('modelTitle').textContent='指令 ['+m.command.join(', ')+']';document.getElementById('modelMeta').textContent='速度 RMSE '+m.velocity.toFixed(3)+' m/s · 转向 RMSE '+m.yaw.toFixed(3)+' rad/s';document.getElementById('state').textContent=m.label+'对照已载入。两侧未作相位或速度匹配。';play.textContent='同时播放'}document.querySelectorAll('[data-motion]').forEach(b=>b.addEventListener('click',()=>show(Number(b.dataset.motion))));play.addEventListener('click',async()=>{if(videos.some(v=>!v.paused)){videos.forEach(v=>v.pause());play.textContent='同时播放'}else{const results=await Promise.allSettled(videos.map(v=>v.play()));if(results.some(r=>r.status==='rejected'))document.getElementById('state').textContent='请点击视频自身的播放按钮。';else play.textContent='同时暂停'}});document.getElementById('reset').addEventListener('click',()=>{videos.forEach(v=>{v.pause();v.currentTime=0});play.textContent='同时播放'});speed.addEventListener('change',()=>videos.forEach(v=>v.playbackRate=Number(speed.value)));show(0);</script></html>'''
    replacements = {'__SEED__':str(provenance['seed']),'__ITER__':str(report['iteration']),'__SURVIVAL__':str(round(survival)),'__TOTAL__':str(total),
        '__REPLICAS__':str(report['replicas_per_command']),'__SECONDS__':str(report['seconds']),
        '__AUDIT__':audit_html(out),'__ROWS__':rows,'__PROVENANCE__':esc(json.dumps(provenance,ensure_ascii=False,indent=2)),'__DATA__':payload}
    for key,value in replacements.items():
        text = text.replace(key,value)
    (out/'index.html').write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--skip-render',action='store_true')
    args = parser.parse_args()
    out = args.out.resolve();(out/'assets').mkdir(parents=True,exist_ok=True)
    entries,manifest = export_experts(out)
    report = json.loads((out/'evaluation.json').read_text())
    if Path(report['checkpoint']).resolve()!=args.checkpoint.resolve():
        raise ValueError('Evaluation does not match requested checkpoint')
    jobs = []
    for e in entries:
        jobs.append((out/'assets'/f'expert_{e["direction"]}.npz',out/'assets'/f'expert_{e["direction"]}.mp4'))
        j=e['command_id'];jobs.append((out/'assets'/f'model_command_{j:02d}.npz',out/'assets'/f'model_command_{j:02d}.mp4'))
    def render(job):
        source,target=job
        if not source.exists():
            raise FileNotFoundError(f'{source}; evaluate with --record-all first')
        with target.with_suffix('.render.log').open('w') as log:
            subprocess.run([sys.executable,str(ROOT/'tools/render_amp.py'),str(source),str(target),'--root-view'],
                cwd=ROOT,env=dict(os.environ,OMP_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT,check=True)
        import imageio_ffmpeg
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-loglevel','error','-y','-ss','1','-i',str(target),
            '-frames:v','1',str(target.with_suffix('.jpg'))],check=True)
        print(target.name,flush=True)
    if not args.skip_render:
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(render,jobs))
    checkpoint_info = torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    provenance=dict(checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        policy=report['policy'],seed=checkpoint_info['config']['seed'],iteration=report['iteration'],source_revision=manifest['source_revision'],
        source_license=manifest['source_license'],experts=entries,source_pose_processing='raw positions/quaternions/joints; no RSI shift or clipping',
        renderer='same local G1 MJCF, 960x540 (encoded 960x544), camera distance2.8 azimuth135+root_yaw elevation-15, follow pelvis position and height',
        comparison='not phase-aligned or speed-matched; model videos are replica0 from the aggregate evaluation')
    (out/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
    plan_path = ROOT/'docs/blinkamp_style_ablation_plan.md'
    if plan_path.exists():
        (out/'experiment_plan.md').write_text(plan_path.read_text().replace('../results/blinkamp/seed1_105000/index.html','index.html'))
    build_html(out,report,entries,provenance)
    print(out/'index.html')


if __name__ == '__main__':
    main()
