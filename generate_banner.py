import os
import math
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from scipy.optimize import linear_sum_assignment

def build_banner():
    print("Step 1: Processing portrait image...")
    img_rgba = Image.open('d:/github/avatar_nobg.png')
    w, h = img_rgba.size
    bbox = img_rgba.getbbox()
    
    # Crop head + shoulders (target ratio 300 / 340)
    crop_w = int((bbox[3] - bbox[1]) * 0.82)
    if crop_w > w:
        crop_w = w
    crop_h = int(crop_w * (340 / 300))
    
    top = max(0, bbox[1] - 20)
    bottom = min(h, top + crop_h)
    if bottom - top < crop_h:
        top = max(0, bottom - crop_h)
        
    left = max(0, int((bbox[0] + bbox[2]) / 2 - crop_w / 2))
    right = min(w, left + crop_w)
    if right - left < crop_w:
        left = max(0, right - crop_w)
        
    cropped = img_rgba.crop((left, top, right, bottom)).resize((300, 340), Image.Resampling.LANCZOS)
    alpha = np.array(cropped.split()[-1])
    mask = alpha > 120
    
    # Preprocessing as per Master Prompt
    rgb = cropped.convert('RGB')
    rgb = ImageOps.autocontrast(rgb, cutoff=1)
    rgb = rgb.filter(ImageFilter.UnsharpMask(radius=3, percent=140))
    rgb = ImageEnhance.Contrast(rgb).enhance(1.3)
    gray = np.array(rgb.convert('L'), dtype=float)
    
    H, W = 340, 300
    
    # Dark Mode Dither
    arr_dark = gray.copy()
    arr_dark[~mask] = 0.0
    dither_dark = np.zeros((H, W), dtype=np.uint8)
    
    for y in range(H):
        x_range = range(W) if y % 2 == 0 else range(W - 1, -1, -1)
        direction = 1 if y % 2 == 0 else -1
        for x in x_range:
            if not mask[y, x]:
                arr_dark[y, x] = 0.0
                continue
            old_val = arr_dark[y, x]
            new_val = 255.0 if old_val >= 128.0 else 0.0
            dither_dark[y, x] = 1 if new_val == 255.0 else 0
            err = old_val - new_val
            
            nx = x + direction
            if 0 <= nx < W and mask[y, nx]:
                arr_dark[y, nx] += err * (7.0 / 16.0)
            if y + 1 < H:
                bx1 = x - direction
                if 0 <= bx1 < W and mask[y + 1, bx1]:
                    arr_dark[y + 1, bx1] += err * (3.0 / 16.0)
                if mask[y + 1, x]:
                    arr_dark[y + 1, x] += err * (5.0 / 16.0)
                bx2 = x + direction
                if 0 <= bx2 < W and mask[y + 1, bx2]:
                    arr_dark[y + 1, bx2] += err * (1.0 / 16.0)
                    
    # Light Mode Dither
    arr_light = 255.0 - gray
    arr_light[~mask] = 0.0
    dither_light = np.zeros((H, W), dtype=np.uint8)
    
    for y in range(H):
        x_range = range(W) if y % 2 == 0 else range(W - 1, -1, -1)
        direction = 1 if y % 2 == 0 else -1
        for x in x_range:
            if not mask[y, x]:
                arr_light[y, x] = 0.0
                continue
            old_val = arr_light[y, x]
            new_val = 255.0 if old_val >= 128.0 else 0.0
            dither_light[y, x] = 1 if new_val == 255.0 else 0
            err = old_val - new_val
            
            nx = x + direction
            if 0 <= nx < W and mask[y, nx]:
                arr_light[y, nx] += err * (7.0 / 16.0)
            if y + 1 < H:
                bx1 = x - direction
                if 0 <= bx1 < W and mask[y + 1, bx1]:
                    arr_light[y + 1, bx1] += err * (3.0 / 16.0)
                if mask[y + 1, x]:
                    arr_light[y + 1, x] += err * (5.0 / 16.0)
                bx2 = x + direction
                if 0 <= bx2 < W and mask[y + 1, bx2]:
                    arr_light[y + 1, bx2] += err * (1.0 / 16.0)
                    
    print(f"Dark mode dots: {np.sum(dither_dark)}, Light mode dots: {np.sum(dither_light)}")
    
    # Step 2: 3 Logos with Optimal Transport
    print("Step 2: Sampling and matching 3 logos...")
    lcx, lcy = 150.0, 170.0
    N = 900
    
    def make_react(cx, cy, N=900):
        pts = []
        n_center = 120
        r_c = 15
        for i in range(n_center):
            th = 2 * np.pi * i / n_center
            r = r_c * np.sqrt((i + 1) / n_center)
            pts.append((cx + r * np.cos(th), cy + r * np.sin(th)))
        n_ell = (N - n_center) // 3
        a, b = 76, 26
        for ang in [0, 60, 120]:
            rad = np.radians(ang)
            cp, sp = np.cos(rad), np.sin(rad)
            for i in range(n_ell):
                t = 2 * np.pi * i / n_ell
                xr = a * np.cos(t)
                yr = b * np.sin(t)
                pts.append((cx + xr * cp - yr * sp, cy + xr * sp + yr * cp))
        while len(pts) < N:
            pts.append(pts[-1])
        return np.array(pts[:N])
        
    def make_node(cx, cy, N=900):
        pts = []
        r = 75
        angles = np.radians([30, 90, 150, 210, 270, 330, 30])
        hex_x = cx + r * np.cos(angles)
        hex_y = cy + r * np.sin(angles)
        edge_pts = 550 // 6
        for i in range(6):
            x1, y1 = hex_x[i], hex_y[i]
            x2, y2 = hex_x[i+1], hex_y[i+1]
            for t in np.linspace(0, 1, edge_pts, endpoint=False):
                pts.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1)))
        n_pts = N - len(pts)
        seg_len = n_pts // 3
        for t in np.linspace(0, 1, seg_len):
            pts.append((cx - 26, cy + 32 - t * 64))
        for t in np.linspace(0, 1, seg_len):
            pts.append((cx - 26 + t * 52, cy - 32 + t * 64))
        for t in np.linspace(0, 1, n_pts - 2*seg_len):
            pts.append((cx + 26, cy + 32 - t * 64))
        return np.array(pts[:N])
        
    def make_ts(cx, cy, N=900):
        pts = []
        side = 136
        half = side / 2
        per_side = 460 // 4
        for t in np.linspace(-half, half, per_side):
            pts.append((cx + t, cy - half))
        for t in np.linspace(-half, half, per_side):
            pts.append((cx + half, cy + t))
        for t in np.linspace(half, -half, per_side):
            pts.append((cx + t, cy + half))
        for t in np.linspace(half, -half, 460 - 3*per_side):
            pts.append((cx - half, cy + t))
        t_pts = 190
        for t in np.linspace(cx - 44, cx - 10, t_pts // 3):
            pts.append((t, cy - 14))
        for t in np.linspace(cy - 14, cy + 34, t_pts - t_pts // 3):
            pts.append((cx - 27, t))
        s_pts = N - len(pts)
        h_len = s_pts // 5
        for t in np.linspace(cx + 40, cx + 10, h_len):
            pts.append((t, cy - 14))
        for t in np.linspace(cy - 14, cy + 10, h_len):
            pts.append((cx + 10, t))
        for t in np.linspace(cx + 10, cx + 40, h_len):
            pts.append((t, cy + 10))
        for t in np.linspace(cy + 10, cy + 34, h_len):
            pts.append((cx + 40, t))
        for t in np.linspace(cx + 40, cx + 10, s_pts - 4*h_len):
            pts.append((t, cy + 34))
        return np.array(pts[:N])

    pts_react = make_react(lcx, lcy, N)
    pts_node = make_node(lcx, lcy, N)
    pts_ts = make_ts(lcx, lcy, N)
    
    cost1 = np.linalg.norm(pts_react[:, None, :] - pts_node[None, :, :], axis=2)
    _, col1 = linear_sum_assignment(cost1)
    pts_node_m = pts_node[col1]
    
    cost2 = np.linalg.norm(pts_node_m[:, None, :] - pts_ts[None, :, :], axis=2)
    _, col2 = linear_sum_assignment(cost2)
    pts_ts_m = pts_ts[col2]
    
    # Step 3: Run-length paths and groupings
    def process_dither_to_svg_paths(dither_mat, num_bands=94):
        runs = []
        for y in range(H):
            in_run = False
            start_x = 0
            for x in range(W):
                if dither_mat[y, x] == 1:
                    if not in_run:
                        in_run = True
                        start_x = x
                else:
                    if in_run:
                        runs.append((start_x, y, x - start_x))
                        in_run = False
            if in_run:
                runs.append((start_x, y, W - start_x))
                
        np.random.seed(42)
        bands = [[] for _ in range(num_bands)]
        
        for (sx, sy, rlen) in runs:
            rx = sx + rlen / 2.0
            ry = float(sy)
            dx = 150.0 - rx
            dy = 170.0 - ry
            dist = math.hypot(dx, dy)
            angle = math.atan2(dy, dx)
            noisy_metric = angle + np.random.normal(0, 0.2) + (dist / 300.0)
            band_idx = int(abs(noisy_metric * 15)) % num_bands
            bands[band_idx].append((sx, sy, rlen, dx * 0.42, dy * 0.42))
            
        band_svg_elements = []
        for b_i, b_runs in enumerate(bands):
            if not b_runs:
                continue
            path_d = []
            mean_dx = np.mean([r[3] for r in b_runs])
            mean_dy = np.mean([r[4] for r in b_runs])
            for (sx, sy, rlen, _, _) in b_runs:
                path_d.append(f"M{sx} {sy}h{rlen}")
            d_str = "".join(path_d)
            
            st_intro = (b_i % 60) * 0.035
            elem = f"""    <g>
      <path d="{d_str}" stroke="currentColor" stroke-width="1" shape-rendering="crispEdges">
        <animate attributeName="opacity" dur="1.8s" begin="{st_intro:.2f}s" fill="freeze" from="0" to="1"/>
        <animate attributeName="opacity" dur="14.2s" begin="3.2s" repeatCount="indefinite"
          keyTimes="0;0.2113;0.3028;0.9085;1.0"
          values="1;1;0;0;1"/>
        <animateTransform attributeName="transform" type="translate" dur="14.2s" begin="3.2s" repeatCount="indefinite"
          keyTimes="0;0.2113;0.3028;0.9085;1.0"
          values="0,0;0,0;{mean_dx:.1f},{mean_dy:.1f};{mean_dx:.1f},{mean_dy:.1f};0,0"/>
      </path>
    </g>"""
            band_svg_elements.append(elem)
            
        return "\n".join(band_svg_elements)

    print("Step 4: Compiling SVG paths and travellers...")
    dark_portrait_svg = process_dither_to_svg_paths(dither_dark)
    light_portrait_svg = process_dither_to_svg_paths(dither_light)
    
    traveller_elements = []
    for i in range(N):
        rx, ry = pts_react[i]
        nx, ny = pts_node_m[i]
        tx, ty = pts_ts_m[i]
        
        vals_cx = f"{rx:.1f};{rx:.1f};{rx:.1f};{nx:.1f};{nx:.1f};{tx:.1f};{tx:.1f};{rx:.1f};{rx:.1f}"
        vals_cy = f"{ry:.1f};{ry:.1f};{ry:.1f};{ny:.1f};{ny:.1f};{ty:.1f};{ty:.1f};{ry:.1f};{ry:.1f}"
        
        dot = f"""    <circle r="1.3" fill="currentColor" opacity="0">
      <animate attributeName="opacity" dur="14.2s" begin="3.2s" repeatCount="indefinite"
        keyTimes="0;0.2113;0.3028;0.9085;0.96;1.0"
        values="0;0;1;1;0;0"/>
      <animate attributeName="cx" dur="14.2s" begin="3.2s" repeatCount="indefinite"
        keyTimes="0;0.2113;0.3028;0.4437;0.5352;0.6761;0.7676;0.9085;1.0"
        values="{vals_cx}"/>
      <animate attributeName="cy" dur="14.2s" begin="3.2s" repeatCount="indefinite"
        keyTimes="0;0.2113;0.3028;0.4437;0.5352;0.6761;0.7676;0.9085;1.0"
        values="{vals_cy}"/>
    </circle>"""
        traveller_elements.append(dot)
        
    travellers_svg = "\n".join(traveller_elements)
    
    # Step 5: Info Readout rows
    info_rows = [
        ("Subject", "Lu Quang Minh"),
        ("Role", "Full-Stack Developer"),
        ("Origin", "Ho Chi Minh City, Vietnam"),
        ("Education", "Sai Gon University"),
        ("Status", "Building + Learning + Shipping"),
        ("ToolChain", "VS Code, Git, Docker, Postman"),
        ("", ""),
        ("Core.Lang", "JavaScript, TypeScript"),
        ("Core.Frontend", "React, Next.js, Tailwind, Shadcn UI"),
        ("Core.Backend", "Node.js, NestJS"),
        ("Core.Database", "MySQL, PostgreSQL, SQL Server"),
        ("Core.Infra", "Docker"),
        ("", ""),
        ("Grid.Mail", "luminh2004@gmail.com"),
        ("Grid.Portfolio", "minhlq.dev"),
        ("Grid.LinkedIn", "linkedin.com/in/minhluquang"),
        ("Grid.GitHub", "github.com/minhluquang"),
        ("Grid.Facebook", "facebook.com/minhluquang")
    ]
    
    def generate_info_readout(label_color, dot_color, val_color, start_y=122, line_spacing=23):
        res = []
        y = start_y
        row_width = 705
        for label, val in info_rows:
            if not label:
                res.append(f'<line x1="435" y1="{y-6}" x2="1140" y2="{y-6}" stroke="{dot_color}" stroke-width="0.7" stroke-dasharray="3 4" stroke-opacity="0.4"/>')
                y += 14
                continue
                
            total_chars = 80
            used_chars = len(label) + len(val)
            dot_count = max(4, total_chars - used_chars - 2)
            dots = "." * dot_count
            
            row_svg = f"""  <text x="435" y="{y}" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" font-size="13.2" textLength="{row_width}" lengthAdjust="spacingAndGlyphs">
    <tspan fill="{label_color}" font-weight="600">{label}</tspan>
    <tspan fill="{dot_color}"> {dots} </tspan>
    <tspan fill="{val_color}">{val}</tspan>
  </text>"""
            res.append(row_svg)
            y += line_spacing
        return "\n".join(res)
        
    print("Step 6: Assembling dark.svg and light.svg...")
    
    dark_info_svg = generate_info_readout(label_color="#22D3EE", dot_color="#334155", val_color="#F8FAFC")
    light_info_svg = generate_info_readout(label_color="#0891B2", dot_color="#CBD5E1", val_color="#0F172A")
    
    # 1. DARK SVG
    dark_svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1180 610" width="100%" height="100%">
  <defs>
    <linearGradient id="dark-border-grad" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#22D3EE" stop-opacity="0.7"/>
      <stop offset="100%" stop-color="#0891B2" stop-opacity="0.3"/>
    </linearGradient>
    <linearGradient id="frame-grad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#1E293B" stop-opacity="0.6"/>
      <stop offset="100%" stop-color="#0F172A" stop-opacity="0.4"/>
    </linearGradient>
  </defs>

  <!-- Terminal Window Background -->
  <rect width="1180" height="610" rx="10" ry="10" fill="#0A101F" stroke="url(#dark-border-grad)" stroke-width="1.5"/>

  <!-- Title Bar -->
  <line x1="0" y1="42" x2="1180" y2="42" stroke="#22D3EE" stroke-width="0.8" stroke-opacity="0.25"/>
  <circle cx="24" cy="21" r="5.5" fill="#EF4444"/>
  <circle cx="42" cy="21" r="5.5" fill="#F59E0B"/>
  <circle cx="60" cy="21" r="5.5" fill="#10B981"/>
  <text x="590" y="26" text-anchor="middle" font-family="ui-monospace, monospace" font-size="13" fill="#94A3B8" letter-spacing="1">profile.sh --live</text>
  <text x="1145" y="26" text-anchor="end" font-family="ui-monospace, monospace" font-size="11" fill="#64748B">SESSION: 0x7F9A</text>

  <!-- LEFT PANEL: VISUAL.MAP -->
  <rect x="35" y="58" width="350" height="525" rx="6" fill="url(#frame-grad)" stroke="#22D3EE" stroke-width="1" stroke-opacity="0.35"/>
  <text x="50" y="80" font-family="ui-monospace, monospace" font-size="12" fill="#22D3EE" font-weight="bold" letter-spacing="1.5">VISUAL.MAP</text>
  <text x="365" y="80" text-anchor="end" font-family="ui-monospace, monospace" font-size="10.5" fill="#64748B">GRID 300x340</text>
  <line x1="50" y1="88" x2="365" y2="88" stroke="#22D3EE" stroke-width="0.6" stroke-opacity="0.25"/>

  <!-- Corner Crosshairs -->
  <path d="M55,108 h12 M55,108 v12 M365,108 h-12 M365,108 v12 M55,458 h12 M55,458 v-12 M365,458 h-12 M365,458 v-12" stroke="#22D3EE" stroke-width="1.2" stroke-opacity="0.6"/>

  <!-- Portrait Frame & Dots -->
  <g transform="translate(60, 114)" color="#A78BFA">
{dark_portrait_svg}
{travellers_svg}
  </g>

  <!-- Left Panel Telemetry Footer -->
  <text x="50" y="490" font-family="ui-monospace, monospace" font-size="10" fill="#64748B">DITHER: FLOYD-STEINBERG 1-BIT</text>
  <text x="50" y="506" font-family="ui-monospace, monospace" font-size="10" fill="#64748B">MORPH: OPTIMAL TRANSPORT (N=900)</text>
  <text x="50" y="522" font-family="ui-monospace, monospace" font-size="10" fill="#10B981">STATUS: STREAMING SYNC 60FPS</text>
  <line x1="50" y1="535" x2="365" y2="535" stroke="#22D3EE" stroke-width="0.6" stroke-opacity="0.2"/>
  <text x="210" y="558" text-anchor="middle" font-family="ui-monospace, monospace" font-size="11" fill="#A78BFA" letter-spacing="2">● LU QUANG MINH ●</text>

  <!-- RIGHT PANEL: SYSTEM.INFO -->
  <g transform="translate(0, 0)">
    <text x="435" y="80" font-family="ui-monospace, monospace" font-size="13" fill="#22D3EE" font-weight="bold" letter-spacing="1.8">SYSTEM.INFO</text>
    <circle cx="560" cy="76" r="4.5" fill="#EF4444">
      <animate attributeName="opacity" values="1;0.2;1" dur="2s" repeatCount="indefinite"/>
    </circle>
    <text x="572" y="80" font-family="ui-monospace, monospace" font-size="12" fill="#EF4444" font-weight="bold">LIVE</text>

    <rect x="995" y="64" width="145" height="24" rx="12" fill="#22D3EE" fill-opacity="0.12" stroke="#22D3EE" stroke-width="1"/>
    <circle cx="1010" cy="76" r="3.5" fill="#10B981"/>
    <text x="1075" y="80" text-anchor="middle" font-family="ui-monospace, monospace" font-size="12" fill="#22D3EE" font-weight="bold">@minhluquang</text>

{dark_info_svg}
  </g>
</svg>"""

    with open('d:/github/dark.svg', 'w', encoding='utf-8') as f:
        f.write(dark_svg)
    print("dark.svg created! Size:", len(dark_svg))
    
    # 2. LIGHT SVG
    light_svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1180 610" width="100%" height="100%">
  <defs>
    <linearGradient id="light-border-grad" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#0891B2" stop-opacity="0.8"/>
      <stop offset="100%" stop-color="#0284C7" stop-opacity="0.4"/>
    </linearGradient>
    <linearGradient id="light-frame-grad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#F1F5F9" stop-opacity="0.8"/>
      <stop offset="100%" stop-color="#E2E8F0" stop-opacity="0.5"/>
    </linearGradient>
  </defs>

  <rect width="1180" height="610" rx="10" ry="10" fill="#F8FAFC" stroke="url(#light-border-grad)" stroke-width="1.5"/>

  <line x1="0" y1="42" x2="1180" y2="42" stroke="#0891B2" stroke-width="0.8" stroke-opacity="0.25"/>
  <circle cx="24" cy="21" r="5.5" fill="#EF4444"/>
  <circle cx="42" cy="21" r="5.5" fill="#F59E0B"/>
  <circle cx="60" cy="21" r="5.5" fill="#10B981"/>
  <text x="590" y="26" text-anchor="middle" font-family="ui-monospace, monospace" font-size="13" fill="#475569" letter-spacing="1">profile.sh --live</text>
  <text x="1145" y="26" text-anchor="end" font-family="ui-monospace, monospace" font-size="11" fill="#94A3B8">SESSION: 0x7F9A</text>

  <rect x="35" y="58" width="350" height="525" rx="6" fill="url(#light-frame-grad)" stroke="#0891B2" stroke-width="1" stroke-opacity="0.35"/>
  <text x="50" y="80" font-family="ui-monospace, monospace" font-size="12" fill="#0891B2" font-weight="bold" letter-spacing="1.5">VISUAL.MAP</text>
  <text x="365" y="80" text-anchor="end" font-family="ui-monospace, monospace" font-size="10.5" fill="#64748B">GRID 300x340</text>
  <line x1="50" y1="88" x2="365" y2="88" stroke="#0891B2" stroke-width="0.6" stroke-opacity="0.25"/>

  <path d="M55,108 h12 M55,108 v12 M365,108 h-12 M365,108 v12 M55,458 h12 M55,458 v-12 M365,458 h-12 M365,458 v-12" stroke="#0891B2" stroke-width="1.2" stroke-opacity="0.6"/>

  <g transform="translate(60, 114)" color="#7C3AED">
{light_portrait_svg}
{travellers_svg}
  </g>

  <text x="50" y="490" font-family="ui-monospace, monospace" font-size="10" fill="#64748B">DITHER: FLOYD-STEINBERG 1-BIT</text>
  <text x="50" y="506" font-family="ui-monospace, monospace" font-size="10" fill="#64748B">MORPH: OPTIMAL TRANSPORT (N=900)</text>
  <text x="50" y="522" font-family="ui-monospace, monospace" font-size="10" fill="#059669">STATUS: STREAMING SYNC 60FPS</text>
  <line x1="50" y1="535" x2="365" y2="535" stroke="#0891B2" stroke-width="0.6" stroke-opacity="0.2"/>
  <text x="210" y="558" text-anchor="middle" font-family="ui-monospace, monospace" font-size="11" fill="#7C3AED" letter-spacing="2">● LU QUANG MINH ●</text>

  <g transform="translate(0, 0)">
    <text x="435" y="80" font-family="ui-monospace, monospace" font-size="13" fill="#0891B2" font-weight="bold" letter-spacing="1.8">SYSTEM.INFO</text>
    <circle cx="560" cy="76" r="4.5" fill="#EF4444">
      <animate attributeName="opacity" values="1;0.2;1" dur="2s" repeatCount="indefinite"/>
    </circle>
    <text x="572" y="80" font-family="ui-monospace, monospace" font-size="12" fill="#EF4444" font-weight="bold">LIVE</text>

    <rect x="995" y="64" width="145" height="24" rx="12" fill="#0891B2" fill-opacity="0.12" stroke="#0891B2" stroke-width="1"/>
    <circle cx="1010" cy="76" r="3.5" fill="#059669"/>
    <text x="1075" y="80" text-anchor="middle" font-family="ui-monospace, monospace" font-size="12" fill="#0891B2" font-weight="bold">@minhluquang</text>

{light_info_svg}
  </g>
</svg>"""

    with open('d:/github/light.svg', 'w', encoding='utf-8') as f:
        f.write(light_svg)
    print("light.svg created! Size:", len(light_svg))

if __name__ == "__main__":
    build_banner()
