// Three.js 轨道球展示模块
import * as THREE from 'three';

(function initEyeOrb() {
    const canvas = document.getElementById('eyeOrbCanvas');
    const widget = document.getElementById('eyeOrbWidget');
    if (!canvas || !widget) return;

    let W = 180, H = 180;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(40, W / H, 0.1, 100);
    camera.position.set(0, 0, 5.2);

    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
    renderer.setClearColor(0x000000, 0);
    renderer.setSize(W, H);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    const NOISE_GLSL = `
    float hash12(vec2 p){ vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
    float vnoise(vec2 p){
        vec2 i = floor(p); vec2 f = fract(p); vec2 u = f * f * (3.0 - 2.0 * f);
        float a = hash12(i); float b = hash12(i + vec2(1.0, 0.0));
        float c = hash12(i + vec2(0.0, 1.0)); float d = hash12(i + vec2(1.0, 1.0));
        return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
    }
    float fbm(vec2 p){ float v = 0.0; float a = 0.5; for(int i = 0; i < 4; i++){ v += a * vnoise(p); p = p * 2.03 + vec2(0.31); a *= 0.5; } return v; }`;

    const uniforms = {
        uTime: { value: 0 },
        uHit: { value: new THREE.Vector3(0, 0, 1) },   // 瞳孔固定在局部 +Z，靠网格旋转朝向鼠标
        uStrength: { value: 1.0 },
        uMouse: { value: new THREE.Vector3(0, 0, 1) }, // 彩虹反应固定在瞳孔周围
        uMouseStr: { value: 0.0 },
        uWave: { value: 0.8 }
    };

    const orbMat = new THREE.ShaderMaterial({
        uniforms,
        vertexShader: `
            uniform vec3 uHit;
            varying vec3 vPos; varying vec3 vWorldPos; varying vec3 vWorldNormal;
            void main(){
                vec3 nrm = normalize(position);
                vec3 Hv = normalize(uHit);
                float dHv = distance(nrm, Hv);
                float tt = clamp((dHv - 0.08) / 0.28, 0.0, 1.0);
                float ss = tt * tt * (3.0 - 2.0 * tt);
                float cc = 1.0 - ss;
                vec3 pos = position * (1.0 - 0.10 * cc * cc);
                float dsdt = 6.0 * tt * (1.0 - tt);
                float dDepth = -0.10 * 2.0 * cc * dsdt / 0.28;
                vec3 gradDir = normalize(Hv - nrm * dot(Hv, nrm) + vec3(0.0001));
                vec3 nPert = normalize(nrm + gradDir * (-dDepth) * 0.5);
                vPos = pos;
                vWorldNormal = normalize(mat3(modelMatrix) * nPert);
                vec4 wp = modelMatrix * vec4(pos, 1.0);
                vWorldPos = wp.xyz;
                gl_Position = projectionMatrix * viewMatrix * wp;
            }`,
        fragmentShader: `
            uniform float uTime; uniform vec3 uHit; uniform float uStrength;
            uniform vec3 uMouse; uniform float uMouseStr; uniform float uWave;
            varying vec3 vPos; varying vec3 vWorldPos; varying vec3 vWorldNormal;
            ${NOISE_GLSL}
            float hexSDF(vec2 p){ vec2 q = abs(p); return max(dot(q, vec2(0.8660254, 0.5)), q.y) - 1.0; }
            vec3 hexInfo(vec2 p){
                vec2 s = vec2(3.4641016, 2.0); vec2 o = vec2(1.7320508, 1.0);
                vec2 c1 = floor(p / s + 0.5) * s; vec2 c2 = floor((p - o) / s + 0.5) * s + o;
                float d1 = hexSDF(p - c1); float d2 = hexSDF(p - c2);
                float pick1 = step(abs(d1), abs(d2));
                return vec3(mix(c2, c1, pick1), mix(d2, d1, pick1));
            }
            float hexD(vec2 p){ return hexInfo(p).z; }
            void main(){
                vec3 n = normalize(vPos); vec3 wn = normalize(vWorldNormal);
                vec3 viewDir = normalize(cameraPosition - vWorldPos);
                float fres = pow(clamp(1.0 - abs(dot(viewDir, wn)), 0.0, 1.0), 2.0);
                float str = clamp(uStrength, 0.0, 1.3);
                vec3 H = normalize(uHit); float dH = distance(n, H);
                vec3 L1 = normalize(vec3(-0.5, 0.5, 0.4));
                float dif1 = clamp(dot(wn, L1) * 0.5 + 0.5, 0.0, 1.0);
                vec3 col = vec3(0.09, 0.55, 0.56);
                col = mix(col, vec3(0.38, 0.84, 0.82), pow(dif1, 1.8) * 0.55);
                col *= 0.92 + 0.18 * smoothstep(0.30, 0.70, fbm(n.xy * 2.0 + 7.7));
                float dif2 = clamp(dot(wn, normalize(vec3(-0.6, -0.5, 0.3))) * 0.5 + 0.5, 0.0, 1.0);
                col = mix(col, vec3(0.70, 0.62, 0.92), pow(dif2, 1.6) * 0.70);
                col = mix(col, vec3(0.70, 0.80, 0.55), 0.40 * smoothstep(0.50, 0.75, fbm(n.yz * 1.3 + 9.2)));
                col += vec3(0.3, 0.7, 0.7) * fbm(n.xy * 3.0 + uTime * 0.15) * 0.06;
                // 高光跟随视线方向（球转哪高光跟哪，不再固定白点）
                vec3 Lg = normalize(viewDir + vec3(0.0, 0.15, 0.0));
                float sheen = pow(clamp(dot(wn, Lg), 0.0, 1.0), 3.0);
                col += vec3(0.7, 0.95, 1.0) * sheen * 0.12;
                float spec2 = pow(clamp(dot(reflect(-viewDir, wn), Lg), 0.0, 1.0), 60.0);
                col += vec3(0.8, 0.9, 1.0) * spec2 * 0.15;
                // 中心瞳孔区域抑制高光，避免白点叠加
                float dHc = distance(n, normalize(uHit));
                spec2 *= smoothstep(0.10, 0.35, dHc);
                sheen *= smoothstep(0.08, 0.25, dHc);
                float rimR = 0.35 + 0.65 * smoothstep(-0.3, 0.9, wn.x);
                col += vec3(0.55, 1.0, 0.97) * pow(fres, 2.5) * rimR * 0.7;
                float d2 = max(dH - 0.60, 0.0);
                float p = mod(uTime * 0.5 - d2, 1.2);
                float ringProf = smoothstep(0.0, 0.12, p) * (1.0 - smoothstep(0.30, 0.55, p));
                ringProf *= smoothstep(0.0, 0.15, d2); ringProf *= 1.0 - smoothstep(1.4, 2.0, d2);
                ringProf *= 0.5 + 0.9 * uWave;
                col += vec3(0.7, 1.0, 1.0) * ringProf * 0.025;
                float dM = distance(n, normalize(uMouse));
                float mGlow = (1.0 - smoothstep(0.1, 0.6, dM)) * uMouseStr;
                float mHue = uTime * 0.15 + dM * 1.2;
                vec3 mouseCol = 0.5 + 0.5 * cos(6.28318 * (mHue + vec3(0.0, 0.33, 0.67)));
                mouseCol = clamp(mouseCol, 0.15, 1.0);
                col += mouseCol * mGlow * 0.10;
                float uu = atan(n.z, n.x); float vv = asin(clamp(n.y, -1.0, 1.0));
                vec2 hexUV = vec2(uu, vv) * 26.0;
                vec3 hi = hexInfo(hexUV); vec2 cid = hi.xy; float hd = hi.z;
                float edge = 1.0 - smoothstep(0.0, 0.10, abs(hd));
                float cellRnd = hash12(cid * 0.371);
                vec2 eOff = vec2(0.12, 0.10);
                float bevel = clamp((hexD(hexUV - eOff) - hexD(hexUV + eOff)) * 1.5, -1.0, 1.0);
                float edgeLit = edge * max(bevel, 0.0);
                float height = 1.0 - smoothstep(-0.55, -0.05, hd);
                float revealBase = 0.10 + 0.35 * smoothstep(0.40, 0.75, fbm(n.xy * 1.8 + n.z * 1.3 + 4.7));
                revealBase = max(revealBase, pow(fres, 1.5) * 0.35);
                float suppress = (1.0 - smoothstep(0.15, 0.60, dH)) * str;
                float reveal = revealBase * (1.0 - suppress * 0.9);
                col += vec3(0.55, 0.95, 0.95) * height * (0.05 + 0.09 * max(bevel, 0.0)) * (0.4 + 0.6 * reveal + ringProf * 0.4);
                float hexBright = reveal * (0.6 + 0.4 * cellRnd) + ringProf * 0.8 * (0.8 + 0.4 * cellRnd);
                hexBright *= 1.0 + mGlow * 0.5;
                col = mix(col, vec3(0.85, 1.0, 1.0), clamp(edgeLit * hexBright, 0.0, 1.0));
                col += vec3(0.85, 1.0, 1.0) * edge * hexBright * 0.35;
                col = mix(col, mouseCol, clamp(edge * mGlow, 0.0, 1.0) * 0.6);
                vec2 duv = hexUV * 5.0;
                float dl = length(fract(duv) - 0.5);
                float dots = 1.0 - smoothstep(0.15, 0.30, dl);
                float dotMask = step(0.60, cellRnd) * (1.0 - edge);
                col += vec3(0.9, 1.0, 1.0) * dots * dotMask * (0.05 + 0.05 * ringProf);
                vec3 upv = vec3(0.0, 1.0, 0.0);
                if (abs(H.y) > 0.9) upv = vec3(1.0, 0.0, 0.0);
                vec3 t1 = normalize(cross(H, upv) + vec3(0.0001));
                vec3 t2 = normalize(cross(H, t1) + vec3(0.0001));
                float ang = atan(dot(n, t2), dot(n, t1));
                float aBase = ang / 6.2831853 * 320.0 + uTime * 0.2 + dH * 1.2;
                float id = floor(aBase); float rnd = hash12(vec2(id, 7.31));
                float bow = (vnoise(vec2(id * 1.7, 3.3)) - 0.5) * 2.0;
                float wig = (vnoise(vec2(id * 5.1, dH * 9.0 - uTime * 4.0)) - 0.5) * 2.0;
                float ripple = sin(dH * 18.0 - uTime * 4.0 + id * 1.3) * 0.35;
                float grow = smoothstep(0.10, 0.7, dH);
                float a = aBase + (bow * 3.0 + wig * 1.4 + ripple) * grow;
                float fr = fract(a) - 0.5;
                float width = 0.22 + 0.34 * hash12(vec2(id, 2.2));
                float fiberWide = 1.0 - smoothstep(width * 0.4, width, abs(fr));
                float fiberCore = 1.0 - smoothstep(width * 0.10, width * 0.30, abs(fr));
                float fiberHalo = 1.0 - smoothstep(width * 0.8, width * 2.0, abs(fr));
                float rimF = max(fiberWide - fiberCore, 0.0);
                float len = 0.35 + rnd * 0.40;
                float env = smoothstep(0.10, 0.16, dH) * (1.0 - smoothstep(len * 0.7, len, dH));
                float flick = 0.7 + 0.3 * sin(uTime * 5.0 + rnd * 40.0);
                float strandB = (0.25 + 0.75 * hash12(vec2(id, 9.4))) * flick;
                float flow = 0.6 + 0.8 * vnoise(vec2(id * 3.7, dH * 10.0 - uTime * 4.5));
                float travel = 0.55 + 0.45 * sin(dH * 26.0 - uTime * 7.0 + id * 0.7);
                float prox = 0.4 + 0.5 * (1.0 - smoothstep(0.15, 0.55, dH));
                float burstBase = env * strandB * flow * travel * prox * str;
                float hue = rnd + dH * 0.35 + uTime * 0.05;
                vec3 strandCol = 0.50 + 0.50 * cos(6.28318 * (hue + vec3(0.00, 0.30, 0.60)));
                strandCol = clamp(strandCol, 0.10, 1.0); strandCol = pow(strandCol, vec3(1.2));
                vec3 fiberCol = mix(vec3(0.90, 0.95, 1.00), strandCol, smoothstep(0.06, 0.22, dH));
                vec3 rimCol = mix(strandCol, vec3(0.55, 0.40, 1.00), 0.35);
                col += strandCol * fiberHalo * burstBase * 0.45;
                col += fiberCol * fiberWide * burstBase * 1.3;
                col += vec3(1.0) * fiberCore * burstBase * 0.6;
                col += rimCol * rimF * burstBase * 1.2;
                vec2 sp = vec2(ang * 15.0, dH * 35.0);
                vec2 spF = fract(sp) - 0.5;
                float spRnd = hash12(floor(sp));
                float sparkle = step(0.93, spRnd) * (1.0 - smoothstep(0.05, 0.30, length(spF)));
                sparkle *= env * str * (0.5 + 0.5 * sin(uTime * 6.0 + spRnd * 40.0));
                col += mix(vec3(1.0), strandCol, 0.7) * sparkle * 1.2;
                float holeMask = 1.0 - smoothstep(0.12, 0.20, dH);
                vec3 holeCol = mix(vec3(0.06, 0.36, 0.40), vec3(0.12, 0.52, 0.55), smoothstep(0.0, 0.18, dH));
                col = mix(col, holeCol, holeMask * 0.9);
                col = min(col, vec3(1.0));
                gl_FragColor = vec4(col, 1.0);
            }`
    });

    const orb = new THREE.Mesh(new THREE.SphereGeometry(1.5, 160, 160), orbMat);
    scene.add(orb);

    // === 瞳孔方向：世界目标 -> 平滑 -> 网格四元数旋转（真 3D 转动，花纹跟随）===
    const Z = new THREE.Vector3(0, 0, 1);
    const targetDir = new THREE.Vector3(0, 0, 1);   // 鼠标即时方向
    const smoothDir = new THREE.Vector3(0, 0, 1);   // 慢速平滑方向
    const targetQuat = new THREE.Quaternion();
    let mouseStrTarget = 0;

    window.addEventListener('pointermove', (e) => {
        const rect = canvas.getBoundingClientRect();
        const cx = rect.left + rect.width / 2, cy = rect.top + rect.height / 2;
        const halfW = rect.width / 2 || W / 2, halfH = rect.height / 2 || H / 2;
        let dx = (e.clientX - cx) / halfW, dy = (e.clientY - cy) / halfH;
        const dist = Math.hypot(dx, dy);
        if (dist > 0.95) { dx = dx / dist * 0.95; dy = dy / dist * 0.95; }
        const zz = Math.sqrt(Math.max(0.01, 1 - dx * dx - dy * dy));
        targetDir.set(dx, -dy, zz).normalize();
        mouseStrTarget = Math.min(0.65, mouseStrTarget + 0.04 + Math.max(0, 1 - dist / 1.2) * 0.06);
    }, { passive: true });

    function syncSize() {
        const rect = widget.getBoundingClientRect();
        const nw = Math.max(2, Math.round(rect.width)), nh = Math.max(2, Math.round(rect.height));
        if (nw === W && nh === H) return;
        W = nw; H = nh;
        camera.aspect = W / H;
        camera.updateProjectionMatrix();
        renderer.setSize(W, H);
        camera.position.z = widget.classList.contains('expanded') ? 5.6 : 5.2;
    }

    const clock = new THREE.Clock();
    (function animate() {
        requestAnimationFrame(animate);
        syncSize();
        const t = clock.getElapsedTime();
        uniforms.uTime.value = t;

        // 慢速跟随：先平滑方向，再 slerp 旋转，双层缓动（更慢更柔）
        smoothDir.lerp(targetDir, 0.018).normalize();
        targetQuat.setFromUnitVectors(Z, smoothDir);
        orb.quaternion.slerp(targetQuat, 0.045);

        mouseStrTarget *= 0.96;
        uniforms.uMouseStr.value += (mouseStrTarget - uniforms.uMouseStr.value) * 0.1;
        uniforms.uWave.value += (0.8 - uniforms.uWave.value) * 0.04;
        orb.position.y = Math.sin(t * 0.5) * 0.04;
        renderer.render(scene, camera);
    })();

    window.addEventListener('resize', syncSize);
})();