/* Artwork sampled into particles that gather and disperse; no model inference. */
(() => {
  const canvas = document.getElementById("vf-light");
  const gl = canvas?.getContext("webgl", { alpha: false, antialias: false, depth: false });
  if (!gl) return; // Keep the static artwork when GPU rendering is unavailable.
  const art = document.querySelector(".vf-art");
  const portrait = art.querySelector(".vf-portrait");
  const images = [...portrait.querySelectorAll("img")];
  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
  const durations = { rest: .3, denoise: 5.2, hold: 3, dissolve: 3.2 };
  const ease = t => t * t * (3 - 2 * t);
  let scene = 0;
  let stage = "rest";
  let clarity = 0;
  let elapsed = 0;
  let time = 0;
  let last = 0;
  let lastDraw = 0;
  let frame = 0;
  let running = !reducedMotion.matches;
  let visible = true;
  let ready = false;
  let failed = false;
  let scenes = [];

  function shader(type, source) {
    const result = gl.createShader(type);
    gl.shaderSource(result, source);
    gl.compileShader(result);
    if (!gl.getShaderParameter(result, gl.COMPILE_STATUS)) {
      const message = gl.getShaderInfoLog(result);
      gl.deleteShader(result);
      throw new Error(message);
    }
    return result;
  }
  let program;
  try {
    const vertex = shader(gl.VERTEX_SHADER, `
      precision highp float;
      attribute vec2 target;
      attribute vec2 origin;
      attribute vec3 color;
      attribute float seed;
      uniform vec2 crop;
      uniform float clarity;
      uniform float time;
      uniform float pointSize;
      varying mediump vec3 radiance;
      void main() {
        // Every visible point keeps its own image sample throughout the cycle.
        // Slightly staggered settling reveals structure continuously, not at a cut.
        float progress = clamp((clarity - seed * 0.12) / (1.0 - seed * 0.12), 0.0, 1.0);
        float settled = progress * progress * (3.0 - 2.0 * progress);
        float disorder = 1.0 - settled;
        float angle = seed * 6.283185;
        vec2 drift = vec2(sin(time * 0.45 + angle), cos(time * 0.38 + angle)) * 0.012;
        vec2 arc = vec2(sin(angle + progress * 3.0), cos(angle + progress * 3.0));
        vec2 position = mix(origin + drift, target, settled);
        position += arc * sin(progress * 3.141593) * disorder * 0.055;
        gl_Position = vec4((position - 0.5) * 2.0 / crop, 0.0, 1.0);
        gl_PointSize = max(1.0, pointSize * mix(1.25, 1.0, settled));
        float appear = smoothstep(0.0, 0.18, clarity);
        radiance = color * appear * mix(0.14, 1.0, settled);
        radiance *= pointSize * pointSize / (gl_PointSize * gl_PointSize);
      }
    `);
    const fragment = shader(gl.FRAGMENT_SHADER, `
      precision mediump float;
      uniform float lightMode;
      varying mediump vec3 radiance;
      void main() {
        vec2 p = gl_PointCoord * 2.0 - 1.0;
        float kernel = 1.0 - smoothstep(0.15, 1.0, dot(p, p));
        float density = dot(radiance, vec3(0.2126, 0.7152, 0.0722)) / 0.85;
        vec3 ink = (vec3(250.0 / 255.0) - vec3(0.23, 0.25, 0.27)) * density;
        // Add light to the dark page; subtract ink from the white page.
        gl_FragColor = vec4(mix(radiance, ink, lightMode) * kernel * 0.86, 1.0);
      }
    `);
    program = gl.createProgram();
    gl.attachShader(program, vertex);
    gl.attachShader(program, fragment);
    gl.linkProgram(program);
    gl.deleteShader(vertex);
    gl.deleteShader(fragment);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
    gl.useProgram(program);
  } catch (error) {
    console.warn("Overview animation unavailable:", error);
    return;
  }
  const uniforms = Object.fromEntries(["crop", "clarity", "time", "pointSize", "lightMode"].map(name => [name, gl.getUniformLocation(program, name)]));
  const attributes = [["target", 2, 0], ["origin", 2, 2], ["color", 3, 4], ["seed", 1, 7]].map(([name, size, offset]) => ({ location: gl.getAttribLocation(program, name), size, offset }));
  attributes.forEach(({ location }) => gl.enableVertexAttribArray(location));
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.ONE, gl.ONE);

  function gaussianPair() {
    const radius = Math.sqrt(-2 * Math.log(1 - Math.random()));
    const angle = Math.random() * Math.PI * 2;
    return [radius * Math.cos(angle), radius * Math.sin(angle)];
  }

  function particles(image) {
    // Sample once at load time. Frames only move an immutable vertex buffer on
    // the GPU; no full-image layer fades in behind the points.
    const width = 900;
    const height = Math.round(width * image.naturalHeight / image.naturalWidth);
    const source = document.createElement("canvas");
    source.width = width;
    source.height = height;
    const ctx = source.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(image, 0, 0, width, height);
    const pixels = ctx.getImageData(0, 0, width, height).data;
    const vertices = new Float32Array(width * height * 8);
    // Irregular Gaussian clusters, well inside the canvas, avoid a rectangular
    // or elliptical noise silhouette. They are independent of the image outline.
    const clouds = Array.from({ length: 7 }, () => [
      0.24 + Math.random() * 0.52, 0.24 + Math.random() * 0.52,
      0.04 + Math.random() * 0.035, 0.04 + Math.random() * 0.035,
    ]);
    let count = 0;
    const edge = t => ease(Math.max(0, Math.min(t / 0.12, 1)));
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const i = (y * width + x) * 4;
        const u = (x + 0.5) / width;
        const v = 1 - (y + 0.5) / height;
        const fade = edge(Math.min(u, 1 - u)) * edge(Math.min(v, 1 - v));
        const r = Math.max(pixels[i] - 16, 0) / 255 * fade;
        const g = Math.max(pixels[i + 1] - 27, 0) / 255 * fade;
        const b = Math.max(pixels[i + 2] - 34, 0) / 255 * fade;
        if (Math.max(r, g, b) < 0.018) continue;
        const cloud = clouds[Math.floor(Math.random() * clouds.length)];
        const [gx, gy] = gaussianPair();
        vertices.set([
          u, v, cloud[0] + gx * cloud[2], cloud[1] + gy * cloud[3],
          r, g, b, Math.random(),
        ], count * 8);
        count++;
      }
    }
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, vertices.subarray(0, count * 8), gl.STATIC_DRAW);
    return { buffer, count, width };
  }

  function draw() {
    if (!ready || failed) return;
    const image = images[scene];
    const aspect = canvas.width / canvas.height;
    const imageAspect = image.naturalWidth / image.naturalHeight;
    const lightMode = document.body.dataset.mdColorScheme !== "slate";
    const background = lightMode ? [250 / 255, 250 / 255, 250 / 255] : [16 / 255, 27 / 255, 34 / 255];
    gl.clearColor(...background, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.blendEquation(lightMode ? gl.FUNC_REVERSE_SUBTRACT : gl.FUNC_ADD);
    const model = scenes[scene];
    gl.bindBuffer(gl.ARRAY_BUFFER, model.buffer);
    attributes.forEach(({ location, size, offset }) => gl.vertexAttribPointer(location, size, gl.FLOAT, false, 32, offset * 4));
    const cropX = Math.min(1, aspect / imageAspect);
    gl.uniform2f(uniforms.crop, cropX, Math.min(1, imageAspect / aspect));
    gl.uniform1f(uniforms.pointSize, canvas.width / model.width / cropX * 1.6);
    gl.uniform1f(uniforms.clarity, clarity);
    gl.uniform1f(uniforms.time, time);
    gl.uniform1f(uniforms.lightMode, lightMode ? 1 : 0);
    gl.drawArrays(gl.POINTS, 0, model.count);
  }
  function announce() {
    portrait.setAttribute("aria-label", `${images[scene].alt} ${running ? "点击暂停动画" : "点击播放动画"}`);
  }
  function animate(now) {
    frame = 0;
    if (!running || !ready || !visible || document.hidden || failed) return;
    const delta = last ? Math.min((now - last) / 1000, .1) : 0;
    last = now;
    time += delta;
    elapsed += delta;
    if (elapsed >= durations[stage]) {
      elapsed -= durations[stage];
      if (stage === "rest") stage = "denoise";
      else if (stage === "denoise") { stage = "hold"; clarity = 1; }
      else if (stage === "hold") stage = "dissolve";
      else {
        // Swap only after every particle has faded into the page background.
        scene = (scene + 1) % images.length;
        stage = "rest";
        clarity = 0;
        announce();
      }
    }
    const t = Math.min(elapsed / durations[stage], 1);
    if (stage === "denoise") clarity = ease(t);
    else if (stage === "dissolve") clarity = 1 - ease(t);
    if (!lastDraw || now - lastDraw >= 1000 / 60) {
      draw();
      lastDraw = now - ((now - lastDraw) % (1000 / 60));
    }
    frame = requestAnimationFrame(animate);
  }
  function schedule() {
    cancelAnimationFrame(frame);
    last = 0;
    lastDraw = 0;
    if (running && ready && visible && !document.hidden && !failed) frame = requestAnimationFrame(animate);
  }
  function load() {
    if (ready || images.some(image => !image.complete || !image.naturalWidth)) return;
    scenes = images.map(particles);
    ready = true;
    if (reducedMotion.matches) { clarity = 1; stage = "hold"; }
    draw();
    art.dataset.ready = "true";
    portrait.disabled = false;
    images.forEach(image => image.setAttribute("aria-hidden", "true"));
    announce();
    schedule();
  }
  portrait.addEventListener("click", () => { running = !running; announce(); schedule(); });
  // Also repaint a paused or reduced-motion scene when the theme changes.
  new MutationObserver(draw).observe(document.body, { attributes: true, attributeFilter: ["data-md-color-scheme"] });
  new ResizeObserver(() => {
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const dpr = Math.min(devicePixelRatio || 1, 1.5);
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
    gl.viewport(0, 0, canvas.width, canvas.height);
    draw();
  }).observe(canvas);
  new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; schedule(); }).observe(canvas);
  document.addEventListener("visibilitychange", schedule);
  reducedMotion.addEventListener("change", () => {
    if (reducedMotion.matches) { running = false; announce(); schedule(); }
  });
  canvas.addEventListener("webglcontextlost", event => {
    event.preventDefault();
    failed = true;
    cancelAnimationFrame(frame);
    art.removeAttribute("data-ready");
    canvas.hidden = true;
    portrait.disabled = true;
    images[0].removeAttribute("aria-hidden");
  });
  window.addEventListener("pagehide", () => {
    cancelAnimationFrame(frame);
  });
  window.addEventListener("pageshow", schedule);
  images.forEach(image => image.addEventListener("load", load, { once: true }));
  load();
})();
