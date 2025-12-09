//Writes depth from lights POV into depth texture
const SHADOW_WGSL = /* wgsl */`
struct ShadowUBO {
  lightViewProj: mat4x4f,
};
@group(0) @binding(0) var<uniform> shadowData : ShadowUBO;

struct ObjUBO {
  model: mat4x4f,
  n3x3: mat3x3f,
};
@group(1) @binding(0) var<uniform> obj : ObjUBO;

@vertex
fn vs(@location(0) p: vec3f) -> @builtin(position) vec4f {
//p is the vertex position of mesh
//object -> world
  let worldPos = obj.model * vec4f(p, 1.0);
  //word -> light clip
  return shadowData.lightViewProj * worldPos;
}

@fragment
fn fs() -> @location(0) vec4f {
  return vec4f(0.0, 0.0, 0.0, 1.0);
}
`;

//main forward pass (phong+shadows)
//uses shadow pass to darken pixels occluded from the light
const WGSL = /* wgsl */`
const PI: f32 = 3.14159265359;

struct DirectionLight {
  direction: vec4f, //direction to light 
  color: vec3f,
  _pad: f32,
};

struct PointLight {
  position: vec4f, //xyz pos
  color: vec3f,
  _pad: f32,
};

struct SpotLight {
  position: vec4f, //xyz
  direction: vec3f, //where the spot points
  _pad1: f32,
  color: vec3f,
  _pad2: f32,
  cutOff: f32, //inner cone cosine
  outerCutOff: f32, //outer cone cosine
};

//struct holding all the lights + counts
struct ComplexLightSystem {
  ambient: vec3f,
  _pad0: f32,
  numPoint: u32,
  numDir: u32,
  numSpot: u32,
  _pad1: u32,
  pointLights: array<PointLight, 10>,
  dirLights: array<DirectionLight, 10>,
  spotLights: array<SpotLight, 10>,
};

struct LightingSettings {
  lightingModel: u32,
};

@group(0) @binding(0) var<storage, read> lights : ComplexLightSystem; //storage buffer
@group(0) @binding(1) var shadowMap : texture_depth_2d; //depth texture from lights POV
@group(0) @binding(2) var shadowSampler : sampler_comparison; //comparison sampler (PCF)

//light matrix for sampling shadow map
struct ShadowUBO {
  lightViewProj: mat4x4f,
};
@group(2) @binding(0) var<uniform> shadowData : ShadowUBO; //uniform since its always the same
@group(3) @binding(0) var<uniform> settings : LightingSettings;

//material term from the .mtl
struct ObjectMaterial {
  Ka: vec4f,
  Ks: vec4f,
  params: vec4f,
};
@group(1) @binding(0) var<uniform> material : ObjectMaterial;
@group(1) @binding(3) var tex : texture_2d<f32>;
@group(1) @binding(4) var texSampler : sampler;
@group(1) @binding(5) var bumpTex : texture_2d<f32>;
@group(1) @binding(6) var bumpSampler : sampler;
@group(1) @binding(7) var roughTex : texture_2d<f32>;
@group(1) @binding(8) var metalTex : texture_2d<f32>;
@group(1) @binding(9) var emissiveTex : texture_2d<f32>;

//camera matrix + world pos
struct CameraUBO {
  viewProj: mat4x4f,
  camPos: vec3f,
};
@group(1) @binding(1) var<uniform> camera : CameraUBO;

//per object transforms
struct ObjUBO {
  model: mat4x4f, //object->world
  n3x3: mat3x3f, //normal matrix
};
@group(1) @binding(2) var<uniform> obj : ObjUBO;

//vertex output 
struct VSOut {
  @builtin(position) pos: vec4f, //clip space pos
  @location(0) nrm: vec3f, //world space normal
  @location(1) wp: vec3f, //world space position
  @location(2) base: vec3f, //base/diffuse color (vertex color as Kd)
  @location(3) shadowPos: vec3f, //shadow map UV (x,y) + depth 
  @location(4) uv: vec2f,
  @location(5) tangent: vec3f,
  @location(6) bitangent: vec3f,
};

@vertex
fn vs(@location(0) p: vec3f, @location(1) n: vec3f, @location(2) c: vec3f, @location(3) uv: vec2f) -> VSOut {
  var out: VSOut;
  //object->world
  let world = obj.model * vec4f(p, 1.0);
  //world->clip
  out.pos = camera.viewProj * world;
  //transform normal to world (using 3x3 matrix)
  let worldNormal = normalize(obj.n3x3 * n);
  out.nrm = worldNormal;
  
  //pass world position + vertex color
  out.wp = world.xyz;
  out.base = c; // Kd = vertex color
  out.uv = uv;

  var tangent = normalize(cross(vec3f(0.0, 1.0, 0.0), worldNormal));
  if (length(tangent) < 0.001) {
    tangent = normalize(cross(vec3f(1.0, 0.0, 0.0), worldNormal));
  }
  let bitangent = normalize(cross(worldNormal, tangent));
  out.tangent = tangent;
  out.bitangent = bitangent;
  
  // compute shadow map position
  let shadowPosH = shadowData.lightViewProj * world;
  // transform from clip space to texture space
  let shadowNDC = shadowPosH.xyz / shadowPosH.w;
  
  //NDC -> texture space
  out.shadowPos = vec3f(
    shadowNDC.xy * vec2f(0.5, -0.5) + vec2f(0.5, 0.5),
    shadowNDC.z
  );
  
  return out;
}

//phong : diffuse+specular
fn phong(N: vec3f, V: vec3f, L: vec3f, lightColor: vec3f, Kd: vec3f, specColor: vec3f, shininess: f32) -> vec3f {
  let diff = max(dot(N, L), 0.0);
  let R = reflect(-L, N);
  let spec = pow(max(dot(V, R), 0.0), shininess);
  return lightColor * (diff * Kd + spec * specColor);
}

fn blinnPhong(N: vec3f, V: vec3f, L: vec3f, lightColor: vec3f, Kd: vec3f, specColor: vec3f, shininess: f32) -> vec3f {
  let diff = max(dot(N, L), 0.0);
  let H = normalize(V + L);
  let spec = pow(max(dot(N, H), 0.0), shininess);
  return lightColor * (diff * Kd + spec * specColor);
}

fn shadePhongFamily(N: vec3f, V: vec3f, L: vec3f, lightColor: vec3f, Kd: vec3f, specColor: vec3f, shininess: f32) -> vec3f {
  switch settings.lightingModel {
    case 0u: { // Phong
      return phong(N, V, L, lightColor, Kd, specColor, shininess);
    }
    case 1u: { // Blinn-Phong
      return blinnPhong(N, V, L, lightColor, Kd, specColor, shininess);
    }
    default: {
      return phong(N, V, L, lightColor, Kd, specColor, shininess);
    }
  }
}

fn fresnelSchlick(cosTheta: f32, F0: vec3f) -> vec3f {
  let clamped = clamp(1.0 - cosTheta, 0.0, 1.0);
  let factor = pow(clamped, 5.0);
  return F0 + (vec3f(1.0, 1.0, 1.0) - F0) * factor;
}

fn distributionGGX(N: vec3f, H: vec3f, roughness: f32) -> f32 {
  let a = roughness * roughness;
  let a2 = a * a;
  let NdotH = max(dot(N, H), 0.0);
  let NdotH2 = NdotH * NdotH;
  let denom = (NdotH2 * (a2 - 1.0) + 1.0);
  return a2 / (PI * denom * denom);
}

fn geometrySchlickGGX(NdotV: f32, roughness: f32) -> f32 {
  let r = roughness + 1.0;
  let k = (r * r) / 8.0;
  return NdotV / (NdotV * (1.0 - k) + k);
}

fn geometrySmith(N: vec3f, V: vec3f, L: vec3f, roughness: f32) -> f32 {
  let NdotV = max(dot(N, V), 0.0);
  let NdotL = max(dot(N, L), 0.0);
  let ggx1 = geometrySchlickGGX(NdotV, roughness);
  let ggx2 = geometrySchlickGGX(NdotL, roughness);
  return ggx1 * ggx2;
}

fn pbrShading(N: vec3f, V: vec3f, L: vec3f, lightColor: vec3f, albedo: vec3f, roughness: f32, metalness: f32) -> vec3f {
  let H = normalize(V + L);
  let NdotL = max(dot(N, L), 0.0);
  let NdotV = max(dot(N, V), 0.0);
  if (NdotL <= 0.0 || NdotV <= 0.0) {
    return vec3f(0.0, 0.0, 0.0);
  }

  let baseReflectivity = vec3f(0.04, 0.04, 0.04);
  let F0 = mix(baseReflectivity, albedo, vec3f(metalness, metalness, metalness));

  let F = fresnelSchlick(max(dot(H, V), 0.0), F0);
  let D = distributionGGX(N, H, roughness);
  let G = geometrySmith(N, V, L, roughness);

  let numerator = F * D * G;
  let denominator = 4.0 * NdotV * NdotL + 0.0001;
  let specular = numerator / denominator;

  let kD = (vec3f(1.0, 1.0, 1.0) - F) * (1.0 - metalness);
  let diffuse = (kD * albedo) / PI;

  return (diffuse + specular) * lightColor * NdotL;
}

fn shadeLightingModel(
  N: vec3f, V: vec3f, L: vec3f,
  lightColor: vec3f,
  Kd: vec3f,
  specColor: vec3f,
  shininess: f32,
  roughness: f32,
  metalness: f32
) -> vec3f {
  switch settings.lightingModel {
    case 2u: {
      return pbrShading(N, V, L, lightColor, Kd, roughness, metalness);
    }
    default: {
      return shadePhongFamily(N, V, L, lightColor, Kd, specColor, shininess);
    }
  }
}

//sample shadow map ; returns 1 if lit, 0 if shadowed
fn calculateShadow(shadowPos: vec3f) -> f32 {
  let inBounds = shadowPos.x >= 0.0 && shadowPos.x <= 1.0 && shadowPos.y >= 0.0 && shadowPos.y <= 1.0 && shadowPos.z >= 0.0 && shadowPos.z <= 1.0;
  
  //3x3 PCF kernel 
  var shadow = 0.0;
  let texelSize = 1.0 / 2048.0;  //matches shadow ma size
  let bias = 0.002;  //small depth offset
  
  for (var x = -1; x <= 1; x++) {
    for (var y = -1; y <= 1; y++) {
      let offset = vec2f(f32(x), f32(y)) * texelSize;

      //comparison sampler
      shadow += textureSampleCompare(shadowMap, shadowSampler, shadowPos.xy + offset, shadowPos.z - bias);
    }
  }
  
  //bounds check after sampling
  return select(1.0, shadow / 9.0, inBounds);
}

@fragment
fn fs(i: VSOut) -> @location(0) vec4f {
  //normal+view vector
  var N = normalize(i.nrm);
  let V = normalize(camera.camPos - i.wp);

  let hasTexture = material.params.y > 0.5;
  let hasBump = material.params.z > 0.5;
  var Kd = i.base;
  if (hasTexture) {
    Kd = textureSample(tex, texSampler, i.uv).rgb;
  }

  if (hasBump) {
    let bumpSample = textureSample(bumpTex, bumpSampler, i.uv).rgb;
    let mapped = bumpSample * 2.0 - vec3f(1.0);
    let T = normalize(i.tangent);
    let B = normalize(i.bitangent);
    let TBN = mat3x3f(T, B, N);
    N = normalize(TBN * mapped);
  }

  let roughSample = textureSample(roughTex, texSampler, i.uv).r;
  let metalSample = textureSample(metalTex, texSampler, i.uv).r;
  let emissive = textureSample(emissiveTex, texSampler, i.uv).rgb;

  let roughness = clamp(roughSample, 0.04, 0.99);
  let metalness = clamp(metalSample, 0.0, 1.0);
  let shininess = max(1.0, material.params.x * pow(max(0.001, 1.0 - roughness), 4.0));
  let specBase = material.Ks.rgb;
  let specColor = mix(specBase, Kd, metalness);

  //ambient term
  var color = lights.ambient * material.Ka.rgb * Kd; // Use material.Ka.xyz
  
  //shadow factor from the map (directional light only)
  let shadow = calculateShadow(i.shadowPos);
  
  //directional lights (with shadow)
  for (var d: u32 = 0u; d < lights.numDir; d = d + 1u) {
    let L = normalize(-lights.dirLights[d].direction.xyz);
    let litColor = shadeLightingModel(N, V, L, lights.dirLights[d].color, Kd, specColor, shininess, roughness, metalness);
    color += litColor * mix(0.3, 1.0, shadow);  
  }

  //point lights 
  for (var p: u32 = 0u; p < lights.numPoint; p = p + 1u) {
    let lightPos = lights.pointLights[p].position.xyz;
    let toLight = lightPos - i.wp;
    let distance = length(toLight);
    let L = normalize(toLight);
    
    //attenuation with distance
    let constant = 1.0;
    let linear = 0.7;      
    let quadratic = 1.8;   
    let attenuation = 1.0 / (constant + linear * distance + quadratic * distance * distance);
    
    let litColor = shadeLightingModel(N, V, L, lights.pointLights[p].color, Kd, specColor, shininess, roughness, metalness);
    color += litColor * attenuation;
  }

  //spotlights
  for (var s: u32 = 0u; s < lights.numSpot; s = s + 1u) {
    let S = lights.spotLights[s];
    let L = normalize(S.position.xyz - i.wp);
    let theta = dot(L, normalize(-S.direction));
    if (theta > S.outerCutOff) {
      let intensity = smoothstep(S.outerCutOff, S.cutOff, theta);
      color += shadeLightingModel(N, V, L, S.color, Kd, specColor, shininess, roughness, metalness) * intensity;
    }
  }

  color += emissive;

  return vec4f(pow(max(color, vec3f(0.0)), vec3f(1.0/2.2)), 1.0);
}
`;

let currentLightingModel = 0;
const stats = {
  0: { frames: 0, time: 0 },
  1: { frames: 0, time: 0 },
  2: { frames: 0, time: 0 },
};
let frameCounter = 0;
const lightingNames = ['Phong', 'Blinn-Phong', 'PBR'];

function recordFrame(model, dt) {
  const entry = stats[model];
  entry.frames += 1;
  entry.time += dt;
}

function computeStatsMessage(model) {
  const name = lightingNames[model] ?? 'Unknown';
  const entry = stats[model];
  if (!entry || entry.frames === 0) {
    return 'Lighting ' + name + ': collecting...';
  }
  const avgMs = (entry.time / entry.frames) * 1000.0;
  const fps = avgMs > 0.0 ? 1000.0 / avgMs : 0.0;
  return 'Lighting ' + name + ': ' + avgMs.toFixed(2) + ' ms (' + fps.toFixed(1) + ' FPS)';
}

//javascript - set up GPU, create pipelines, buffers, gemoetry, and draw with 2 passes
async function main() {
const canvas = document.getElementById('canvas');
  if (!navigator.gpu) {
    document.body.innerHTML = '<h1>WebGPU Not Supported</h1>'; return;
  }
const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    document.body.innerHTML = '<h1>WebGPU Adapter Not Available</h1>'; return;
  }
  const device = await adapter.requestDevice();
const context = canvas.getContext('webgpu');
const format  = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode:'opaque' });

  //main depth buffer
  const depthTex = device.createTexture({
    size:[canvas.width, canvas.height],
    format:'depth24plus',
    usage: GPUTextureUsage.RENDER_ATTACHMENT
  });

  //shadow map resources (depth texture+sampler+uniform)
  const SHADOW_MAP_SIZE = 1024; //for better quality
  const shadowDepthTexture = device.createTexture({
    size: [SHADOW_MAP_SIZE, SHADOW_MAP_SIZE],
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    format: 'depth32float',
  });

  const shadowSampler = device.createSampler({
    compare: 'less', //compare model -> enable PCF
    magFilter: 'linear',
    minFilter: 'linear',
  });

  //shadow uniform buffer
  const shadowUBO = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  //shadow pass pipeline (depth only, 1st pass)
  const shadowModule = device.createShaderModule({ code: SHADOW_WGSL });
  const shadowPipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: shadowModule, entryPoint: 'vs', buffers: [{ arrayStride: 44, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] }] },
    fragment: { module: shadowModule, entryPoint: 'fs', targets: [] },
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' }
  });

  //main pipeline (phone and shadow sampling)
  const module = device.createShaderModule({ code: WGSL });
  module.getCompilationInfo().then(info => {
    if (info.messages.length > 0) {
      console.log('Shader compilation info:', info.messages);
    }
  });

  const vbufLayout = [{
    arrayStride: 44,
    attributes: [
      { shaderLocation:0, offset:0,  format:'float32x3' }, //position
      { shaderLocation:1, offset:12, format:'float32x3' }, //normal
      { shaderLocation:2, offset:24, format:'float32x3' }, //color (fallback Kd)
      { shaderLocation:3, offset:36, format:'float32x2' }  //uv
    ]
  }];

const pipeline = device.createRenderPipeline({
  layout:'auto',
    vertex:   { module, entryPoint:'vs', buffers:vbufLayout },
    fragment: { module, entryPoint:'fs', targets:[{ format }] },
    primitive:{ topology:'triangle-list', cullMode:'none' },
    depthStencil:{ format:'depth24plus', depthWriteEnabled:true, depthCompare:'less' }
  });

  const settingsUBO = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const settingsData = new Uint32Array(4);
  settingsData[0] = currentLightingModel >>> 0;
  device.queue.writeBuffer(settingsUBO, 0, settingsData);
  const settingsBG = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(3),
    entries: [
      { binding: 0, resource: { buffer: settingsUBO } },
    ],
  });

  const textureCache = new Map();

  async function loadTexture(device, url){
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const blob = await response.blob();
    const bitmap = await createImageBitmap(blob);
    const texture = device.createTexture({
      size: [bitmap.width, bitmap.height],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    device.queue.copyExternalImageToTexture({ source: bitmap }, { texture }, [bitmap.width, bitmap.height]);
    const sampler = device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      mipmapFilter: 'linear',
      addressModeU: 'repeat',
      addressModeV: 'repeat',
    });
    return { texture, sampler };
  }

  const defaultTexture = device.createTexture({
    size:[1,1,1],
    format:'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
  });
  device.queue.writeTexture(
    { texture: defaultTexture },
    new Uint8Array([255,255,255,255]),
    { bytesPerRow: 4 },
    [1,1,1]
  );
  const defaultTextureView = defaultTexture.createView();
  const defaultSampler = device.createSampler({
    magFilter:'linear',
    minFilter:'linear',
    addressModeU:'repeat',
    addressModeV:'repeat'
  });

  const defaultNormalTexture = device.createTexture({
    size:[1,1,1],
    format:'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
  });
  device.queue.writeTexture(
    { texture: defaultNormalTexture },
    new Uint8Array([128,128,255,255]),
    { bytesPerRow: 4 },
    [1,1,1]
  );
  const defaultNormalView = defaultNormalTexture.createView();

  function makeSolidTexture(r,g,b,a=255){
    const tex = device.createTexture({
      size:[1,1,1],
      format:'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
    });
    device.queue.writeTexture({ texture: tex }, new Uint8Array([r,g,b,a]), { bytesPerRow:4 }, [1,1,1]);
    return tex.createView();
  }
  const defaultRoughView = makeSolidTexture(255,255,255,255); // roughness 1.0
  const defaultMetalView = makeSolidTexture(0,0,0,255);       // metalness 0.0
  const defaultEmissiveView = makeSolidTexture(0,0,0,255);    // emissive off

  //camera UBO (viewProj + camPos)
  //size 80 - 4x4mat (64) + camPos (12) +padding
  const cameraUBO = device.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });


  //matrix helper
  const M = {
    //identity matrix
    I(){ return new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]); },
   //translate
    T(x,y,z){ const m=M.I(); m[12]=x; m[13]=y; m[14]=z; return m; },
    //scale
    S(x,y,z){ return new Float32Array([x,0,0,0, 0,y,0,0, 0,0,z,0, 0,0,0,1]); },
    //rotate Y
    RY(a){ const c=Math.cos(a), s=Math.sin(a); return new Float32Array([c,0,-s,0, 0,1,0,0, s,0,c,0, 0,0,0,1]); },
   //multiply matrices
    mul(a,b){ const o=new Float32Array(16); for(let c=0;c<4;c++)for(let r=0;r<4;r++)o[c*4+r]=a[0*4+r]*b[c*4+0]+a[1*4+r]*b[c*4+1]+a[2*4+r]*b[c*4+2]+a[3*4+r]*b[c*4+3]; return o; },
    //perspective
    perspective(fovy,asp,near,far){ const f=1/Math.tan(fovy/2),nf=1/(near-far); return new Float32Array([ f/asp,0,0,0, 0,f,0,0, 0,0,(far+near)*nf,-1, 0,0,(2*far*near)*nf,0 ]); },
    //orthographic matrix
    ortho(l,r,b,t,n,f){ return new Float32Array([2/(r-l),0,0,0, 0,2/(t-b),0,0, 0,0,2/(n-f),0, (l+r)/(l-r),(t+b)/(b-t),(n+f)/(n-f),1]); },
    //lookAt matrix
    lookAt(e,c,u){ let zx=e[0]-c[0],zy=e[1]-c[1],zz=e[2]-c[2]; let rl=1/Math.hypot(zx,zy,zz); zx*=rl; zy*=rl; zz*=rl;
      let xx=u[1]*zz-u[2]*zy,xy=u[2]*zx-u[0]*zz,xz=u[0]*zy-u[1]*zx; rl=1/Math.hypot(xx,xy,xz); xx*=rl; xy*=rl; xz*=rl;
      const yx=zy*xz-zz*xy,yy=zz*xx-zx*xz,yz=zx*xy-zy*xx; const m=new Float32Array([xx,yx,zx,0, xy,yy,zy,0, xz,yz,zz,0, 0,0,0,1]); const t=M.T(-e[0],-e[1],-e[2]); return M.mul(m,t); },
    //build 3x3 inverse for normals
      normalFromMat4(m){ const a=m,b=new Float32Array(9); const a00=a[0],a01=a[1],a02=a[2],a10=a[4],a11=a[5],a12=a[6],a20=a[8],a21=a[9],a22=a[10];
      const b01=a22*a11-a12*a21, b11=-a22*a10+a12*a20, b21=a21*a10-a11*a20; let det=a00*b01+a01*b11+a02*b21; if(!det) det=1e-6; det=1.0/det;
      b[0]=b01*det; b[1]=(-a22*a01+a02*a21)*det; b[2]=(a12*a01-a02*a11)*det; b[3]=b11*det; b[4]=(a22*a00-a02*a20)*det; b[5]=(-a12*a00+a02*a10)*det; b[6]=b21*det; b[7]=(-a21*a00+a01*a20)*det; b[8]=(a11*a00-a01*a10)*det; return b; }
  };

  function writeMatUBO(buf, mat, hasTexture, hasBump){
    const Ka = mat.Ka ?? [.05,.05,.06];
    const Ks = mat.Ks ?? [.8,.8,.85];
    const shiny = mat.shiny ?? 64;
    const data = new Float32Array([
      Ka[0], Ka[1], Ka[2], 1.0,
      Ks[0], Ks[1], Ks[2], 1.0,
      shiny, hasTexture ? 1.0 : 0.0, hasBump ? 1.0 : 0.0, 0.0
    ]);
    device.queue.writeBuffer(buf, 0, data);
  }

  //OBJ/MTL loader
  async function getText(url){ return await (await fetch(url)).text(); }
  async function loadMTL(url){
    const txt = await getText(url); 
    const mats={}; let cur=null;
    for(const raw of txt.split(/\r?\n/)){ const s=raw.trim(); if(!s||s[0]==='#') continue; const p=s.split(/\s+/);
      if(p[0]==='newmtl'){ cur=p[1]; mats[cur]={ Ka:[.05,.05,.05], Kd:[.7,.7,.7], Ks:[.04,.04,.04], shiny:32, map_Kd:null, map_Bump:null, map_Rough:null, map_Metal:null, map_Emissive:null }; }
      else if(!cur) continue;
      else if(p[0]==='Ka') mats[cur].Ka=[+p[1],+p[2],+p[3]];
      else if(p[0]==='Kd') mats[cur].Kd=[+p[1],+p[2],+p[3]];
      else if(p[0]==='Ks') mats[cur].Ks=[+p[1],+p[2],+p[3]];
      else if(p[0]==='Ns') mats[cur].shiny=+p[1];
      else if(p[0]==='map_Kd') mats[cur].map_Kd=p.slice(1).join(' ');
      else if(p[0]==='map_Bump' || p[0]==='map_bump' || p[0]==='bump') mats[cur].map_Bump=p[p.length-1];
      else if(p[0]==='map_Pr' || p[0]==='map_Roughness') mats[cur].map_Rough=p[p.length-1];
      else if(p[0]==='map_Pm' || p[0]==='map_Metalness') mats[cur].map_Metal=p[p.length-1];
      else if(p[0]==='map_Ke' || p[0]==='map_Emissive') mats[cur].map_Emissive=p[p.length-1];
    } return mats;
  }
  function computeNormals(pos, idx){
    //quick per-vertex avg of face normals
    const n=new Float32Array(pos.length);
    for(let i=0;i<idx.length;i+=3){
      const a=idx[i]*3,b=idx[i+1]*3,c=idx[i+2]*3;
      const ax=pos[a],ay=pos[a+1],az=pos[a+2], bx=pos[b],by=pos[b+1],bz=pos[b+2], cx=pos[c],cy=pos[c+1],cz=pos[c+2];
      const ux=bx-ax,uy=by-ay,uz=bz-az, vx=cx-ax,vy=cy-ay,vz=cz-az;
      const nx=uy*vz-uz*vy, ny=uz*vx-ux*vz, nz=ux*vy-uy*vx;
      n[a]+=nx; n[a+1]+=ny; n[a+2]+=nz; n[b]+=nx; n[b+1]+=ny; n[b+2]+=nz; n[c]+=nx; n[c+1]+=ny; n[c+2]+=nz;
    }
    for(let i=0;i<n.length;i+=3){ const x=n[i],y=n[i+1],z=n[i+2]; const l=1/Math.hypot(x,y,z); n[i]=x*l; n[i+1]=y*l; n[i+2]=z*l; }
    return n;
  }
  async function loadOBJ(url){
    const txt = await getText(url);
    const base = url.lastIndexOf('/') >= 0 ? url.slice(0, url.lastIndexOf('/') + 1) : '';
    const V=[], VN=[], VT=[], faces=[], use=[]; let curMat=null, mtlName=null;
    for(const raw of txt.split(/\r?\n/)){ 
      const s=raw.trim(); if(!s||s[0]==='#') continue; const p=s.split(/\s+/);
      if(p[0]==='v')  V.push([+p[1],+p[2],+p[3]]);
      else if(p[0]==='vn') VN.push([+p[1],+p[2],+p[3]]);
      else if(p[0]==='vt') VT.push([+p[1],+p[2]]);
      else if(p[0]==='f'){ const tri=p.slice(1).map(q=>q.split('/').map(x=>parseInt(x||'0'))); for(let i=1;i<tri.length-1;i++){ faces.push([tri[0],tri[i],tri[i+1]]); use.push(curMat); } }
      else if(p[0]==='usemtl'){ curMat=p[1]; }
      else if(p[0]==='mtllib'){ mtlName=p[1]; }
    }
    const mtl = mtlName ? await loadMTL(base + mtlName) : {};
    const groups={}; let k=0;
    for(let f=0; f<faces.length; f++){
      const tri=faces[f]; const mname=use[f]||'default';
      if(!groups[mname]) groups[mname]={ pos:[], nrm:[], uv:[], idx:[], kd: (mtl[mname]?.Kd)||(mtl[mname]?.Ka)||[0.7,0.7,0.7], mat:{ Ka:(mtl[mname]?.Ka)||[.05,.05,.05], Ks:(mtl[mname]?.Ks)||[.04,.04,.04], shiny:(mtl[mname]?.shiny)||64, map_Kd: mtl[mname]?.map_Kd || null, map_Bump: mtl[mname]?.map_Bump || null, map_Rough: mtl[mname]?.map_Rough || null, map_Metal: mtl[mname]?.map_Metal || null, map_Emissive: mtl[mname]?.map_Emissive || null } };
      for(const v of tri){
        const vi=v[0]-1, ti=(v[1]||0)-1, ni=(v[2]||0)-1; const P=V[vi];
        groups[mname].pos.push(P[0],P[1],P[2]);
        if(ni>=0 && VN[ni]){ const W=VN[ni]; groups[mname].nrm.push(W[0],W[1],W[2]); } else { groups[mname].nrm.push(0,0,0); }
        if(ti>=0 && VT[ti]){ const T=VT[ti]; groups[mname].uv.push(T[0],1.0 - T[1]); } else { groups[mname].uv.push(0,0); }
        groups[mname].idx.push(k++);
      }
    }
    for(const g of Object.values(groups)){
      let has=false; for(let i=0;i<g.nrm.length;i+=3){ if(g.nrm[i]||g.nrm[i+1]||g.nrm[i+2]){ has=true; break; } }
      if(!has) g.nrm = Array.from(computeNormals(g.pos, g.idx));
    }
    return groups;
  }
  
  //interleave pos/normal/color into one vertex/index buffer
  function makeGeo(group){
    const count=group.pos.length/3;
    const inter=new Float32Array(count*11);
    for(let i=0;i<count;i++){
      inter[i*11+0]=group.pos[i*3+0]; inter[i*11+1]=group.pos[i*3+1]; inter[i*11+2]=group.pos[i*3+2];
      inter[i*11+3]=group.nrm[i*3+0]; inter[i*11+4]=group.nrm[i*3+1]; inter[i*11+5]=group.nrm[i*3+2];
      inter[i*11+6]=group.kd[0]; inter[i*11+7]=group.kd[1]; inter[i*11+8]=group.kd[2];
      inter[i*11+9]=group.uv?.[i*2] ?? 0;
      inter[i*11+10]=group.uv?.[i*2+1] ?? 0;
    }
    const vbuf=device.createBuffer({ size:inter.byteLength, usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(vbuf,0,inter);
    const ibuf=device.createBuffer({ size:group.idx.length*4, usage:GPUBufferUsage.INDEX|GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(ibuf,0,new Uint32Array(group.idx));
    return { vbuf, ibuf, indexCount: group.idx.length };
  }

  //container for drawables (geometry and per object UBOs)
  const drawables=[];
  function addDrawable(group, model, overrideMat, texRes){
    const mat = overrideMat || group.mat;
   //per object material UBO
    const matUBO = device.createBuffer({ size: 48, usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST });
    const hasTexture = !!texRes?.albedo;
    const hasBump = !!texRes?.bump;
    writeMatUBO(matUBO, mat, hasTexture, hasBump);
    //object transform UBO (normal matrix+model)
    const objUBO = device.createBuffer({ size:112, usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST });
    const diffuseView = texRes?.albedo?.view ?? defaultTextureView;
    const diffuseSampler = texRes?.albedo?.sampler ?? defaultSampler;
    const bumpView = texRes?.bump?.view ?? defaultNormalView;
    const bumpSampler = texRes?.bump?.sampler ?? defaultSampler;
    const roughView = texRes?.roughness?.view ?? defaultRoughView;
    const metalView = texRes?.metalness?.view ?? defaultMetalView;
    const emissiveView = texRes?.emissive?.view ?? defaultEmissiveView;
    //main pass bind group (material+camera+object+texture)
    const bind = device.createBindGroup({ layout:pipeline.getBindGroupLayout(1), entries:[
      { binding:0, resource:{ buffer:matUBO } },
      { binding:1, resource:{ buffer:cameraUBO } },
      { binding:2, resource:{ buffer:objUBO } },
      { binding:3, resource: diffuseView },
      { binding:4, resource: diffuseSampler },
      { binding:5, resource: bumpView },
      { binding:6, resource: bumpSampler },
      { binding:7, resource: roughView },
      { binding:8, resource: metalView },
      { binding:9, resource: emissiveView }
    ]});
    //shadow UBO -LightviewProj
    const shadowBind2 = device.createBindGroup({ layout: pipeline.getBindGroupLayout(2), entries:[
      { binding: 0, resource: { buffer: shadowUBO } }
    ]});

    //shadow pass per object bind group
    const shadowBind1 = device.createBindGroup({ layout: shadowPipeline.getBindGroupLayout(1), entries:[
      { binding: 0, resource: { buffer: objUBO } }
    ]});
    drawables.push({
      geo: makeGeo(group), model, bind, objUBO, shadowBind1, shadowBind2,
      draw(pass) {
        pass.setBindGroup(1, this.bind);
        pass.setVertexBuffer(0, this.geo.vbuf);
        pass.setIndexBuffer(this.geo.ibuf, 'uint32');
        pass.drawIndexed(this.geo.indexCount);
      },
      drawShadow(pass) {
        pass.setBindGroup(1, this.shadowBind1);
        pass.setVertexBuffer(0, this.geo.vbuf);
        pass.setIndexBuffer(this.geo.ibuf, 'uint32');
        pass.drawIndexed(this.geo.indexCount);
      }
    });
  }

  //vector normalizer
  const normalizeVector = (v) => {
    const len = Math.hypot(...v);
    return v.map(c => c / len);
  };

  //storage buffer for light structs
  const lightBuffer = device.createBuffer({
    size: 10240,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });


  //bind group 0 (lights, shadow map, sampler)
  const bg0 = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: lightBuffer } },
      { binding: 1, resource: shadowDepthTexture.createView() },
      { binding: 2, resource: shadowSampler }
    ]
  });

  //shadow pass set#0
  const shadowBG0 = device.createBindGroup({
    layout: shadowPipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: shadowUBO } }]
  });


  //for screen UI
  // Intensity multiplier set to 1.0 (no slider)

  const lightingSelect = document.getElementById('lightingModel');
  if (lightingSelect instanceof HTMLSelectElement) {
    lightingSelect.value = String(currentLightingModel);
  }
  const hudElement = document.getElementById('hud');
  if (hudElement) {
    hudElement.textContent = computeStatsMessage(currentLightingModel);
  }
  if (lightingSelect instanceof HTMLSelectElement) {
    lightingSelect.addEventListener('change', (event) => {
      const target = event.target;
      if (target instanceof HTMLSelectElement) {
        const parsed = Number.parseInt(target.value, 10);
        currentLightingModel = Number.isNaN(parsed) ? 0 : parsed;
        settingsData[0] = currentLightingModel >>> 0;
        if (hudElement) {
          hudElement.textContent = computeStatsMessage(currentLightingModel);
        }
      }
    });
  }

  const lightToggles = {
    dir: true,
    point: true,
    spot: true,
  };
  document.getElementById('dirToggle').addEventListener('change', (e) => lightToggles.dir = e.target.checked);
  document.getElementById('pointToggle').addEventListener('change', (e) => lightToggles.point = e.target.checked);
  document.getElementById('spotToggle').addEventListener('change', (e) => lightToggles.spot = e.target.checked);


  //Building light system (1 direction light, 10 point lights, 3 spotlights)
  const lightSystem = {
    ambient: [0.15, 0.15, 0.2],  // Subtle ambient for nighttime scene
    dirLights: [],
    pointLights: [],
    spotLights: [],
    numDir: 1, numPoint: 10, numSpot: 3,  // 1 directional, 10 point, 3 spot
  };
  
  //1 directional light
  const initialDirLights = [
    { direction: [0, -1, 0, 0], color: [0.675, 0.675, 0.63] },  // Main overhead light (sun-like) - ONLY 1 directional light
  ];
  
  // 
  const normalizeVec = (v) => {
    const len = Math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    return len > 0 ? v.map(c => c / len) : v;
  };

  //initializing 10 pt lights
  const initialPointLights = [];
  for (let i = 0; i < 10; i++) {
    const angle = (i / 10) * Math.PI * 2;
    const radius = 10 + (i % 3) * 8;  //changing radius
    const x = Math.cos(angle) * radius;
    const z = Math.sin(angle) * radius;
    const y = 0.2;  //keep close to ground
    initialPointLights.push({
      position: [x, y, z, 1],
      baseColor: [2.0, 2.0, 2.5], 
      color: [2.0, 2.0, 2.5]
    });
  }
  
  const moonPos = [-25, 30, -25];
  const moonDir = normalizeVec([-moonPos[0], -moonPos[1], -moonPos[2]]);
  const initialSpotLights = [
    { position: [...moonPos, 1], direction: moonDir, color: [0.675, 0.675, 0.79], cutOff: Math.cos(Math.PI / 8), outerCutOff: Math.cos(Math.PI / 6) },  // Moonlight from above
    { position: [Math.cos(Math.PI/4) * 8, 5, Math.sin(Math.PI/4) * 8, 1], direction: [-Math.cos(Math.PI/4), -1, -Math.sin(Math.PI/4)], color: [0.56, 0.45, 0.34], cutOff: Math.cos(Math.PI / 12), outerCutOff: Math.cos(Math.PI / 9) },  // Warm spotlight
    { position: [Math.cos(5*Math.PI/4) * 8, 5, Math.sin(5*Math.PI/4) * 8, 1], direction: [-Math.cos(5*Math.PI/4), -1, -Math.sin(5*Math.PI/4)], color: [0.34, 0.45, 0.56], cutOff: Math.cos(Math.PI / 12), outerCutOff: Math.cos(Math.PI / 9) },  // Cool spotlight
  ];

  //pad arrays to 10 as expected by WGSL
  lightSystem.dirLights = [...initialDirLights];
  while(lightSystem.dirLights.length < 10) lightSystem.dirLights.push({ direction: [0,0,0,0], color: [0,0,0]});
  lightSystem.pointLights = [...initialPointLights];
  while(lightSystem.pointLights.length < 10) lightSystem.pointLights.push({ position: [0,0,0,0], baseColor: [0,0,0], color: [0,0,0]});
  lightSystem.spotLights = [...initialSpotLights];
  while(lightSystem.spotLights.length < 10) lightSystem.spotLights.push({ position: [0,0,0,0], direction: [0,0,0], color: [0,0,0], cutOff: 0, outerCutOff: 0});
  
  console.log('Light counts:', lightSystem.numDir, lightSystem.numPoint, lightSystem.numSpot); //LOGGING # of lights to console
  console.log('Point lights initialized:', lightSystem.pointLights.length);
  
  const ufoPos = [0, 12, 0];  //UFO position
  let ufoRotation = 0;
  
  function animateLights(time) {
    // Rotate UFO slowly (0.1 radians per second)
    ufoRotation = time * 0.2;
    
    //rotate saucer
    const ufoTransform = M.mul(M.mul(M.T(...ufoPos), M.RY(ufoRotation)), M.S(2.5, 2.5, 2.5));
    for (let i = ufoDrawablesStartIndex; i < ufoDrawablesEndIndex; i++) {
      drawables[i].model = ufoTransform;
    }
    
    //move point lights along rings + pulse intensity
    for (let i = 0; i < 10; i++) {
      const baseAngle = (i / 10) * Math.PI * 2;
      const radius = 10 + (i % 3) * 8; 
      const rotationSpeed = 0.3 + (i % 3) * 0.1;  
      const angle = baseAngle + time * rotationSpeed;
      
      const x = Math.cos(angle) * radius;
      const z = Math.sin(angle) * radius;
      const y = 0.2; //keep them close to ground
      
      lightSystem.pointLights[i].position = [x, y, z, 1];
      
      
      const intensity = 1.0 + 0.2 * Math.sin(time * 3 + i * 0.5); //subtle ulse
      lightSystem.pointLights[i].color = lightSystem.pointLights[i].baseColor.map(c => c * intensity);
    }

    //orbit 2 spotlights below UFO
    const spotAngle1 = time * 0.3;
    lightSystem.spotLights[1].position = [Math.cos(spotAngle1) * 8, 5, Math.sin(spotAngle1) * 8, 1];
    lightSystem.spotLights[1].direction = [-Math.cos(spotAngle1), -1, -Math.sin(spotAngle1)];

    const spotAngle2 = time * 0.2 + Math.PI;
    lightSystem.spotLights[2].position = [Math.cos(spotAngle2) * 6, 4, Math.sin(spotAngle2) * 6, 1];
    lightSystem.spotLights[2].direction = [-Math.cos(spotAngle2), -1, -Math.sin(spotAngle2)];
  }

  //uploading light system into one buffer (from the WGSL layout)
function uploadLights() {
  const buffer = new ArrayBuffer(10240);
  const floatView = new Float32Array(buffer);
  const uintView = new Uint32Array(buffer);

  //ambient (vec3) + pad
  floatView.set(lightSystem.ambient, 0);
  
  //counts (u32) -> index 4..7
  uintView[4] = lightToggles.point ? lightSystem.numPoint : 0;
  uintView[5] = lightToggles.dir ? lightSystem.numDir : 0;
  uintView[6] = lightToggles.spot ? lightSystem.numSpot : 0;

  const black = [0, 0, 0];

  //point lights array
  let byteOffset = 32; 
  for (let i = 0; i < 10; i++) {
    const p = lightSystem.pointLights[i];
    const baseColor = p.color;  // No intensity multiplier
    const color = lightToggles.point ? baseColor : black;
    floatView.set(p.position, byteOffset / 4); byteOffset += 16;
    floatView.set(color, byteOffset / 4); byteOffset += 16;
  }

  //directional lights blocks 
  byteOffset = 352;
  for (let i = 0; i < 10; i++) {
    const d = lightSystem.dirLights[i];
    const baseColor = d.color;  // No intensity multiplier
    const isActive = i < lightSystem.numDir;
    const color = (lightToggles.dir && isActive) ? baseColor : black;
    floatView.set(d.direction, byteOffset / 4); byteOffset += 16;
    floatView.set(color, byteOffset / 4); byteOffset += 16;
  }

  //spotlight block
  byteOffset = 672;
  for (let i = 0; i < 10; i++) {
    const s = lightSystem.spotLights[i];
    const baseColor = s.color;  // No intensity multiplier
    const isActive = i < lightSystem.numSpot;
    const color = (lightToggles.spot && isActive) ? baseColor : black;
    const floatOffset = byteOffset / 4;
    
    floatView.set(s.position, floatOffset + 0);  // position: vec4f uses 4 floats
    floatView.set(s.direction, floatOffset + 4); // direction: vec3f uses 3 floats
    // floatOffset + 7 is empty (padding)
    floatView.set(color, floatOffset + 8);       // color: vec3f uses 3 floats
    // floatOffset + 11 is empty (padding)
    floatView[floatOffset + 12] = s.cutOff;      // cutOff: f32 uses 1 float
    floatView[floatOffset + 13] = s.outerCutOff; // outerCutOff: f32 uses 1 float
    // floatOffset + 14 and 15 are empty (padding)

    byteOffset += 64; // Stride of SpotLight struct is 64 bytes
  }
  
  device.queue.writeBuffer(lightBuffer, 0, buffer);
}

  //camera controls
  let angle= -Math.PI / 4, radius=35, height=5; const up=[0,1,0];
  window.addEventListener('keydown',e=>{
    if(e.key==='a') angle-=0.08; if(e.key==='d') angle+=0.08;
    if(e.key==='w') radius=Math.max(2, radius-0.5); if(e.key==='s') radius+=0.5;
    if(e.key==='q') height-=0.25; if(e.key==='e') height+=0.25;
  });

  //write camera matrices each frame
  function updateCamera(dt){
    angle += dt*0.15;
    const asp=canvas.width/Math.max(1,canvas.height);
    const P=M.perspective(Math.PI/4,asp,0.01,200);
    const eye=[Math.sin(angle)*radius, height, Math.cos(angle)*radius];
    const V=M.lookAt(eye,[0,7,0],up); // Re-centered lookAt
    const VP=M.mul(P,V);
    device.queue.writeBuffer(cameraUBO,0,VP);
    device.queue.writeBuffer(cameraUBO,64,new Float32Array([eye[0],eye[1],eye[2],0]));
  }

  function drawAll(pass){
    for(const d of drawables) d.draw(pass);
  }

  //helper for gemoetry
  async function addModel(url, model, overrideMat, options = {}){
    const { loadTextures = true } = options;
    console.log('Fetching OBJ', url);
    const groups = await loadOBJ(url);
    const entries = Object.values(groups);
    if (entries.length === 0) {
      console.warn(`No drawable groups found in ${url}. Skipping.`);
      return;
    }
    const base=url.slice(0,url.lastIndexOf('/')+1);
    for(const g of entries) {
      let textures = {};
      if (loadTextures && g.mat.map_Kd) {
        const texUrl = base + g.mat.map_Kd;
        console.log('Attempting to load texture:', texUrl);
        if (textureCache.has(texUrl)) {
          textures.albedo = textureCache.get(texUrl);
        } else {
          try {
            const res = await loadTexture(device, texUrl);
            res.view = res.texture.createView();
            textureCache.set(texUrl, res);
            textures.albedo = res;
            console.log('Loaded texture:', texUrl);
          } catch (err) {
            console.warn('Failed to load texture', texUrl, err);
          }
        }
      }
      if (loadTextures && g.mat.map_Bump) {
        const bumpUrl = base + g.mat.map_Bump;
        console.log('Attempting to load bump map:', bumpUrl);
        if (textureCache.has(bumpUrl)) {
          textures.bump = textureCache.get(bumpUrl);
        } else {
          try {
            const res = await loadTexture(device, bumpUrl);
            res.view = res.texture.createView();
            textureCache.set(bumpUrl, res);
            textures.bump = res;
            console.log('Loaded bump map:', bumpUrl);
          } catch (err) {
            console.warn('Failed to load bump map', bumpUrl, err);
          }
        }
      }
      if (loadTextures && g.mat.map_Rough) {
        const roughUrl = base + g.mat.map_Rough;
        console.log('Attempting to load roughness map:', roughUrl);
        if (textureCache.has(roughUrl)) {
          textures.roughness = textureCache.get(roughUrl);
        } else {
          try {
            const res = await loadTexture(device, roughUrl);
            res.view = res.texture.createView();
            textureCache.set(roughUrl, res);
            textures.roughness = res;
            console.log('Loaded roughness map:', roughUrl);
          } catch (err) {
            console.warn('Failed to load roughness map', roughUrl, err);
          }
        }
      }
      if (loadTextures && g.mat.map_Metal) {
        const metalUrl = base + g.mat.map_Metal;
        console.log('Attempting to load metalness map:', metalUrl);
        if (textureCache.has(metalUrl)) {
          textures.metalness = textureCache.get(metalUrl);
        } else {
          try {
            const res = await loadTexture(device, metalUrl);
            res.view = res.texture.createView();
            textureCache.set(metalUrl, res);
            textures.metalness = res;
            console.log('Loaded metalness map:', metalUrl);
          } catch (err) {
            console.warn('Failed to load metalness map', metalUrl, err);
          }
        }
      }
      if (loadTextures && g.mat.map_Emissive) {
        const emiUrl = base + g.mat.map_Emissive;
        console.log('Attempting to load emissive map:', emiUrl);
        if (textureCache.has(emiUrl)) {
          textures.emissive = textureCache.get(emiUrl);
        } else {
          try {
            const res = await loadTexture(device, emiUrl);
            res.view = res.texture.createView();
            textureCache.set(emiUrl, res);
            textures.emissive = res;
            console.log('Loaded emissive map:', emiUrl);
          } catch (err) {
            console.warn('Failed to load emissive map', emiUrl, err);
          }
        }
      }
      if (Object.keys(textures).length === 0) textures = null;
      addDrawable(g, model, overrideMat, textures);
    }
  }

  //
  function createIcosphereGeometry(subdivisions = 1) {
    //quick sphere (for moon)
    const t = (1.0 + Math.sqrt(5.0)) / 2.0;
    let vertices = [
      [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
      [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
      [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ];
    let faces = [
      [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
      [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
      [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
      [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ];

    vertices = vertices.map(normalizeVector);
    
    let midCache = new Map();
    const getMidPoint = (p1, p2) => {
      const key = Math.min(p1, p2) + "-" + Math.max(p1, p2);
      if (midCache.has(key)) return midCache.get(key);
      const v1 = vertices[p1];
      const v2 = vertices[p2];
      const mid = normalizeVector(v1.map((c, i) => (c + v2[i]) / 2));
      vertices.push(mid);
      const index = vertices.length - 1;
      midCache.set(key, index);
      return index;
    };
    
    for (let i = 0; i < subdivisions; i++) {
      const faces2 = [];
      faces.forEach(face => {
        const a = getMidPoint(face[0], face[1]);
        const b = getMidPoint(face[1], face[2]);
        const c = getMidPoint(face[2], face[0]);
        faces2.push([face[0], a, c]);
        faces2.push([face[1], b, a]);
        faces2.push([face[2], c, b]);
        faces2.push([a, b, c]);
      });
      faces = faces2;
    }

    const pos = vertices.flat();
    const idx = faces.flat();
    const nrm = pos; //unit shere -> normals == positions
    return { pos, nrm, idx };
  }

  function addMoon() {
    const moonGeo = createIcosphereGeometry(3);
    const kd = [0.8, 0.8, 0.9];
    const group = {
      pos: moonGeo.pos, nrm: moonGeo.nrm, idx: moonGeo.idx, kd: kd,
      mat: { Ka: [0.8, 0.8, 0.9], Ks: [0.1, 0.1, 0.1], shiny: 1 }
    };
    addDrawable(group, M.mul(M.T(...moonPos), M.S(2,2,2)), undefined, null);
  }

  //generating tree gemoetry
  function createTreeGeometry() {
    const trunkSides = 6;
    const trunkHeight = 2;
    const trunkRadius = 0.2;
    const trunkPos = [], trunkNrm = [], trunkIdx = [], trunkKd = [0.3, 0.2, 0.15];  // Brown trunk
    for (let i = 0; i < trunkSides; i++) {
      const angle = (i / trunkSides) * Math.PI * 2;
      const x = Math.cos(angle) * trunkRadius;
      const z = Math.sin(angle) * trunkRadius;
      trunkPos.push(x, 0, z,  x, trunkHeight, z);
      trunkNrm.push(x, 0, z,  x, 0, z);
    }
    for (let i = 0; i < trunkSides; i++) {
      const i0 = i * 2; const i1 = i0 + 1;
      const i2 = ((i + 1) % trunkSides) * 2; const i3 = i2 + 1;
      trunkIdx.push(i0, i2, i1,  i1, i2, i3);
    }

    //adding leaves
    const leafSides = 8;
    const leafHeight = 3;
    const leafRadius = 1;
    const leafPos = [0, trunkHeight + leafHeight, 0], leafNrm = [], leafIdx = [], leafKd = [0.15, 0.4, 0.2];  // Dark green leaves
    leafNrm.push(0,1,0);
    const tipIndex = 0; let baseIndex = 1;
    for (let i = 0; i < leafSides; i++) {
        const angle = (i / leafSides) * Math.PI * 2;
        const x = Math.cos(angle) * leafRadius;
        const z = Math.sin(angle) * leafRadius;
        leafPos.push(x, trunkHeight, z);
        leafNrm.push(0, 1, 0);
    }
    for (let i = 0; i < leafSides; i++) {
        leafIdx.push(tipIndex, baseIndex + i, baseIndex + ((i + 1) % leafSides));
    }

    return [
      { pos: trunkPos, nrm: trunkNrm, idx: trunkIdx, kd: trunkKd, mat: { Ka: trunkKd, Ks: [0.1, 0.1, 0.1], shiny: 4 } },
      { pos: leafPos, nrm: leafNrm, idx: leafIdx, kd: leafKd, mat: { Ka: leafKd, Ks: [0.2, 0.2, 0.2], shiny: 16 } }
    ];
  }


  //adding trees
  function addTrees() {
    const treeGeometries = createTreeGeometry();
    for (let i = 0; i < 50; i++) {
      const x = (Math.random() - 0.5) * 70;
      const z = (Math.random() - 0.5) * 70;
      const modelMat = M.T(x, -5, z); // Lower trees to new ground level
      for (const geo of treeGeometries) {
        addDrawable(geo, modelMat, undefined, null);
      }
    }
  }
  //generating and adding ground 
  function addGround(){
    const y = -5;
    const pos=[ -40,y,-40,  40,y,-40,  40,y,40,  -40,y,40 ];
    const nrm=[ 0,1,0, 0,1,0, 0,1,0, 0,1,0 ];
    const idx=[ 0,1,2,  0,2,3 ];
    const kd=[0.2, 0.4, 0.2];  // Moderate grass green
    const group={ pos, nrm, idx, kd, mat:{ Ka:kd, Ks:[0.1,0.1,0.1], shiny:8 } };
    addDrawable(group, M.I(), group.mat, null);
  }

  //loading models
  const ufoDrawablesStartIndex = drawables.length;
  await addModel('UFO_Empty.obj', M.mul(M.T(...ufoPos), M.S(2.5, 2.5, 2.5)));
  const ufoDrawablesEndIndex = drawables.length;
  
  addMoon();
  addGround();
  console.log('Ground added successfully.');
  addTrees();
  console.log('Trees added successfully.');

  
  //render loop (2 passes)
  let tPrev = performance.now();
function frame(){
    const t=performance.now(); const dt=(t-tPrev)*0.001; tPrev=t;
    updateCamera(dt);
    animateLights(t * 0.001);
    uploadLights();
    device.queue.writeBuffer(settingsUBO, 0, settingsData);
    recordFrame(currentLightingModel, dt);
    frameCounter++;
    const hudMessage = computeStatsMessage(currentLightingModel);
    if (hudElement) {
      hudElement.textContent = hudMessage;
    }
    if (frameCounter % 120 === 0) {
      console.log(hudMessage);
    }

    //update per-object UBOs
    for(const d of drawables){
        const n3 = M.normalFromMat4(d.model);
        const n4 = new Float32Array([ n3[0],n3[1],n3[2],0,  n3[3],n3[4],n3[5],0,  n3[6],n3[7],n3[8],0 ]);
        device.queue.writeBuffer(d.objUBO, 0, d.model);
        device.queue.writeBuffer(d.objUBO, 64, n4);
    }

    //building lightviewProj
    const lightDir = lightSystem.dirLights[0].direction;
    const lightPos = [-lightDir[0] * 60, -lightDir[1] * 60, -lightDir[2] * 60];  // At [0, 60, 0]
    const lightView = M.lookAt(lightPos, [0, 0, 0], [0, 0, -1]);  // Look at origin, up is -Z
    const lightProj = M.ortho(-50, 50, -50, 50, 10, 70);  // near/far: from light to ground
    const lightViewProj = M.mul(lightProj, lightView);
    device.queue.writeBuffer(shadowUBO, 0, lightViewProj);

    const enc=device.createCommandEncoder();
    
    // PASS 1: creating shadow map
    const shadowPass = enc.beginRenderPass({
      colorAttachments: [],
      depthStencilAttachment: {
        view: shadowDepthTexture.createView(),
        depthLoadOp: 'clear',
        depthClearValue: 1.0,
        depthStoreOp: 'store'
      }
    });
    shadowPass.setPipeline(shadowPipeline);
    shadowPass.setBindGroup(0, shadowBG0);
    for (const d of drawables) {
      d.drawShadow(shadowPass);
    }
    shadowPass.end();

    // PASS 2: rendering camera view, shadow map
    const pass=enc.beginRenderPass({
      colorAttachments:[{ view: context.getCurrentTexture().createView(), loadOp:'clear', storeOp:'store', clearValue:{r:0.05,g:0.07,b:0.12,a:1} }],
      depthStencilAttachment:{ view: depthTex.createView(), depthLoadOp:'clear', depthClearValue:1, depthStoreOp:'store' }
  });
  pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg0); //lights + shadowmap+samler
    pass.setBindGroup(3, settingsBG);
    for (const d of drawables) {
      pass.setBindGroup(2, d.shadowBind2); //lightviewproj
      d.draw(pass);
    }
  pass.end();
    
    device.queue.submit([enc.finish()]);
  requestAnimationFrame(frame);
}
frame();
}

main().catch(e => console.error(e));