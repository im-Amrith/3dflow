import React, { useState, useRef, useEffect, useMemo, useCallback } from "react";
import { createRoot } from "react-dom/client";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";
import { Camera, Hand, Settings2, Palette, Activity, Maximize2, Minimize2, Video, VideoOff, Aperture, Heart, Zap, Disc, Flower, Wind, ArrowDown, Music, Volume2, VolumeX } from "lucide-react";

// --- Types ---

type ShapeType = 'sphere' | 'heart' | 'flower' | 'saturn' | 'fireworks';

interface HandGestureState {
  tension: number; // 0 (relaxed/closed) to 1 (open/spread)
  distance: number; // 0 (close) to 1 (far)
  isPresent: boolean;
  handCount: number;
  position: { x: number, y: number, z: number };
}

// --- Utils ---

// Math for shapes
const generateTargetPositions = (count: number, shape: ShapeType, radius: number = 2): Float32Array => {
  const positions = new Float32Array(count * 3);
  
  for (let i = 0; i < count; i++) {
    const i3 = i * 3;
    let x = 0, y = 0, z = 0;

    if (shape === 'sphere') {
      const u = Math.random();
      const v = Math.random();
      const theta = 2 * Math.PI * u;
      const phi = Math.acos(2 * v - 1);
      // Use a "thick shell" distribution instead of full volume to look less "clumpy"
      // Radius between 0.5 and 1.0
      const r = (0.5 + Math.random() * 0.5) * radius; 
      x = r * Math.sin(phi) * Math.cos(theta);
      y = r * Math.sin(phi) * Math.sin(theta);
      z = r * Math.cos(phi);
    } 
    else if (shape === 'heart') {
      // Parametric heart
      const t = Math.random() * Math.PI * 2;
      const u = Math.random(); // volume factor
      // Base heart curve
      let hx = 16 * Math.pow(Math.sin(t), 3);
      let hy = 13 * Math.cos(t) - 5 * Math.cos(2*t) - 2 * Math.cos(3*t) - Math.cos(4*t);
      // Scale down
      hx /= 10; 
      hy /= 10;
      // Add volume
      const r = (1 - Math.abs(u - 0.5) * 2) * 0.5; // Thicker in middle
      const zOffset = (Math.random() - 0.5) * r * radius;
      
      x = hx * radius * 0.8;
      y = hy * radius * 0.8;
      z = zOffset;
    }
    else if (shape === 'flower') {
      // Rose curve: r = cos(k * theta)
      const k = 4; // petals
      const theta = Math.random() * Math.PI * 2;
      const phi = (Math.random() - 0.5) * Math.PI; // 3D thickness
      const rBase = Math.cos(k * theta) + 2; // +2 to keep it open
      const r = rBase * radius * 0.4 * Math.random();
      
      x = r * Math.cos(theta);
      y = r * Math.sin(theta);
      z = (Math.random() - 0.5) * 0.5 * radius; // Flattened 3D
    }
    else if (shape === 'saturn') {
      const isRing = Math.random() > 0.6; // 40% ring, 60% planet
      if (isRing) {
        const theta = Math.random() * Math.PI * 2;
        const dist = radius * (1.5 + Math.random() * 0.8);
        x = dist * Math.cos(theta);
        z = dist * Math.sin(theta);
        y = (Math.random() - 0.5) * 0.1; // Thin ring
      } else {
        const u = Math.random();
        const v = Math.random();
        const theta = 2 * Math.PI * u;
        const phi = Math.acos(2 * v - 1);
        const r = Math.cbrt(Math.random()) * radius * 0.8;
        x = r * Math.sin(phi) * Math.cos(theta);
        y = r * Math.sin(phi) * Math.sin(theta);
        z = r * Math.cos(phi);
      }
    }
    else if (shape === 'fireworks') {
       // Explosion from center
       const theta = Math.random() * Math.PI * 2;
       const phi = Math.acos(2 * Math.random() - 1);
       const r = radius * (0.2 + Math.random() * 2.5); // Spread out
       x = r * Math.sin(phi) * Math.cos(theta);
       y = r * Math.sin(phi) * Math.sin(theta);
       z = r * Math.cos(phi);
    }

    positions[i3] = x;
    positions[i3 + 1] = y;
    positions[i3 + 2] = z;
  }
  return positions;
};

// --- Shaders ---

const vertexShader = `
  uniform float uTime;
  uniform float uSize;
  uniform float uScale;
  attribute float aScale;
  attribute vec3 aRandom;
  
  varying vec3 vColor;
  
  void main() {
    vec4 modelPosition = modelMatrix * vec4(position, 1.0);
    
    // Add subtle organic movement (micro-jitter in shader)
    float time = uTime * 0.5;
    modelPosition.x += sin(time * aRandom.x + modelPosition.y) * 0.02;
    modelPosition.y += cos(time * aRandom.y + modelPosition.x) * 0.02;
    modelPosition.z += sin(time * aRandom.z + modelPosition.z) * 0.02;

    vec4 viewPosition = viewMatrix * modelPosition;
    vec4 projectedPosition = projectionMatrix * viewPosition;

    gl_Position = projectedPosition;
    
    // Size attenuation
    // Safe calculation for point size
    float depth = -viewPosition.z;
    if (depth < 0.1) depth = 0.1;
    gl_PointSize = uSize * uScale * aScale * (10.0 / depth);
    
    if (gl_PointSize < 0.0) gl_PointSize = 0.0;
  }
`;

const fragmentShader = `
  uniform vec3 uColor;
  uniform float uOpacity;
  
  void main() {
    // Circular particle
    vec2 coord = gl_PointCoord - vec2(0.5);
    float distanceToCenter = length(coord);
    if (distanceToCenter > 0.5) discard;

    // Soft glow edge
    float strength = 0.05 / distanceToCenter - 0.1; 
    
    gl_FragColor = vec4(uColor, strength * uOpacity);
  }
`;

// --- Audio Component ---

const AmbientSound = ({ 
  enabled, 
  gestureState 
}: { 
  enabled: boolean; 
  gestureState: React.MutableRefObject<HandGestureState>;
}) => {
  const audioCtx = useRef<AudioContext | null>(null);
  const masterGain = useRef<GainNode | null>(null);
  const nextNoteTime = useRef<number>(0);
  
  // Pentatonic Scale (F Major) - Shifted up 2 octaves for Chimes
  // F5, G5, A5, C6, D6, F6, G6, A6, C7, D7
  const scale = [698.46, 783.99, 880.00, 1046.50, 1174.66, 1396.91, 1567.98, 1760.00, 2093.00, 2349.32];

  useEffect(() => {
    if (enabled) {
      const AudioContext = window.AudioContext || (window as any).webkitAudioContext;
      const ctx = new AudioContext();
      const gain = ctx.createGain();
      gain.connect(ctx.destination);
      gain.gain.value = 0.4; 
      
      audioCtx.current = ctx;
      masterGain.current = gain;
    } else {
      if (audioCtx.current) {
        audioCtx.current.close();
        audioCtx.current = null;
        masterGain.current = null;
      }
    }
    
    return () => {
      if (audioCtx.current) {
        audioCtx.current.close();
      }
    };
  }, [enabled]);

  const playChime = (freq: number, pan: number, vol: number) => {
      if (!audioCtx.current || !masterGain.current) return;
      const ctx = audioCtx.current;
      
      const osc = ctx.createOscillator();
      const noteGain = ctx.createGain();
      const panner = ctx.createStereoPanner();
      
      // Sine wave for pure bell/chime sound
      osc.type = 'sine'; 
      osc.frequency.setValueAtTime(freq, ctx.currentTime);
      
      // Bell Envelope
      const now = ctx.currentTime;
      noteGain.gain.setValueAtTime(0, now);
      noteGain.gain.linearRampToValueAtTime(vol, now + 0.02); // Fast attack
      noteGain.gain.exponentialRampToValueAtTime(0.001, now + 3.0); // Long decay
      
      panner.pan.value = pan;
      
      osc.connect(panner);
      panner.connect(noteGain);
      noteGain.connect(masterGain.current);
      
      osc.start();
      osc.stop(now + 3.5);
  };

  useFrame((state) => {
    if (!audioCtx.current || !masterGain.current) return;
    
    const { isPresent, tension, position } = gestureState.current;
    const ctx = audioCtx.current;
    const time = ctx.currentTime;

    if (isPresent && time > nextNoteTime.current) {
        // Map Y position to pitch index
        const yNorm = (position.y + 5) / 12; 
        const index = Math.floor(Math.max(0, Math.min(0.99, yNorm)) * scale.length);
        
        // Add some randomness to pitch selection for "wind chime" feel
        const randomOffset = Math.floor((Math.random() - 0.5) * 2);
        const finalIndex = Math.max(0, Math.min(scale.length - 1, index + randomOffset));
        const freq = scale[finalIndex];
        
        // Map X to pan
        const pan = Math.max(-1, Math.min(1, position.x / 10));
        
        // Trigger rate depends on tension (Open hand = windy/fast, Closed = calm)
        // Add randomness to timing
        const baseDelay = THREE.MathUtils.mapLinear(tension, 0, 1, 0.3, 0.05);
        const randomDelay = Math.random() * 0.1;
        
        // Randomize volume slightly
        const vol = 0.1 + Math.random() * 0.1;
        
        playChime(freq, pan, vol);
        
        nextNoteTime.current = time + baseDelay + randomDelay;
    }
  });

  return null;
};

// --- Components ---

const Particles = ({ 
  count, 
  shape, 
  color, 
  size, 
  noiseStrength, 
  gravity,
  wind,
  gestureState 
}: { 
  count: number; 
  shape: ShapeType; 
  color: string; 
  size: number;
  noiseStrength: number;
  gravity: number;
  wind: number;
  gestureState: React.MutableRefObject<HandGestureState>;
}) => {
  const mesh = useRef<THREE.Points>(null);
  
  // -- Initialization --
  // Initialize ALL buffers in useMemo to ensure they are synchronized with 'count'
  const { positions, randoms, scales, velocities } = useMemo(() => {
    const pos = generateTargetPositions(count, 'sphere'); // Start as sphere
    const rnd = new Float32Array(count * 3);
    const scl = new Float32Array(count);
    const vel = new Float32Array(count * 3); // Velocity buffer
    
    for (let i = 0; i < count; i++) {
      rnd[i * 3] = Math.random();
      rnd[i * 3 + 1] = Math.random();
      rnd[i * 3 + 2] = Math.random();
      scl[i] = 0.5 + Math.random(); // Scale 0.5 to 1.5
    }
    
    return { positions: pos, randoms: rnd, scales: scl, velocities: vel };
  }, [count]);

  // -- Physics State --
  // We use Refs for physics targets to decouple from render, but velocities are now buffer-managed
  // We keep a ref to the velocity array to mutate it without React overhead, 
  // but we initialize it from the memoized value to keep length in sync.
  const targetPositionsRef = useRef<Float32Array>(positions); 
  const velocitiesRef = useRef<Float32Array>(velocities);

  // Sync refs when count changes
  useEffect(() => {
    velocitiesRef.current = velocities;
    targetPositionsRef.current = generateTargetPositions(count, shape);
  }, [count, velocities]);

  // Handle Shape Change
  useEffect(() => {
    targetPositionsRef.current = generateTargetPositions(count, shape);
  }, [shape, count]);

  // Uniforms
  const uniforms = useMemo(() => ({
    uTime: { value: 0 },
    uSize: { value: size },
    uColor: { value: new THREE.Color(color) },
    uScale: { value: 1.0 },
    uOpacity: { value: 0.8 }
  }), []);

  // Update loop
  useFrame((state, delta) => {
    if (!mesh.current) return;
    
    const dt = Math.min(delta, 0.1);

    // Update Uniforms
    const material = mesh.current.material as THREE.ShaderMaterial;
    if (material.uniforms) {
        material.uniforms.uTime.value = state.clock.elapsedTime;
        material.uniforms.uSize.value = size * (window.devicePixelRatio || 1);
        material.uniforms.uColor.value.lerp(new THREE.Color(color), 0.1);

        // Gesture Influence
        const { tension, distance, isPresent, handCount } = gestureState.current;
        let targetScale = 1.0;
        if (isPresent && handCount >= 2) {
            targetScale = THREE.MathUtils.mapLinear(Math.max(0.1, Math.min(distance, 0.8)), 0.1, 0.8, 0.5, 2.5);
        }
        material.uniforms.uScale.value = THREE.MathUtils.lerp(
            material.uniforms.uScale.value,
            targetScale,
            0.1
        );
    }

    // --- PHYSICS SIMULATION ---
    const geometry = mesh.current.geometry;
    if (!geometry || !geometry.attributes.position) return;

    const posAttribute = geometry.attributes.position;
    const currentPos = posAttribute.array as Float32Array;
    const targets = targetPositionsRef.current;
    const vel = velocitiesRef.current;

    // Safety check for array lengths
    if (currentPos.length !== targets.length || currentPos.length !== vel.length) return;
    
    const { isPresent, tension } = gestureState.current;
    
    // Dynamic physics parameters
    // When bursting (tension > 0.6), reduce spring stiffness so particles can fly away freely
    // instead of being "squashed" against their target positions.
    const isBursting = isPresent && tension > 0.6;
    const currentStiffness = isBursting ? 0.05 : 3.0;
    const currentDamping = isBursting ? 0.95 : 0.90; // Less friction when flying

    for (let i = 0; i < count; i++) {
      const i3 = i * 3;
      
      let cx = currentPos[i3];
      let cy = currentPos[i3 + 1];
      let cz = currentPos[i3 + 2];
      
      let vx = vel[i3];
      let vy = vel[i3 + 1];
      let vz = vel[i3 + 2];
      
      const tx = targets[i3];
      const ty = targets[i3 + 1];
      const tz = targets[i3 + 2];
      
      // Forces
      let fx = (tx - cx) * currentStiffness;
      let fy = (ty - cy) * currentStiffness;
      let fz = (tz - cz) * currentStiffness;
      
      // Global Forces
      fy -= gravity * 2.5; 
      fx += wind * 2.5;
      
      // Floor Boundary (Bounce/Slide)
      if (cy < -6.0) {
         fy += ((-6.0 - cy) * 20.0); // Push up strongly
         vy *= 0.5; // Friction on floor
      }

      // Noise (Always add a tiny bit of micro-jitter to prevent "dead" look)
      const time = state.clock.elapsedTime;
      const baseNoise = 0.05; 
      const totalNoise = noiseStrength + baseNoise;
      
      const nx = cx * 0.5 + time;
      const ny = cy * 0.5 + time;
      
      fx += Math.sin(ny * 2.0) * totalNoise * 5.0;
      fy += Math.cos(nx * 2.0) * totalNoise * 5.0;
      fz += Math.sin(time + cx) * totalNoise * 5.0;

      // Interaction
      if (isPresent) {
         const { x: hx, y: hy, z: hz } = gestureState.current.position;
         
         const dx = hx - cx;
         const dy = hy - cy;
         const dz = hz - cz;
         
         // If hand is open (tension > 0.6), BURST/REPEL strongly
         // If hand is closed/relaxed, ATTRACT
         
         if (tension > 0.6) {
             // Burst Mode
             const distSq = dx*dx + dy*dy + dz*dz;
             const dist = Math.sqrt(distSq) + 0.001;
             
             // Stronger, larger repulsion field
             const repulsionRadius = 8.0; 
             const burstStrength = (tension - 0.5) * 50.0; // Reduced from 80

             if (dist < repulsionRadius) {
                 // Exponential falloff for punchier feel
                 const force = (1.0 - dist / repulsionRadius) * burstStrength;
                 
                 // Reduced multiplier from 5.0 to 2.0 to prevent shooting off screen
                 fx -= (dx / dist) * force * 2.0;
                 fy -= (dy / dist) * force * 2.0;
                 fz -= (dz / dist) * force * 2.0;
             }
         } else {
             // Attraction Mode (Follow Hand)
             // Stronger attraction when fist is tight (tension near 0)
             const attractStrength = (1.0 - tension) * 5.0;
             
             fx += dx * attractStrength;
             fy += dy * attractStrength;
             fz += dz * attractStrength;
         }
      }

      // Integration
      vx += fx * dt;
      vy += fy * dt;
      vz += fz * dt;
      
      vx *= currentDamping;
      vy *= currentDamping;
      vz *= currentDamping;
      
      cx += vx * dt;
      cy += vy * dt;
      cz += vz * dt;
      
      currentPos[i3] = cx;
      currentPos[i3 + 1] = cy;
      currentPos[i3 + 2] = cz;
      
      vel[i3] = vx;
      vel[i3 + 1] = vy;
      vel[i3 + 2] = vz;
    }
    
    posAttribute.needsUpdate = true;
  });

  return (
    <points ref={mesh} frustumCulled={false}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={positions.length / 3}
          array={positions}
          itemSize={3}
          usage={THREE.DynamicDrawUsage}
        />
        <bufferAttribute
          attach="attributes-aScale"
          count={scales.length}
          array={scales}
          itemSize={1}
        />
        <bufferAttribute
          attach="attributes-aRandom"
          count={randoms.length / 3}
          array={randoms}
          itemSize={3}
        />
      </bufferGeometry>
      <shaderMaterial
        vertexShader={vertexShader}
        fragmentShader={fragmentShader}
        uniforms={uniforms}
        transparent={true}
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
};

// --- Hand Tracker ---

const HandTracker = ({ 
  isEnabled, 
  onGestureUpdate,
  videoRef
}: { 
  isEnabled: boolean;
  onGestureUpdate: (state: HandGestureState) => void;
  videoRef: React.RefObject<HTMLVideoElement>;
}) => {
  useEffect(() => {
    if (!isEnabled || !videoRef.current) return;

    let hands: any;
    let camera: any;

    const onResults = (results: any) => {
      const multiHandLandmarks = results.multiHandLandmarks;
      
      if (!multiHandLandmarks || multiHandLandmarks.length === 0) {
        onGestureUpdate({ 
          tension: 0.5, 
          distance: 0.5, 
          isPresent: false, 
          handCount: 0,
          position: { x: 0, y: 0, z: 0 }
        });
        return;
      }

      let totalTension = 0;
      let totalX = 0;
      let totalY = 0;
      let totalZ = 0;
      let pointCount = 0;

      multiHandLandmarks.forEach((landmarks: any) => {
        const wrist = landmarks[0];
        const middleTip = landmarks[12];
        const middleMCP = landmarks[9];
        
        // Calculate centroid
        landmarks.forEach((lm: any) => {
            totalX += lm.x;
            totalY += lm.y;
            totalZ += lm.z;
            pointCount++;
        });

        const palmSize = Math.sqrt(
          Math.pow(wrist.x - middleMCP.x, 2) + Math.pow(wrist.y - middleMCP.y, 2)
        );
        
        const fingers = [8, 12, 16, 20, 4];
        let extensions = 0;
        fingers.forEach(idx => {
           const tip = landmarks[idx];
           const dist = Math.sqrt(Math.pow(wrist.x - tip.x, 2) + Math.pow(wrist.y - tip.y, 2));
           extensions += (dist / palmSize); 
        });
        
        const avgExt = extensions;
        const t = Math.min(1, Math.max(0, (avgExt - 5.0) / 4.0));
        totalTension += t;
      });
      
      const avgTension = totalTension / multiHandLandmarks.length;
      
      // Calculate average position
      const avgX = pointCount > 0 ? totalX / pointCount : 0.5;
      const avgY = pointCount > 0 ? totalY / pointCount : 0.5;
      const avgZ = pointCount > 0 ? totalZ / pointCount : 0;

      // Map to world coordinates
      // Invert X for mirror effect: (0.5 - avgX)
      const worldX = (0.5 - avgX) * 16; 
      const worldY = (0.5 - avgY) * 10;
      const worldZ = avgZ * 5;

      let handDist = 0.5; 
      if (multiHandLandmarks.length >= 2) {
        const h1 = multiHandLandmarks[0][0]; 
        const h2 = multiHandLandmarks[1][0]; 
        handDist = Math.sqrt(Math.pow(h1.x - h2.x, 2) + Math.pow(h1.y - h2.y, 2));
      }

      onGestureUpdate({
        tension: avgTension,
        distance: handDist,
        isPresent: true,
        handCount: multiHandLandmarks.length,
        position: { x: worldX, y: worldY, z: worldZ }
      });
    };

    const loadMediaPipe = async () => {
        // @ts-ignore
        if (typeof window.Hands === 'undefined' || typeof window.Camera === 'undefined') {
          console.warn("MediaPipe scripts loading...");
          setTimeout(loadMediaPipe, 500);
          return;
        }

        try {
            console.log("Initializing MediaPipe Hands...");
            // @ts-ignore
            hands = new window.Hands({locateFile: (file) => {
               return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
            }});

            hands.setOptions({
              maxNumHands: 2,
              modelComplexity: 1,
              minDetectionConfidence: 0.5,
              minTrackingConfidence: 0.5
            });

            hands.onResults(onResults);

            if (videoRef.current && isMounted) {
               // Ensure video is ready for stream
               videoRef.current.muted = true;
               videoRef.current.setAttribute('playsinline', 'true');
               
               // @ts-ignore
               camera = new window.Camera(videoRef.current, {
                onFrame: async () => {
                   if (videoRef.current && hands && isMounted) {
                      await hands.send({image: videoRef.current});
                   }
                },
                width: 640,
                height: 480
               });
               await camera.start();
               if (isMounted) console.log("Camera started");
            }
        } catch (e) {
            console.error("Failed to initialize MediaPipe", e);
        }
    };

    // Use a flag to prevent multiple initializations or race conditions
    let isMounted = true;
    loadMediaPipe();

    return () => {
      isMounted = false;
      if (camera) {
          camera.stop();
          camera = null;
      }
      if (hands) {
          hands.close();
          hands = null;
      }
    };
  }, [isEnabled, videoRef]);

  return null;
};

// --- Main App ---

export default function App() {
  const [shape, setShape] = useState<ShapeType>('sphere');
  const [color, setColor] = useState('#60a5fa');
  const [particleCount, setParticleCount] = useState(5000);
  const [particleSize, setParticleSize] = useState(25);
  const [noiseStrength, setNoiseStrength] = useState(0.2);
  const [gravity, setGravity] = useState(0);
  const [wind, setWind] = useState(0);
  
  const [cameraEnabled, setCameraEnabled] = useState(false);
  const [musicEnabled, setMusicEnabled] = useState(false);
  const [showControls, setShowControls] = useState(true);

  const gestureState = useRef<HandGestureState>({ 
    tension: 0.5, 
    distance: 0.5, 
    isPresent: false, 
    handCount: 0,
    position: { x: 0, y: 0, z: 0 }
  });
  
  const videoRef = useRef<HTMLVideoElement>(null);

  const takeScreenshot = () => {
    const canvas = document.querySelector('canvas');
    if (canvas) {
      const link = document.createElement('a');
      link.download = 'particles.png';
      link.href = canvas.toDataURL('image/png');
      link.click();
    }
  };

  const shapes: {id: ShapeType, label: string, icon: any}[] = [
    { id: 'sphere', label: 'Sphere', icon: Disc },
    { id: 'heart', label: 'Heart', icon: Heart },
    { id: 'flower', label: 'Flower', icon: Flower },
    { id: 'saturn', label: 'Saturn', icon: Aperture },
    { id: 'fireworks', label: 'Burst', icon: Zap },
  ];

  return (
    <div className="relative w-full h-screen bg-black overflow-hidden font-sans text-white">
      {/* 3D Canvas */}
      <div className="absolute inset-0 z-0">
        <Canvas camera={{ position: [0, 0, 8], fov: 60 }} gl={{ preserveDrawingBuffer: true }}>
          <ambientLight intensity={0.5} />
          <AmbientSound enabled={musicEnabled} gestureState={gestureState} />
          <Particles 
            count={particleCount}
            shape={shape}
            color={color}
            size={particleSize}
            noiseStrength={noiseStrength}
            gravity={gravity}
            wind={wind}
            gestureState={gestureState}
          />
        </Canvas>
      </div>

      {/* Hidden Video for MP */}
      <video 
        ref={videoRef} 
        className="absolute top-0 left-0 opacity-0 pointer-events-none" 
        style={{ width: '640px', height: '480px' }}
        playsInline 
        muted
        autoPlay
      />
      
      <HandTracker 
        isEnabled={cameraEnabled} 
        videoRef={videoRef}
        onGestureUpdate={(state) => {
          gestureState.current = state;
        }}
      />

      {/* UI Overlay */}
      <div className={`absolute top-4 left-4 z-10 transition-transform duration-300 ${showControls ? 'translate-x-0' : '-translate-x-full'}`}>
        <div className="bg-black/40 backdrop-blur-md border border-white/10 p-6 rounded-2xl w-80 shadow-2xl h-[90vh] overflow-y-auto">
          <div className="flex justify-between items-center mb-6">
            <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400">
              Particle Play
            </h1>
            <div className="flex gap-2">
               <button 
                 onClick={() => setMusicEnabled(!musicEnabled)} 
                 className={`p-2 rounded-full transition ${musicEnabled ? 'bg-blue-500/20 text-blue-400' : 'hover:bg-white/10 text-gray-400'}`}
                 title="Toggle Music"
               >
                 {musicEnabled ? <Volume2 size={18} /> : <VolumeX size={18} />}
               </button>
               <button onClick={takeScreenshot} className="p-2 hover:bg-white/10 rounded-full transition" title="Save Image">
                 <Camera size={18} />
               </button>
            </div>
          </div>

          <div className="mb-6 p-4 bg-white/5 rounded-xl border border-white/5">
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm font-medium flex items-center gap-2">
                {cameraEnabled ? <Video size={16} className="text-green-400"/> : <VideoOff size={16} className="text-gray-400"/>}
                Gesture Control
              </span>
              <button 
                onClick={() => setCameraEnabled(!cameraEnabled)}
                className={`w-12 h-6 rounded-full relative transition-colors ${cameraEnabled ? 'bg-blue-600' : 'bg-gray-600'}`}
              >
                <div className={`absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform ${cameraEnabled ? 'translate-x-6' : ''}`} />
              </button>
            </div>
            <p className="text-xs text-gray-400 leading-relaxed">
              {cameraEnabled 
                ? "Move hand to attract particles. Open palm to repel." 
                : "Enable camera to interact with hands."}
            </p>
          </div>

          <div className="space-y-6">
            <div>
              <label className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3 block">Shape</label>
              <div className="grid grid-cols-5 gap-2">
                {shapes.map((s) => (
                  <button
                    key={s.id}
                    onClick={() => setShape(s.id)}
                    className={`flex flex-col items-center justify-center p-2 rounded-lg transition-all ${
                      shape === s.id ? 'bg-blue-500/20 text-blue-400 border border-blue-500/50' : 'hover:bg-white/5 text-gray-400'
                    }`}
                  >
                    <s.icon size={20} />
                  </button>
                ))}
              </div>
            </div>

            <div className="space-y-4">
               <div>
                  <div className="flex justify-between text-xs text-gray-400 mb-1">
                    <span>Count</span>
                    <span>{particleCount}</span>
                  </div>
                  <input 
                    type="range" min="1000" max="20000" step="1000" 
                    value={particleCount} 
                    onChange={(e) => setParticleCount(Number(e.target.value))}
                    className="w-full h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                  />
               </div>
               
               <div>
                  <div className="flex justify-between text-xs text-gray-400 mb-1">
                    <span>Size</span>
                    <span>{particleSize}px</span>
                  </div>
                  <input 
                    type="range" min="10" max="100" step="1" 
                    value={particleSize} 
                    onChange={(e) => setParticleSize(Number(e.target.value))}
                    className="w-full h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                  />
               </div>

               <div>
                  <div className="flex justify-between text-xs text-gray-400 mb-1">
                    <span>Chaos / Noise</span>
                    <span>{Math.round(noiseStrength * 100)}%</span>
                  </div>
                  <input 
                    type="range" min="0" max="2" step="0.1" 
                    value={noiseStrength} 
                    onChange={(e) => setNoiseStrength(Number(e.target.value))}
                    className="w-full h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
                  />
               </div>

               <div className="pt-2 border-t border-white/5 mt-4">
                 <label className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3 block">Physics</label>
                 
                 <div className="mb-4">
                    <div className="flex justify-between text-xs text-gray-400 mb-1 items-center">
                      <span className="flex items-center gap-1"><ArrowDown size={12}/> Gravity</span>
                      <span>{gravity}</span>
                    </div>
                    <input 
                      type="range" min="0" max="10" step="0.5" 
                      value={gravity} 
                      onChange={(e) => setGravity(Number(e.target.value))}
                      className="w-full h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-purple-500"
                    />
                 </div>

                 <div>
                    <div className="flex justify-between text-xs text-gray-400 mb-1 items-center">
                      <span className="flex items-center gap-1"><Wind size={12}/> Wind</span>
                      <span>{wind}</span>
                    </div>
                    <input 
                      type="range" min="-10" max="10" step="0.5" 
                      value={wind} 
                      onChange={(e) => setWind(Number(e.target.value))}
                      className="w-full h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-purple-500"
                    />
                 </div>
               </div>
            </div>

            <div>
               <label className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3 block">Color</label>
               <div className="flex gap-2">
                  {['#60a5fa', '#f472b6', '#a78bfa', '#34d399', '#fbbf24', '#ffffff'].map(c => (
                    <button 
                      key={c}
                      onClick={() => setColor(c)}
                      className={`w-8 h-8 rounded-full border-2 transition-transform hover:scale-110 ${color === c ? 'border-white' : 'border-transparent'}`}
                      style={{ backgroundColor: c }}
                    />
                  ))}
               </div>
            </div>
          </div>
        </div>
      </div>

      <button 
        onClick={() => setShowControls(!showControls)}
        className="absolute top-4 left-4 z-20 p-2 bg-white/10 backdrop-blur rounded-lg hover:bg-white/20 transition-opacity"
        style={{ opacity: showControls ? 0 : 1 }}
      >
        <Settings2 size={20} />
      </button>

      {cameraEnabled && (
         <div className="absolute top-4 right-4 z-10 flex flex-col gap-2 items-end">
           <div className="flex items-center gap-2 bg-black/60 backdrop-blur px-3 py-1.5 rounded-full border border-green-500/30">
              <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
              <span className="text-xs font-mono text-green-400">CAMERA ACTIVE</span>
           </div>
         </div>
      )}
    </div>
  );
}

const container = document.getElementById('root');
if (container) {
  const root = createRoot(container);
  root.render(<App />);
}