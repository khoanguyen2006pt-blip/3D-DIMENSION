import React, { useEffect, useRef, useState, useMemo } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { DataPoint, ColorMode } from '../types';
import { getErrorMessage } from '../utils/typeGuards';
import {
  RotateCcw,
  Play,
  Pause,
  ZoomIn,
  ZoomOut,
  Camera,
  Layers,
  Compass,
  Palette,
  ChevronRight,
  ChevronLeft,
  Sliders,
} from 'lucide-react';

interface ThreeCanvasProps {
  dataPoints: DataPoint[];
  colorMode: ColorMode;
  selectedFeature?: string;
  onSelectPoint: (point: DataPoint | null) => void;
  selectedPoint: DataPoint | null;
  reductionMethod: string;
  onCameraChange?: (camPos: THREE.Vector3, targetPos: THREE.Vector3) => void;
  syncCameraPos?: THREE.Vector3 | null;
  syncTargetPos?: THREE.Vector3 | null;
  titleBadge?: string;
  axisFeatures?: string[];
  showCentroids?: boolean;
  pcaComponents?: number[][];
  featureNames?: string[];
  showBiplot?: boolean;
  biplotTopK?: number;
  biplotScale?: number;
  onToggleBiplot?: () => void;
  theme?: 'dark' | 'light';
  onToggleTheme?: () => void;
}

// Categorical palette for cluster coloring (independent from UI theme)
const CLUSTER_COLORS = [
  '#4E79A7', // Steel Blue
  '#F28E2B', // Orange
  '#E15759', // Coral Red
  '#76B7B2', // Teal
  '#59A14F', // Green
  '#EDC948', // Yellow
  '#B07AA1', // Purple
  '#FF9DA7', // Pink
  '#9C755F', // Brown
  '#BAB0AC', // Gray
];

export const ThreeCanvas: React.FC<ThreeCanvasProps> = ({
  dataPoints = [],
  colorMode,
  selectedFeature,
  onSelectPoint,
  selectedPoint,
  reductionMethod,
  onCameraChange,
  syncCameraPos,
  syncTargetPos,
  titleBadge,
  axisFeatures,
  showCentroids = true,
  pcaComponents,
  featureNames,
  showBiplot = false,
  biplotTopK = 6,
  biplotScale = 30,
  onToggleBiplot,
}) => {
  const canvasMountRef = useRef<HTMLDivElement>(null);
  const [hoveredPoint, setHoveredPoint] = useState<DataPoint | null>(null);
  const [tooltipPos, setTooltipPos] = useState<{ x: number; y: number } | null>(null);
  const [isAutoRotating, setIsAutoRotating] = useState(false);
  const [isToolbarCollapsed, setIsToolbarCollapsed] = useState(false);
  const [pointScale, setPointScale] = useState(1.3);
  const [centroidsVisible, setCentroidsVisible] = useState(showCentroids);
  const [biplotVisible, setBiplotVisible] = useState(showBiplot);
  const [isLegendExpanded, setIsLegendExpanded] = useState(true);
  const [snapshotError, setSnapshotError] = useState<string | null>(null);

  // Dynamic ground-truth classification labels extracted from dataPoints
  const uniqueLabels = useMemo(() => {
    const set = new Set<string>();
    for (const p of dataPoints) {
      if (p.label !== undefined && p.label !== null && String(p.label).trim() !== '') {
        set.add(String(p.label).trim());
      }
    }
    return Array.from(set).sort();
  }, [dataPoints]);

  const labelColorMap = useMemo(() => {
    const map: Record<string, string> = {};
    uniqueLabels.forEach((label, idx) => {
      if (label === 'M' || label === '1') map[label] = '#E15759';
      else if (label === 'B' || label === '0') map[label] = '#59A14F';
      else map[label] = CLUSTER_COLORS[idx % CLUSTER_COLORS.length];
    });
    return map;
  }, [uniqueLabels]);

  useEffect(() => {
    if (showBiplot !== undefined) {
      setBiplotVisible(showBiplot);
    }
  }, [showBiplot]);

  // Persistent Three.js references
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const instancedMeshRef = useRef<THREE.InstancedMesh | null>(null);
  const centroidGroupRef = useRef<THREE.Group | null>(null);
  const biplotGroupRef = useRef<THREE.Group | null>(null);

  // Normalized points to fit cleanly in [-36, 36] coordinate box
  const normalizedPoints = useMemo(() => {
    if (dataPoints.length === 0) return [];

    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    let minZ = Infinity, maxZ = -Infinity;

    for (const p of dataPoints) {
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
      if (p.z < minZ) minZ = p.z;
      if (p.z > maxZ) maxZ = p.z;
    }

    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;
    const rangeZ = maxZ - minZ || 1;
    const maxRange = Math.max(rangeX, rangeY, rangeZ);

    const midX = (minX + maxX) / 2;
    const midY = (minY + maxY) / 2;
    const midZ = (minZ + maxZ) / 2;

    const scale = 72 / maxRange;

    return dataPoints.map((p) => ({
      ...p,
      x: (p.x - midX) * scale,
      y: (p.y - midY) * scale,
      z: (p.z - midZ) * scale,
    }));
  }, [dataPoints]);

  const pointsRef = useRef(dataPoints);
  pointsRef.current = dataPoints;

  // 1. INITIALIZE THREE.JS SCENE
  useEffect(() => {
    const container = canvasMountRef.current;
    if (!container) return;

    const width = container.clientWidth || 600;
    const height = container.clientHeight || 500;

    // Scientific Background: #0D1117
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0D1117);
    sceneRef.current = scene;

    // Centroid Group
    const centroidGroup = new THREE.Group();
    scene.add(centroidGroup);
    centroidGroupRef.current = centroidGroup;

    // Biplot Feature Vectors Group
    const biplotGroup = new THREE.Group();
    scene.add(biplotGroup);
    biplotGroupRef.current = biplotGroup;

    // Camera
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.set(65, 45, 80);
    cameraRef.current = camera;

    // WebGL Renderer
    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      powerPreference: 'high-performance',
      preserveDrawingBuffer: true,
    });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;

    renderer.domElement.style.width = '100%';
    renderer.domElement.style.height = '100%';
    renderer.domElement.style.display = 'block';
    renderer.domElement.style.touchAction = 'none';
    renderer.domElement.style.cursor = 'grab';

    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // OrbitControls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.rotateSpeed = 0.9;
    controls.zoomSpeed = 1.1;
    controls.panSpeed = 0.9;
    controls.maxDistance = 260;
    controls.minDistance = 15;
    controls.autoRotate = false;
    controls.autoRotateSpeed = 1.2;

    controls.touches = {
      ONE: THREE.TOUCH.ROTATE,
      TWO: THREE.TOUCH.DOLLY_PAN,
    };
    controlsRef.current = controls;

    if (onCameraChange) {
      controls.addEventListener('change', () => {
        onCameraChange(camera.position, controls.target);
      });
    }

    controls.addEventListener('start', () => {
      renderer.domElement.style.cursor = 'grabbing';
    });
    controls.addEventListener('end', () => {
      renderer.domElement.style.cursor = 'grab';
    });

    // Clean, Neutral Lighting (avoids tinting scientific clusters)
    const ambientLight = new THREE.AmbientLight(0xffffff, 1.1);
    scene.add(ambientLight);

    const keyLight = new THREE.DirectionalLight(0xffffff, 1.2);
    keyLight.position.set(60, 80, 60);
    scene.add(keyLight);

    const fillLight = new THREE.DirectionalLight(0xd1d5db, 0.7);
    fillLight.position.set(-60, -40, -60);
    scene.add(fillLight);

    // Subtle Scientific Floor Grid (#303A49 and #1c2430)
    const gridHelper = new THREE.GridHelper(80, 20, 0x303A49, 0x1a222d);
    gridHelper.position.y = -36;
    scene.add(gridHelper);

    // Subtle Bounding Wireframe Box
    const boxGeo = new THREE.BoxGeometry(72, 72, 72);
    const boxMat = new THREE.MeshBasicMaterial({
      color: 0x303A49,
      wireframe: true,
      transparent: true,
      opacity: 0.18,
    });
    scene.add(new THREE.Mesh(boxGeo, boxMat));

    // Pointer Events for Raycasting (Hover / Select point)
    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();

    const updatePointer = (clientX: number, clientY: number) => {
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((clientY - rect.top) / rect.height) * 2 + 1;
    };

    let pointerDownPos = { x: 0, y: 0 };
    const handlePointerDown = (e: PointerEvent) => {
      pointerDownPos = { x: e.clientX, y: e.clientY };
    };

    const handlePointerUp = (e: PointerEvent) => {
      const dist = Math.hypot(e.clientX - pointerDownPos.x, e.clientY - pointerDownPos.y);
      if (dist < 5) {
        updatePointer(e.clientX, e.clientY);
        const mesh = instancedMeshRef.current;
        if (!mesh) return;

        raycaster.setFromCamera(pointer, camera);
        const intersects = raycaster.intersectObject(mesh);

        if (intersects.length > 0 && intersects[0].instanceId !== undefined) {
          const idx = intersects[0].instanceId;
          const pt = pointsRef.current[idx];
          if (pt) onSelectPoint(pt);
        }
      }
    };

    const handlePointerMove = (e: PointerEvent) => {
      updatePointer(e.clientX, e.clientY);
      const mesh = instancedMeshRef.current;
      if (!mesh) return;

      raycaster.setFromCamera(pointer, camera);
      const intersects = raycaster.intersectObject(mesh);

      if (intersects.length > 0 && intersects[0].instanceId !== undefined) {
        const idx = intersects[0].instanceId;
        const pt = pointsRef.current[idx];
        setHoveredPoint(pt || null);
        setTooltipPos({ x: e.clientX, y: e.clientY });
        renderer.domElement.style.cursor = 'pointer';
      } else {
        setHoveredPoint(null);
        setTooltipPos(null);
        renderer.domElement.style.cursor = 'grab';
      }
    };

    const dom = renderer.domElement;
    dom.addEventListener('pointerdown', handlePointerDown);
    dom.addEventListener('pointerup', handlePointerUp);
    dom.addEventListener('pointermove', handlePointerMove);

    // Keyboard Navigation
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.key === 'r' || e.key === 'R') {
        handleResetCamera();
      } else if (e.key === '+' || e.key === '=') {
        handleZoom('in');
      } else if (e.key === '-' || e.key === '_') {
        handleZoom('out');
      }
    };
    window.addEventListener('keydown', handleKeyDown);

    // Animation Loop
    let animId: number;
    const animate = () => {
      animId = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const w = entry.contentRect.width;
        const h = entry.contentRect.height;
        if (w > 0 && h > 0) {
          camera.aspect = w / h;
          camera.updateProjectionMatrix();
          renderer.setSize(w, h);
        }
      }
    });
    resizeObserver.observe(container);

    return () => {
      resizeObserver.disconnect();
      window.removeEventListener('keydown', handleKeyDown);
      dom.removeEventListener('pointerdown', handlePointerDown);
      dom.removeEventListener('pointerup', handlePointerUp);
      dom.removeEventListener('pointermove', handlePointerMove);
      cancelAnimationFrame(animId);
      if (container.contains(dom)) {
        container.removeChild(dom);
      }
      renderer.dispose();
    };
  }, [onSelectPoint, onCameraChange]);

  // Dual Viewport Sync
  useEffect(() => {
    if (syncCameraPos && syncTargetPos && cameraRef.current && controlsRef.current) {
      cameraRef.current.position.copy(syncCameraPos);
      controlsRef.current.target.copy(syncTargetPos);
      controlsRef.current.update();
    }
  }, [syncCameraPos, syncTargetPos]);

  // Auto-rotate toggle
  useEffect(() => {
    if (controlsRef.current) {
      controlsRef.current.autoRotate = isAutoRotating;
    }
  }, [isAutoRotating]);

  // 2. UPDATE INSTANCED MESH DATA POINTS & CENTROIDS
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;

    if (instancedMeshRef.current) {
      scene.remove(instancedMeshRef.current);
      instancedMeshRef.current.geometry.dispose();
      (instancedMeshRef.current.material as THREE.Material).dispose();
      instancedMeshRef.current = null;
    }

    if (centroidGroupRef.current) {
      while (centroidGroupRef.current.children.length > 0) {
        const obj = centroidGroupRef.current.children[0];
        centroidGroupRef.current.remove(obj);
      }
    }

    const count = normalizedPoints.length;
    if (count === 0) return;

    const sphereGeo = new THREE.SphereGeometry(1, 16, 16);
    const sphereMat = new THREE.MeshStandardMaterial({
      roughness: 0.25,
      metalness: 0.35,
    });

    const mesh = new THREE.InstancedMesh(sphereGeo, sphereMat, count);
    const dummy = new THREE.Object3D();
    const color = new THREE.Color();

    let fMin = Infinity, fMax = -Infinity;
    if (colorMode === 'feature' && selectedFeature) {
      for (const p of dataPoints) {
        const val = p.originalFeatures[selectedFeature];
        if (typeof val === 'number') {
          if (val < fMin) fMin = val;
          if (val > fMax) fMax = val;
        }
      }
    }
    const fRange = fMax - fMin || 1;

    for (let i = 0; i < count; i++) {
      const pt = normalizedPoints[i];
      const isSelected = selectedPoint && selectedPoint.id === pt.id;
      const r = (isSelected ? 1.9 : 1.0) * pointScale;

      dummy.position.set(pt.x, pt.y, pt.z);
      dummy.scale.set(r, r, r);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);

      if (isSelected) {
        color.set('#FFFFFF');
      } else if (colorMode === 'ground-truth' && pt.label !== undefined && pt.label !== null) {
        const hex = labelColorMap[String(pt.label)] || '#929EAF';
        color.set(hex);
      } else if (colorMode === 'cluster') {
        if (pt.cluster === -1) {
          color.set('#64748B'); // Noise
        } else {
          const hex = CLUSTER_COLORS[pt.cluster % CLUSTER_COLORS.length] || '#4E79A7';
          color.set(hex);
        }
      } else if (colorMode === 'feature' && selectedFeature) {
        const val = pt.originalFeatures[selectedFeature] || 0;
        const norm = Math.max(0, Math.min(1, (val - fMin) / fRange));
        color.setHSL(0.66 - norm * 0.66, 0.85, 0.55);
      } else {
        color.set('#8B9CFB');
      }

      mesh.setColorAt(i, color);
    }

    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;

    scene.add(mesh);
    instancedMeshRef.current = mesh;

    // Centroids visualization
    if (centroidsVisible && colorMode === 'cluster') {
      const clusterSums: Record<number, { x: number; y: number; z: number; count: number }> = {};
      for (const p of normalizedPoints) {
        if (p.cluster === -1) continue;
        if (!clusterSums[p.cluster]) {
          clusterSums[p.cluster] = { x: 0, y: 0, z: 0, count: 0 };
        }
        clusterSums[p.cluster].x += p.x;
        clusterSums[p.cluster].y += p.y;
        clusterSums[p.cluster].z += p.z;
        clusterSums[p.cluster].count += 1;
      }

      Object.entries(clusterSums).forEach(([clStr, sum]) => {
        const cId = parseInt(clStr, 10);
        const cx = sum.x / sum.count;
        const cy = sum.y / sum.count;
        const cz = sum.z / sum.count;
        const cColor = CLUSTER_COLORS[cId % CLUSTER_COLORS.length] || '#8B9CFB';

        const coreGeo = new THREE.SphereGeometry(2.0, 16, 16);
        const coreMat = new THREE.MeshBasicMaterial({
          color: new THREE.Color(cColor),
          wireframe: true,
          transparent: true,
          opacity: 0.8,
        });
        const coreMesh = new THREE.Mesh(coreGeo, coreMat);
        coreMesh.position.set(cx, cy, cz);

        centroidGroupRef.current?.add(coreMesh);
      });
    }
  }, [normalizedPoints, colorMode, selectedFeature, selectedPoint, pointScale, centroidsVisible, dataPoints, labelColorMap]);

  // 3. RENDER PCA FEATURE LOADINGS (BIPLOT)
  useEffect(() => {
    if (!biplotGroupRef.current) return;

    while (biplotGroupRef.current.children.length > 0) {
      const obj = biplotGroupRef.current.children[0];
      biplotGroupRef.current.remove(obj);
    }

    if (!biplotVisible || reductionMethod !== 'PCA' || !pcaComponents || !featureNames || pcaComponents.length < 3) {
      return;
    }

    const d = featureNames.length;
    const v1 = pcaComponents[0];
    const v2 = pcaComponents[1];
    const v3 = pcaComponents[2];
    if (!v1 || !v2 || !v3) return;

    const featureMagnitudes: { name: string; x: number; y: number; z: number; mag: number }[] = [];
    for (let j = 0; j < d; j++) {
      const x = v1[j] || 0;
      const y = v2[j] || 0;
      const z = v3[j] || 0;
      const mag = Math.hypot(x, y, z);
      featureMagnitudes.push({ name: featureNames[j], x, y, z, mag });
    }

    featureMagnitudes.sort((a, b) => b.mag - a.mag);
    const topFeatures = featureMagnitudes.slice(0, biplotTopK);

    topFeatures.forEach((feat) => {
      const dir = new THREE.Vector3(feat.x, feat.y, feat.z).normalize();
      const origin = new THREE.Vector3(0, 0, 0);

      const arrowHelper = new THREE.ArrowHelper(dir, origin, biplotScale, 0x8B9CFB, 3.0, 1.4);
      biplotGroupRef.current?.add(arrowHelper);

      const sprite = createTextLabelSprite(feat.name, '#8B9CFB');
      sprite.position.copy(dir).multiplyScalar(biplotScale + 4);
      biplotGroupRef.current?.add(sprite);
    });
  }, [biplotVisible, reductionMethod, pcaComponents, featureNames, biplotTopK, biplotScale]);

  const handleResetCamera = () => {
    if (controlsRef.current && cameraRef.current) {
      cameraRef.current.position.set(65, 45, 80);
      controlsRef.current.target.set(0, 0, 0);
      controlsRef.current.update();
    }
  };

  const handleZoom = (inOut: 'in' | 'out') => {
    if (cameraRef.current && controlsRef.current) {
      const factor = inOut === 'in' ? 0.78 : 1.25;
      cameraRef.current.position.multiplyScalar(factor);
      controlsRef.current.update();
    }
  };

  const handleTakeSnapshot = () => {
    if (!rendererRef.current) return;
    try {
      const dataUrl = rendererRef.current.domElement.toDataURL('image/png');
      const link = document.createElement('a');
      link.download = `3d_explorer_${reductionMethod.toLowerCase()}_${Date.now()}.png`;
      link.href = dataUrl;
      link.click();
    } catch (e: unknown) {
      setSnapshotError(`Lỗi chụp ảnh: ${getErrorMessage(e)}`);
      window.setTimeout(() => setSnapshotError(null), 5000);
    }
  };

  return (
    <div className="relative w-full h-full flex flex-row overflow-hidden select-none bg-[#0D1117] text-[#E6EAF0]">
      {/* 1. MAIN 3D CANVAS VIEWPORT (Takes maximal area) */}
      <div className="relative flex-1 h-full overflow-hidden">
        {/* Dedicated 3D Canvas Mount Point */}
        <div ref={canvasMountRef} className="absolute inset-0 w-full h-full" style={{ touchAction: 'none' }} />

        {/* Error Notification */}
        {snapshotError && (
          <div role="alert" className="absolute right-4 top-4 z-30 px-3 py-1.5 rounded bg-[#2d1b1f] border border-[#7f1d1d] text-xs text-[#f87171]">
            {snapshotError}
          </div>
        )}

        {/* Viewport Header Badge */}
        <div className="absolute left-3 top-3 z-20 flex items-center gap-2 px-3 py-1.5 rounded bg-[#151B23]/90 border border-[#303A49] text-xs backdrop-blur-sm">
          <span className="w-2 h-2 rounded-full bg-[#8B9CFB]" />
          <span className="font-semibold text-[11px] tracking-wide text-[#E6EAF0]">
            {titleBadge || `KHÔNG GIAN 3D (${reductionMethod})`}
          </span>
          <span className="text-[#303A49]">|</span>
          <span className="font-mono text-[11px] text-[#929EAF]">
            {dataPoints.length.toLocaleString('vi-VN')} điểm
          </span>
        </div>

        {/* Axis Info overlay if Raw Data */}
        {axisFeatures && axisFeatures.length > 0 && (
          <div className="absolute left-3 top-12 z-20 flex flex-wrap gap-1 max-w-sm">
            {axisFeatures.map((feat, idx) => (
              <span key={feat} className="px-2 py-0.5 rounded bg-[#151B23]/90 border border-[#303A49] text-[10px] text-[#929EAF]">
                Trục {idx === 0 ? 'X' : idx === 1 ? 'Y' : 'Z'}: <strong className="text-[#E6EAF0]">{feat}</strong>
              </span>
            ))}
          </div>
        )}

        {/* Floating Minimal Legend */}
        {colorMode !== 'ground-truth' && (
          <div className="absolute left-3 bottom-3 z-20 flex flex-col gap-1 p-2 rounded bg-[#151B23]/90 border border-[#303A49] text-[11px] backdrop-blur-sm">
            <div
              onClick={() => setIsLegendExpanded(!isLegendExpanded)}
              className="flex items-center justify-between gap-3 cursor-pointer select-none text-[10px] uppercase font-semibold text-[#929EAF]"
            >
              <div className="flex items-center gap-1.5">
                <Palette size={11} className="text-[#8B9CFB]" />
                <span>{colorMode === 'cluster' ? 'Cụm thuật toán' : 'Thang đo đặc trưng'}</span>
              </div>
              <span className="text-[9px]">{isLegendExpanded ? '▲' : '▼'}</span>
            </div>

            {isLegendExpanded && (
              <div className="pt-1.5 border-t border-[#303A49] flex flex-col gap-1">
                {colorMode === 'cluster' && (
                  <div className="flex items-center gap-2 flex-wrap max-w-xs">
                    {CLUSTER_COLORS.slice(0, 6).map((col, idx) => (
                      <div key={idx} className="flex items-center gap-1">
                        <span className="w-2 h-2 rounded-full inline-block" style={{ backgroundColor: col }} />
                        <span className="text-[#929EAF] font-mono text-[10px]">C{idx}</span>
                      </div>
                    ))}
                    <div className="flex items-center gap-1">
                      <span className="w-2 h-2 rounded-full bg-[#64748B] inline-block" />
                      <span className="text-[#929EAF] font-mono text-[10px]">Nhiễu</span>
                    </div>
                  </div>
                )}
                {colorMode === 'feature' && (
                  <div className="flex items-center gap-2">
                    <span className="text-[#38bdf8] font-mono text-[10px]">Thấp</span>
                    <div className="w-24 h-1.5 rounded bg-gradient-to-r from-blue-500 via-amber-400 to-rose-500" />
                    <span className="text-[#f87171] font-mono text-[10px]">Cao</span>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Point Hover Tooltip */}
        {hoveredPoint && tooltipPos && (
          <div
            className="fixed z-50 pointer-events-none transform -translate-x-1/2 -translate-y-full mb-3 text-xs p-2.5 rounded border border-[#303A49] bg-[#151B23] text-[#E6EAF0] shadow-xl backdrop-blur-md"
            style={{ left: `${tooltipPos.x}px`, top: `${tooltipPos.y - 8}px` }}
          >
            <div className="flex items-center justify-between gap-4 border-b border-[#303A49] pb-1 mb-1.5">
              <span className="text-[#8B9CFB] font-mono font-bold">ID: {String(hoveredPoint.id)}</span>
              {hoveredPoint.label !== undefined && (
                <span
                  className="px-1.5 py-0.2 rounded text-[10px] font-mono font-bold border"
                  style={{
                    backgroundColor: `${labelColorMap[String(hoveredPoint.label)] || '#64748B'}20`,
                    borderColor: `${labelColorMap[String(hoveredPoint.label)] || '#64748B'}60`,
                    color: labelColorMap[String(hoveredPoint.label)] || '#E6EAF0',
                  }}
                >
                  Nhãn: {String(hoveredPoint.label)}
                </span>
              )}
            </div>
            <div className="space-y-0.5 text-[11px] text-[#929EAF]">
              <div>
                Cụm:{' '}
                <strong className="text-[#E6EAF0]">
                  {hoveredPoint.cluster === -1 ? 'Điểm nhiễu' : `Cụm ${hoveredPoint.cluster}`}
                </strong>
              </div>
              <div className="font-mono text-[10px] text-[#929EAF]">
                Tọa độ: ({hoveredPoint.x.toFixed(2)}, {hoveredPoint.y.toFixed(2)}, {hoveredPoint.z.toFixed(2)})
              </div>
              {selectedFeature && hoveredPoint.originalFeatures[selectedFeature] !== undefined && (
                <div className="text-[#8B9CFB] pt-0.5 font-mono">
                  {selectedFeature}: {hoveredPoint.originalFeatures[selectedFeature]}
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* 2. DOCKED 3D CONTROLS TOOLBAR (On the right, compact and collapsible) */}
      <div
        className={`shrink-0 h-full border-l border-[#303A49] bg-[#151B23] flex flex-col items-center py-2.5 z-20 transition-all ${
          isToolbarCollapsed ? 'w-9' : 'w-14'
        }`}
      >
        {/* Collapse / Expand Toggle */}
        <button
          type="button"
          onClick={() => setIsToolbarCollapsed(!isToolbarCollapsed)}
          title={isToolbarCollapsed ? 'Mở rộng thanh công cụ 3D' : 'Thu gọn thanh công cụ 3D'}
          className="p-1.5 rounded hover:bg-[#202936] text-[#929EAF] hover:text-[#E6EAF0] mb-2 transition-colors"
        >
          {isToolbarCollapsed ? <ChevronLeft size={14} /> : <ChevronRight size={14} />}
        </button>

        {!isToolbarCollapsed ? (
          <div className="flex flex-col items-center gap-1.5 w-full px-1.5">
            {/* Reset Camera / Fit View */}
            <button
              type="button"
              onClick={handleResetCamera}
              title="Đặt lại góc nhìn mặc định (Fit View / phím R)"
              className="w-full flex flex-col items-center justify-center gap-1 py-1.5 rounded bg-[#202936] hover:bg-[#2a3647] border border-[#303A49] text-[9px] text-[#E6EAF0] transition-colors"
            >
              <RotateCcw size={13} className="text-[#8B9CFB]" />
              <span>Reset</span>
            </button>

            {/* Auto-rotate 360 */}
            <button
              type="button"
              onClick={() => setIsAutoRotating(!isAutoRotating)}
              title={isAutoRotating ? 'Dừng tự động xoay' : 'Bật tự động xoay 360°'}
              className={`w-full flex flex-col items-center justify-center gap-1 py-1.5 rounded border text-[9px] transition-colors ${
                isAutoRotating
                  ? 'bg-[#202936] border-[#8B9CFB] text-[#8B9CFB]'
                  : 'bg-[#202936] hover:bg-[#2a3647] border-[#303A49] text-[#929EAF] hover:text-[#E6EAF0]'
              }`}
            >
              {isAutoRotating ? <Pause size={13} /> : <Play size={13} />}
              <span>{isAutoRotating ? 'Dừng' : 'Tự xoay'}</span>
            </button>

            {/* Zoom In & Out */}
            <div className="flex items-center gap-1 w-full">
              <button
                type="button"
                onClick={() => handleZoom('in')}
                title="Phóng to (+)"
                className="flex-1 flex items-center justify-center py-1.5 rounded bg-[#202936] hover:bg-[#2a3647] border border-[#303A49] text-[#E6EAF0] transition-colors"
              >
                <ZoomIn size={12} />
              </button>
              <button
                type="button"
                onClick={() => handleZoom('out')}
                title="Thu nhỏ (-)"
                className="flex-1 flex items-center justify-center py-1.5 rounded bg-[#202936] hover:bg-[#2a3647] border border-[#303A49] text-[#E6EAF0] transition-colors"
              >
                <ZoomOut size={12} />
              </button>
            </div>

            <div className="w-full h-px bg-[#303A49] my-1" />

            {/* Centroids Toggle */}
            <button
              type="button"
              onClick={() => setCentroidsVisible(!centroidsVisible)}
              title={centroidsVisible ? 'Ẩn tâm cụm' : 'Hiện tâm cụm'}
              className={`w-full flex flex-col items-center justify-center gap-1 py-1.5 rounded border text-[9px] transition-colors ${
                centroidsVisible
                  ? 'bg-[#202936] border-[#fbbf24] text-[#fbbf24]'
                  : 'bg-[#202936] hover:bg-[#2a3647] border-[#303A49] text-[#929EAF]'
              }`}
            >
              <Layers size={13} />
              <span>Tâm cụm</span>
            </button>

            {/* Biplot toggle (PCA only) */}
            {reductionMethod === 'PCA' && (
              <button
                type="button"
                onClick={() => {
                  if (onToggleBiplot) onToggleBiplot();
                  else setBiplotVisible(!biplotVisible);
                }}
                title={biplotVisible ? 'Ẩn véc-tơ PCA Biplot' : 'Hiện véc-tơ PCA Biplot'}
                className={`w-full flex flex-col items-center justify-center gap-1 py-1.5 rounded border text-[9px] transition-colors ${
                  biplotVisible
                    ? 'bg-[#202936] border-[#8B9CFB] text-[#8B9CFB]'
                    : 'bg-[#202936] hover:bg-[#2a3647] border-[#303A49] text-[#929EAF]'
                }`}
              >
                <Compass size={13} />
                <span>Biplot</span>
              </button>
            )}

            {/* Point Scale Control */}
            <div className="w-full flex flex-col items-center gap-1 py-1.5 px-1 rounded bg-[#202936] border border-[#303A49] text-[9px]">
              <span className="text-[#929EAF] font-medium">Cỡ điểm</span>
              <input
                type="range"
                min="0.6"
                max="3.0"
                step="0.2"
                value={pointScale}
                onChange={(e) => setPointScale(parseFloat(e.target.value))}
                className="w-10 h-1 cursor-pointer accent-[#8B9CFB]"
                title={`Kích thước: ${pointScale.toFixed(1)}x`}
              />
              <span className="font-mono text-[#8B9CFB]">{pointScale.toFixed(1)}x</span>
            </div>

            <div className="w-full h-px bg-[#303A49] my-1" />

            {/* Snapshot */}
            <button
              type="button"
              onClick={handleTakeSnapshot}
              title="Chụp ảnh 3D phân giải cao (PNG)"
              className="w-full flex flex-col items-center justify-center gap-1 py-1.5 rounded bg-[#162a23] hover:bg-[#1f3b31] border border-[#065f46] text-[9px] text-[#34d399] transition-colors"
            >
              <Camera size={13} />
              <span>Chụp</span>
            </button>
          </div>
        ) : (
          /* Minimized Strip Icons */
          <div className="flex flex-col items-center gap-2">
            <button
              type="button"
              onClick={handleResetCamera}
              title="Đặt lại góc nhìn"
              className="p-1 rounded text-[#929EAF] hover:text-[#8B9CFB]"
            >
              <RotateCcw size={14} />
            </button>
            <button
              type="button"
              onClick={() => setIsAutoRotating(!isAutoRotating)}
              title="Tự động xoay"
              className={`p-1 rounded ${isAutoRotating ? 'text-[#8B9CFB]' : 'text-[#929EAF]'}`}
            >
              {isAutoRotating ? <Pause size={14} /> : <Play size={14} />}
            </button>
            <button
              type="button"
              onClick={handleTakeSnapshot}
              title="Chụp ảnh 3D"
              className="p-1 rounded text-[#34d399]"
            >
              <Camera size={14} />
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

function createTextLabelSprite(text: string, strokeColor: string = '#8B9CFB'): THREE.Sprite {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 64;
  const ctx = canvas.getContext('2d');
  if (ctx) {
    ctx.fillStyle = 'rgba(21, 27, 35, 0.9)';
    ctx.roundRect(4, 4, 248, 56, 8);
    ctx.fill();
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 2;
    ctx.roundRect(4, 4, 248, 56, 8);
    ctx.stroke();

    ctx.fillStyle = '#E6EAF0';
    ctx.font = 'bold 20px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, 128, 32);
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.minFilter = THREE.LinearFilter;
  const spriteMat = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false });
  const sprite = new THREE.Sprite(spriteMat);
  sprite.scale.set(12, 3, 1);
  return sprite;
}
