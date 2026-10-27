import React, { lazy, Suspense, useState, useEffect, useCallback, useRef, useMemo } from 'react';
import * as THREE from 'three';
import Papa from 'papaparse';
import {
  ParsedDataset,
  DataPoint,
  DataValue,
  ReductionConfig,
  ClusteringConfig,
  ColorMode,
  EvaluationMetrics,
  MIN_NUMERIC_FEATURES,
  ReductionMethod,
  DatasetSummary,
} from './types';
import { parseCSVString, extractFeatureMatrix } from './utils/dataParser';
import { computePCA } from './algorithms/pca';
import { computeTSNE } from './algorithms/tsne';
import { computeUMAP } from './algorithms/umap';
import { computeKMeans } from './algorithms/kmeans';
import { computeDBSCAN } from './algorithms/dbscan';
import { computeSilhouetteScore, computeARI } from './algorithms/metrics';
import { initLiveTSNE, stepLiveTSNE, LiveTSNEState } from './algorithms/liveOptimization';
import { ControlSidebar } from './components/ControlSidebar';
import type { SidebarTab } from './components/ControlSidebar';
import { MetricsPanel } from './components/MetricsPanel';
import { DataInspector } from './components/DataInspector';
import { RawDataModal } from './components/RawDataModal';
import { DatasetManagerModal } from './components/DatasetManagerModal';
import { ExperimentRecord } from './types';
import {
  getErrorMessage,
  isComputeResult,
  isDatasetDocument,
  isDatasetSummary,
  isExperimentRecord,
  isRecord,
} from './utils/typeGuards';

const ThreeCanvas = lazy(() =>
  import('./components/ThreeCanvas').then((module) => ({ default: module.ThreeCanvas }))
);
const ExperimentsModal = lazy(() =>
  import('./components/ExperimentsModal').then((module) => ({ default: module.ExperimentsModal }))
);
const ElbowModal = lazy(() =>
  import('./components/ElbowModal').then((module) => ({ default: module.ElbowModal }))
);
const SecurityModal = lazy(() =>
  import('./components/SecurityModal').then((module) => ({ default: module.SecurityModal }))
);

type ReductionOutput = {
  coordinates: [number, number, number][];
  components?: number[][];
  explainedVariance?: [number, number, number];
  totalExplainedVariance?: number;
};

export function App() {
  const [dataset, setDataset] = useState<ParsedDataset | null>(null);
  const [activeArea, setActiveArea] = useState<SidebarTab>('dataset');
  const [selectedFeatures, setSelectedFeatures] = useState<string[]>([]);
  const [dataPoints, setDataPoints] = useState<DataPoint[]>([]);
  const rawPreviewPoints = useMemo<DataPoint[]>(() => {
    if (!dataset || selectedFeatures.length < MIN_NUMERIC_FEATURES) return [];

    const axes = selectedFeatures.slice(0, 3);
    const getAxisValue = (row: ParsedDataset['rawRows'][number], feature: string | undefined): number => {
      if (!feature) return 0;
      const value = row[feature];
      if (typeof value === 'number' && Number.isFinite(value)) return value;
      if (typeof value === 'string' && value.trim() !== '' && Number.isFinite(Number(value))) {
        return Number(value);
      }
      return 0;
    };

    return dataset.rawRows.map((row, index) => {
      const originalFeatures: Record<string, number> = {};
      for (const feature of dataset.featureColumns) {
        originalFeatures[feature] = getAxisValue(row, feature);
      }

      return {
        id: (dataset.idColumn ? row[dataset.idColumn] : null) ?? index + 1,
        label: dataset.labelColumn ? row[dataset.labelColumn] : undefined,
        originalFeatures,
        x: getAxisValue(row, axes[0]),
        y: getAxisValue(row, axes[1]),
        z: getAxisValue(row, axes[2]),
        cluster: 0,
      };
    });
  }, [dataset, selectedFeatures]);
  const [selectedPoint, setSelectedPoint] = useState<DataPoint | null>(null);
  const [isComputing, setIsComputing] = useState<boolean>(false);
  const [loadingMessage, setLoadingMessage] = useState<string>('');
  const [dbStatus, setDbStatus] = useState<string>('Đang kiểm tra...');
  const [notice, setNotice] = useState<{ type: 'success' | 'error' | 'info'; message: string } | null>(null);

  // Dataset Management & Theme State
  const [datasetList, setDatasetList] = useState<DatasetSummary[]>([]);
  const [activeDatasetId, setActiveDatasetId] = useState<string>('cancer_wdbc');
  const [isRawDataModalOpen, setIsRawDataModalOpen] = useState<boolean>(false);
  const [isDatasetManagerOpen, setIsDatasetManagerOpen] = useState<boolean>(false);

  const [theme, setTheme] = useState<'dark' | 'light'>(() => {
    try {
      const saved = localStorage.getItem('theme');
      return saved === 'light' ? 'light' : 'dark';
    } catch {
      return 'dark';
    }
  });

  useEffect(() => {
    if (theme === 'light') {
      document.body.classList.add('light-theme');
    } else {
      document.body.classList.remove('light-theme');
    }
    try {
      localStorage.setItem('theme', theme);
    } catch {
      // ignore
    }
  }, [theme]);

  const handleToggleTheme = () => {
    setTheme((prev) => (prev === 'light' ? 'dark' : 'light'));
  };

  // Elbow Modal, Security Modal & PCA Biplot State
  const [isElbowModalOpen, setIsElbowModalOpen] = useState<boolean>(false);
  const [isSecurityModalOpen, setIsSecurityModalOpen] = useState<boolean>(false);
  const [pcaComponents, setPcaComponents] = useState<number[][] | undefined>(undefined);
  const [featureMatrix, setFeatureMatrix] = useState<number[][]>([]);
  const [biplotVisible, setBiplotVisible] = useState<boolean>(false);
  const [biplotTopK, setBiplotTopK] = useState<number>(6);
  const [biplotScale, setBiplotScale] = useState<number>(30);

  // Dual Viewport State
  const [isDualView, setIsDualView] = useState<boolean>(false);
  const [dualSecondaryMethod, setDualSecondaryMethod] = useState<ReductionMethod>('t-SNE');
  const [dualDataPoints, setDualDataPoints] = useState<DataPoint[]>([]);

  // Throttled Camera Sync to prevent destroying Viewport B during orbit
  const [syncedCamPos, setSyncedCamPos] = useState<THREE.Vector3 | null>(null);
  const [syncedTargetPos, setSyncedTargetPos] = useState<THREE.Vector3 | null>(null);
  const syncRafRef = useRef<number | null>(null);

  // Live Optimization State
  const [isLiveAnimating, setIsLiveAnimating] = useState<boolean>(false);
  const [liveCurrentIter, setLiveCurrentIter] = useState<number>(0);
  const liveStateRef = useRef<LiveTSNEState | null>(null);
  const animFrameRef = useRef<number | null>(null);

  // Experiments & MongoDB Database state
  const [isExperimentsModalOpen, setIsExperimentsModalOpen] = useState(false);
  const [experiments, setExperiments] = useState<ExperimentRecord[]>([]);
  const [isSavingExperiment, setIsSavingExperiment] = useState(false);

  const [reductionConfig, setReductionConfig] = useState<ReductionConfig>({
    method: 'PCA',
    normalization: 'z-score',
    perplexity: 30,
    learningRate: 100,
    iterations: 250,
    nNeighbors: 15,
    minDist: 0.1,
  });

  const [clusteringConfig, setClusteringConfig] = useState<ClusteringConfig>({
    method: 'K-Means',
    k: 2,
    epsilon: 3.5,
    minPts: 5,
  });

  const [colorMode, setColorMode] = useState<ColorMode>('ground-truth');
  const [selectedColorFeature, setSelectedColorFeature] = useState<string>('');

  const [metrics, setMetrics] = useState<EvaluationMetrics>({
    silhouetteScore: null,
    adjustedRandIndex: null,
    clusterCounts: {},
  });

  useEffect(() => {
    if (!notice) return;
    const timeout = window.setTimeout(() => setNotice(null), 7000);
    return () => window.clearTimeout(timeout);
  }, [notice]);

  // Check Database connection health periodically
  useEffect(() => {
    const checkDbHealth = async () => {
      try {
        const res = await fetch('/api/health');
        if (res.ok) {
          const data: unknown = await res.json();
          setDbStatus(isRecord(data) && typeof data.database === 'string' ? data.database : 'Online');
        }
      } catch {
        setDbStatus('Trực tiếp (Client-Side)');
      }
    };
    checkDbHealth();
    const interval = setInterval(checkDbHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  // Reset all computation & analytical states to ensure 100% data consistency
  const resetComputationState = useCallback(() => {
    setDataPoints([]);
    setDualDataPoints([]);
    setSelectedPoint(null);
    setPcaComponents(undefined);
    setFeatureMatrix([]);
    setMetrics({
      silhouetteScore: null,
      adjustedRandIndex: null,
      clusterCounts: {},
    });
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
      animFrameRef.current = null;
    }
    setIsLiveAnimating(false);
    liveStateRef.current = null;
    setLiveCurrentIter(0);
  }, []);

  // Fetch all datasets from backend
  const fetchDatasetList = useCallback(async () => {
    try {
      const res = await fetch('/api/datasets');
      if (res.ok) {
        const list: unknown = await res.json();
        if (Array.isArray(list)) {
          setDatasetList(list.filter(isDatasetSummary));
        }
      }
    } catch (e: unknown) {
      console.warn('Không thể nạp danh sách bộ dữ liệu từ server:', e);
    }
  }, []);

  // Select and switch active dataset
  const handleSelectDataset = useCallback(async (id: string) => {
    try {
      setIsComputing(true);
      setLoadingMessage('Đang nạp bộ dữ liệu...');
      resetComputationState();

      // 1. Try fetching dataset by ID from backend
      try {
        const res = await fetch(`/api/datasets/${id}`);
        if (res.ok) {
          const data: unknown = await res.json();
          if (isDatasetDocument(data)) {
            const parsed: ParsedDataset = {
              id: data.id,
              name: data.name,
              totalRows: data.totalRows,
              headers: Object.keys(data.rawRows[0] || {}),
              idColumn: data.idColumn,
              labelColumn: data.labelColumn,
              featureColumns: data.features,
              rawRows: data.rawRows,
            };
            setDataset(parsed);
            setActiveDatasetId(id);
            setSelectedFeatures(parsed.featureColumns);
            if (parsed.featureColumns.length > 0) {
              setSelectedColorFeature(parsed.featureColumns[0]);
            }
            setNotice({ type: 'success', message: `Đã chọn bộ dữ liệu "${parsed.name}".` });
            return;
          }
        }
      } catch (apiError: unknown) {
        console.warn(`Không thể nạp dataset ${id} từ API, thử fallback:`, apiError);
      }

      // 2. Fallbacks for standard datasets if server offline
      if (id.includes('cancer')) {
        const staticRes = await fetch('/data.csv');
        if (!staticRes.ok) throw new Error('Không thể tải bộ dữ liệu mẫu /data.csv.');
        const text = await staticRes.text();
        const parsed = parseCSVString(text, 'Wisconsin Breast Cancer');
        setDataset(parsed);
        setActiveDatasetId('cancer_wdbc');
        setSelectedFeatures(parsed.featureColumns);
        if (parsed.featureColumns.length > 0) {
          setSelectedColorFeature(parsed.featureColumns[0]);
        }
      } else if (id.includes('wholesale')) {
        const staticRes = await fetch('/wholesale.csv');
        if (!staticRes.ok) throw new Error('Không thể tải bộ dữ liệu mẫu /wholesale.csv.');
        const text = await staticRes.text();
        const parsed = parseCSVString(text, 'Wholesale Customers');
        setDataset(parsed);
        setActiveDatasetId('wholesale_customers');
        setSelectedFeatures(parsed.featureColumns);
        if (parsed.featureColumns.length > 0) {
          setSelectedColorFeature(parsed.featureColumns[0]);
        }
      }
    } catch (err: unknown) {
      console.error('Không thể nạp bộ dữ liệu:', err);
      setNotice({ type: 'error', message: `Không thể nạp dữ liệu: ${getErrorMessage(err)}` });
    } finally {
      setIsComputing(false);
      setLoadingMessage('');
    }
  }, [resetComputationState]);

  // Rename a dataset
  const handleRenameDataset = async (id: string, newName: string) => {
    try {
      const res = await fetch(`/api/datasets/${id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: newName }),
      });
      if (!res.ok) throw new Error('Không thể đổi tên bộ dữ liệu.');
      await fetchDatasetList();
      if (dataset && (activeDatasetId === id || dataset.name === newName)) {
        setDataset((prev) => (prev ? { ...prev, name: newName } : null));
      }
      setNotice({ type: 'success', message: `Đã đổi tên bộ dữ liệu thành "${newName}".` });
    } catch (err: unknown) {
      setNotice({ type: 'error', message: `Lỗi đổi tên: ${getErrorMessage(err)}` });
    }
  };

  // Delete an uploaded dataset
  const handleDeleteDataset = async (id: string) => {
    try {
      const res = await fetch(`/api/datasets/${id}`, {
        method: 'DELETE',
      });
      if (!res.ok) {
        const data: unknown = await res.json().catch(() => null);
        const errMsg = isRecord(data) && typeof data.error === 'string' ? data.error : 'Không thể xóa bộ dữ liệu';
        throw new Error(errMsg);
      }
      await fetchDatasetList();
      setNotice({ type: 'success', message: 'Đã xóa bộ dữ liệu thành công.' });

      // If active dataset was deleted, switch back to cancer_wdbc
      if (activeDatasetId === id) {
        await handleSelectDataset('cancer_wdbc');
      }
    } catch (err: unknown) {
      setNotice({ type: 'error', message: `Lỗi xóa: ${getErrorMessage(err)}` });
    }
  };

  // Initial Load: Fetch list and Wisconsin Breast Cancer
  const loadInitialDataset = useCallback(async () => {
    await fetchDatasetList();
    await handleSelectDataset('cancer_wdbc');
  }, [fetchDatasetList, handleSelectDataset]);

  useEffect(() => {
    loadInitialDataset();
  }, [loadInitialDataset]);

  // Upload file directly to Backend MongoDB and switch to it
  const handleFileUpload = async (file: File) => {
    try {
      setIsComputing(true);
      setLoadingMessage('Đang kiểm tra cấu trúc CSV...');
      resetComputationState();

      const parsed = parseCSVString(await file.text(), file.name);

      const formData = new FormData();
      formData.append('file', file);

      let res: Response;
      try {
        setLoadingMessage('Đang tải CSV lên máy chủ...');
        res = await fetch('/api/datasets/upload', {
          method: 'POST',
          body: formData,
        });
      } catch (networkError: unknown) {
        console.warn('Backend không kết nối được; dùng CSV đã kiểm tra trong trình duyệt:', networkError);
        setDataset(parsed);
        setSelectedFeatures(parsed.featureColumns);
        setSelectedColorFeature(parsed.featureColumns[0] ?? '');
        setActiveDatasetId('custom_client');
        setNotice({
          type: 'info',
          message: 'CSV hợp lệ và đã được mở trong trình duyệt. Máy chủ không kết nối được nên tệp chưa được lưu vào CSDL.',
        });
        return;
      }

      if (!res.ok) {
        const errorBody: unknown = await res.json().catch(() => null);
        const message = isRecord(errorBody) && typeof errorBody.error === 'string'
          ? errorBody.error
          : `Máy chủ từ chối tệp CSV (HTTP ${res.status}).`;
        throw new Error(message);
      }

      const data: unknown = await res.json();
      if (!isDatasetDocument(data)) {
        throw new Error('Máy chủ trả về dữ liệu CSV không đúng cấu trúc.');
      }

      setDataset({
        id: data.id,
        name: data.name,
        totalRows: data.totalRows,
        headers: Object.keys(data.rawRows[0] || {}),
        idColumn: data.idColumn,
        labelColumn: data.labelColumn,
        featureColumns: data.features,
        rawRows: data.rawRows,
      });
      setActiveDatasetId(data.id);
      setSelectedFeatures(data.features);
      setSelectedColorFeature(data.features[0] ?? '');
      await fetchDatasetList();
      setNotice({ type: 'success', message: `Đã tải và lưu bộ dữ liệu "${data.name}" (${data.totalRows} dòng).` });
    } catch (err: unknown) {
      setNotice({ type: 'error', message: `Lỗi tải file: ${getErrorMessage(err)}` });
    } finally {
      setIsComputing(false);
      setLoadingMessage('');
    }
  };

  const handleLoadPreloaded = async (type: 'cancer' | 'wholesale' | 'synthetic') => {
    if (type === 'cancer') {
      await handleSelectDataset('cancer_wdbc');
    } else if (type === 'wholesale') {
      await handleSelectDataset('wholesale_customers');
    }
  };

  const handleToggleFeature = (col: string) => {
    setSelectedFeatures((prev) =>
      prev.includes(col) ? prev.filter((f) => f !== col) : [...prev, col]
    );
  };

  const handleSelectAllFeatures = () => {
    if (dataset) setSelectedFeatures(dataset.featureColumns);
  };

  const handleDeselectAllFeatures = () => {
    setSelectedFeatures([]);
  };

  const computeCoordinates = (matrix: number[][], method: ReductionMethod): ReductionOutput => {
    if (method === 'PCA') {
      return computePCA(matrix);
    } else if (method === 't-SNE') {
      const coords = computeTSNE(matrix, {
        perplexity: reductionConfig.perplexity,
        learningRate: reductionConfig.learningRate,
        iterations: reductionConfig.iterations,
      });
      return { coordinates: coords };
    } else {
      const coords = computeUMAP(matrix, {
        nNeighbors: reductionConfig.nNeighbors,
        minDist: reductionConfig.minDist,
      });
      return { coordinates: coords };
    }
  };

  // 3. Pipeline Calculation: Can execute via Backend API or locally
  const runPipeline = useCallback(async () => {
    if (!dataset || selectedFeatures.length < MIN_NUMERIC_FEATURES) return;

    setActiveArea('reduction');
    setIsComputing(true);
    setLoadingMessage(`Đang giảm chiều dữ liệu bằng ${reductionConfig.method}...`);

    const startTime = performance.now();
    await new Promise((r) => setTimeout(r, 40));

    try {
      const matrix = extractFeatureMatrix(
        dataset.rawRows,
        selectedFeatures,
        reductionConfig.normalization
      );
      setFeatureMatrix(matrix);

      const groundTruth: DataValue[] = dataset.labelColumn
        ? dataset.rawRows.map((r) => r[dataset.labelColumn!] ?? null)
        : [];

      let coordinates: [number, number, number][] = [];
      let clusterAssignments: number[] = [];
      let expVar: [number, number, number] | undefined = undefined;
      let totalExpVar: number | undefined = undefined;
      let wasFromBackend = false;

      // 1. Try Backend Compute API first (with SHA-256 Multi-Tier Caching & Gzip)
      try {
        const computeRes = await fetch('/api/compute', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            matrix,
            groundTruth,
            reductionConfig,
            clusteringConfig: { ...clusteringConfig, method: 'None' },
          }),
        });

        if (!computeRes.ok) {
          throw new Error(`API tính toán trả về lỗi HTTP ${computeRes.status}.`);
        }
        const computeData: unknown = await computeRes.json();
        if (!isComputeResult(computeData)) {
          throw new Error('Máy chủ trả về kết quả tính toán không hợp lệ.');
        }
          coordinates = computeData.coordinates;
          expVar = computeData.explainedVariance;
          totalExpVar = computeData.totalExplainedVariance;
          clusterAssignments = new Array(coordinates.length).fill(0);
          if (computeData.components) {
            setPcaComponents(computeData.components);
          } else {
            setPcaComponents(undefined);
          }
          wasFromBackend = true;
      } catch {
        // Backend API offline on static cloud -> compute locally seamlessly
      }

      // 2. Client-Side Fallback if Backend is unreachable
      if (!wasFromBackend) {
        const primaryRes = computeCoordinates(matrix, reductionConfig.method);
        coordinates = primaryRes.coordinates;

        if (reductionConfig.method === 'PCA' && primaryRes.components) {
          setPcaComponents(primaryRes.components);
        } else {
          setPcaComponents(undefined);
        }

        clusterAssignments = new Array(coordinates.length).fill(0);

        expVar = primaryRes.explainedVariance;
        totalExpVar = primaryRes.totalExplainedVariance;
      }

      const durationMs = Math.round(performance.now() - startTime);

      setMetrics({
        silhouetteScore: null,
        adjustedRandIndex: null,
        explainedVariance: expVar,
        totalVarianceExplained: totalExpVar,
        clusterCounts: {},
        noiseCount: 0,
        executionTimeMs: durationMs,
      });

      // Assemble Primary DataPoints
      const points: DataPoint[] = dataset.rawRows.map((row, idx) => {
        const id = (dataset.idColumn ? row[dataset.idColumn] : null) ?? idx + 1;
        const label = dataset.labelColumn ? row[dataset.labelColumn] : undefined;

        const originalFeatures: Record<string, number> = {};
        for (const col of dataset.featureColumns) {
          originalFeatures[col] = Number(row[col]);
        }

        const coord = coordinates[idx] || [0, 0, 0];

        return {
          id,
          originalFeatures,
          label,
          x: coord[0],
          y: coord[1],
          z: coord[2],
          cluster: clusterAssignments[idx] ?? 0,
        };
      });

      setDataPoints(points);

      // ALWAYS prepare Dual DataPoints using the secondary method
      const secMethod = dualSecondaryMethod === reductionConfig.method ? (reductionConfig.method === 'PCA' ? 't-SNE' : 'PCA') : dualSecondaryMethod;
      const secondaryRes = computeCoordinates(matrix, secMethod);
      const secCoords = secondaryRes.coordinates;

      const dualPoints: DataPoint[] = dataset.rawRows.map((row, idx) => {
        const id = (dataset.idColumn ? row[dataset.idColumn] : null) ?? idx + 1;
        const label = dataset.labelColumn ? row[dataset.labelColumn] : undefined;

        const originalFeatures: Record<string, number> = {};
        for (const col of dataset.featureColumns) {
          originalFeatures[col] = Number(row[col]);
        }

        const coord = secCoords[idx] || [0, 0, 0];

        return {
          id,
          originalFeatures,
          label,
          x: coord[0],
          y: coord[1],
          z: coord[2],
          cluster: clusterAssignments[idx] ?? 0,
        };
      });

      setDualDataPoints(dualPoints);
      setNotice({
        type: 'success',
        message: `Đã giảm chiều 3D thành công bằng ${reductionConfig.method} (${durationMs} ms)!`,
      });
    } catch (err: unknown) {
      console.error(err);
      setNotice({ type: 'error', message: `Lỗi tính toán: ${getErrorMessage(err)}` });
    } finally {
      setIsComputing(false);
      setLoadingMessage('');
    }
  }, [dataset, selectedFeatures, reductionConfig, clusteringConfig, dualSecondaryMethod]);

  useEffect(() => {
    setDataPoints([]);
    setDualDataPoints([]);
    setSelectedPoint(null);
    setPcaComponents(undefined);
    setFeatureMatrix([]);
    setMetrics({
      silhouetteScore: null,
      adjustedRandIndex: null,
      clusterCounts: {},
    });
  }, [dataset, selectedFeatures, reductionConfig]);

  const handleApplyClustering = (config: ClusteringConfig = clusteringConfig) => {
    if (dataPoints.length === 0 || isComputing) return;

    const startTime = performance.now();
    const coordinates = dataPoints.map((point) => [point.x, point.y, point.z] as [number, number, number]);
    let clusterAssignments: number[];
    let noiseCount = 0;

    if (config.method === 'K-Means') {
      clusterAssignments = computeKMeans(coordinates, config.k).clusters;
    } else if (config.method === 'DBSCAN') {
      const result = computeDBSCAN(coordinates, config.epsilon, config.minPts);
      clusterAssignments = result.clusters;
      noiseCount = result.noiseCount;
    } else {
      clusterAssignments = new Array(coordinates.length).fill(0);
    }

    const clusterCounts: Record<number, number> = {};
    for (const cluster of clusterAssignments) {
      if (cluster >= 0) clusterCounts[cluster] = (clusterCounts[cluster] || 0) + 1;
    }

    const groundTruth: DataValue[] = dataset?.labelColumn && dataset.rawRows.length === clusterAssignments.length
      ? dataset.rawRows.map((row) => row[dataset.labelColumn!])
      : [];
    const silhouetteScore = computeSilhouetteScore(coordinates, clusterAssignments);
    const adjustedRandIndex = groundTruth.length > 0
      ? computeARI(groundTruth, clusterAssignments)
      : null;

    setActiveArea('clustering');
    const durationMs = Math.round(performance.now() - startTime);

    setDataPoints((points) => points.map((point, index) => ({
      ...point,
      cluster: clusterAssignments[index] ?? 0,
    })));
    setDualDataPoints((points) => points.map((point, index) => ({
      ...point,
      cluster: clusterAssignments[index] ?? 0,
    })));
    setMetrics((current) => ({
      ...current,
      silhouetteScore,
      adjustedRandIndex,
      clusterCounts,
      noiseCount,
      executionTimeMs: (current.executionTimeMs || 0) + durationMs,
    }));
    setNotice({
      type: 'success',
      message: `Đã áp dụng phân cụm ${config.method} thành công (${durationMs} ms)!`,
    });
  };
  const isRawDataArea = activeArea === 'dataset' && dataPoints.length === 0;
  const displayedPoints = isRawDataArea ? rawPreviewPoints : (dataPoints.length > 0 ? dataPoints : rawPreviewPoints);

  // 4. LIVE OPTIMIZATION ANIMATION LOOP FOR t-SNE / UMAP
  const handleToggleLiveAnimation = () => {
    if (isLiveAnimating) {
      setIsLiveAnimating(false);
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    } else {
      if (!dataset || selectedFeatures.length < MIN_NUMERIC_FEATURES) return;

      if (!liveStateRef.current || liveStateRef.current.isCompleted) {
        const matrix = extractFeatureMatrix(
          dataset.rawRows,
          selectedFeatures,
          reductionConfig.normalization
        );
        liveStateRef.current = initLiveTSNE(
          matrix,
          reductionConfig.perplexity,
          reductionConfig.learningRate,
          reductionConfig.iterations
        );
      }

      setIsLiveAnimating(true);
      runLiveLoop();
    }
  };

  const handleResetLiveAnimation = () => {
    setIsLiveAnimating(false);
    if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);

    if (dataset && selectedFeatures.length >= MIN_NUMERIC_FEATURES) {
      const matrix = extractFeatureMatrix(
        dataset.rawRows,
        selectedFeatures,
        reductionConfig.normalization
      );
      const state = initLiveTSNE(
        matrix,
        reductionConfig.perplexity,
        reductionConfig.learningRate,
        reductionConfig.iterations
      );
      liveStateRef.current = state;
      setLiveCurrentIter(0);

      setDataPoints((prev) =>
        prev.map((pt, idx) => ({
          ...pt,
          x: state.Y[idx * 3] || 0,
          y: state.Y[idx * 3 + 1] || 0,
          z: state.Y[idx * 3 + 2] || 0,
        }))
      );
    }
  };

  const runLiveLoop = () => {
    const state = liveStateRef.current;
    if (!state || state.isCompleted) {
      setIsLiveAnimating(false);
      return;
    }

    const newCoords = stepLiveTSNE(state, 4);
    setLiveCurrentIter(state.currentIter);

    setDataPoints((prev) =>
      prev.map((pt, idx) => {
        const c = newCoords[idx] || [0, 0, 0];
        return {
          ...pt,
          x: c[0],
          y: c[1],
          z: c[2],
        };
      })
    );

    if (!state.isCompleted) {
      animFrameRef.current = requestAnimationFrame(runLiveLoop);
    } else {
      setIsLiveAnimating(false);
    }
  };

  // 5. Throttled Dual Viewport Camera Synchronization
  const handlePrimaryCameraChange = (camPos: THREE.Vector3, targetPos: THREE.Vector3) => {
    if (isDualView) {
      if (syncRafRef.current) return;
      syncRafRef.current = requestAnimationFrame(() => {
        setSyncedCamPos(camPos.clone());
        setSyncedTargetPos(targetPos.clone());
        syncRafRef.current = null;
      });
    }
  };

  // 6. Save Experiment to MongoDB
  const handleSaveExperiment = async () => {
    if (!dataset) return;
    try {
      setIsSavingExperiment(true);
      const res = await fetch('/api/experiments', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          datasetId: dataset.name,
          datasetName: dataset.name,
          reductionMethod: reductionConfig.method,
          clusteringMethod: clusteringConfig.method,
          k: clusteringConfig.k,
          silhouetteScore: metrics.silhouetteScore,
          ari: metrics.adjustedRandIndex,
          params: {
            reduction: reductionConfig,
            clustering: clusteringConfig,
          },
        }),
      });
      if (!res.ok) {
        throw new Error(`Máy chủ từ chối lưu thí nghiệm (HTTP ${res.status}).`);
      }
      setNotice({ type: 'success', message: 'Đã lưu kết quả thí nghiệm thành công.' });
      await fetchExperiments();
    } catch (e: unknown) {
      setNotice({ type: 'error', message: `Không thể lưu thí nghiệm: ${getErrorMessage(e)}` });
    } finally {
      setIsSavingExperiment(false);
    }
  };

  const fetchExperiments = async () => {
    try {
      const res = await fetch('/api/experiments');
      if (!res.ok) {
        throw new Error(`Không thể tải lịch sử (HTTP ${res.status}).`);
      }
      const data: unknown = await res.json();
      if (!Array.isArray(data) || !data.every(isExperimentRecord)) {
        throw new Error('Dữ liệu lịch sử từ máy chủ không đúng định dạng.');
      }
      setExperiments(data);
    } catch (e: unknown) {
      console.error('Không thể tải lịch sử thí nghiệm:', e);
      setNotice({ type: 'error', message: `Không thể tải lịch sử: ${getErrorMessage(e)}` });
    }
  };

  const handleOpenHistory = () => {
    fetchExperiments();
    setIsExperimentsModalOpen(true);
  };

  // 7. EXPORT ENRICHED CSV WITH 3D COORDINATES & PREDICTED CLUSTERS
  const handleExportCSV = () => {
    if (!dataset || dataPoints.length === 0) return;

    try {
      const rowsToExport = dataPoints.map((pt) => {
        return {
          id: pt.id,
          label_ground_truth: pt.label ?? '',
          predicted_cluster: pt.cluster === -1 ? 'Noise' : pt.cluster,
          coord_x_3d: Number(pt.x.toFixed(4)),
          coord_y_3d: Number(pt.y.toFixed(4)),
          coord_z_3d: Number(pt.z.toFixed(4)),
          ...pt.originalFeatures,
        };
      });

      const csvContent = Papa.unparse(rowsToExport);
      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${dataset.name.replace(/\s+/g, '_')}_3D_Clustered_Results.csv`;
      link.click();
      URL.revokeObjectURL(url);
    } catch (e: unknown) {
      setNotice({ type: 'error', message: `Lỗi xuất file CSV: ${getErrorMessage(e)}` });
    }
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden font-sans select-none bg-[#0D1117] text-[#E6EAF0]">
      {notice && (
        <div
          role={notice.type === 'error' ? 'alert' : 'status'}
          aria-live={notice.type === 'error' ? 'assertive' : 'polite'}
          className={`absolute top-3 right-3 z-[70] flex max-w-md items-center gap-2.5 rounded border px-3 py-2 text-xs shadow-lg backdrop-blur-md ${
            notice.type === 'error'
              ? 'border-[#7f1d1d] bg-[#2d1b1f] text-[#f87171]'
              : notice.type === 'success'
                ? 'border-[#065f46] bg-[#162a23] text-[#34d399]'
                : 'border-[#303A49] bg-[#151B23] text-[#E6EAF0]'
          }`}
        >
          <span className="flex-1">{notice.message}</span>
          <button
            type="button"
            onClick={() => setNotice(null)}
            className="shrink-0 font-bold px-1 text-current/70 hover:text-current"
            aria-label="Đóng thông báo"
          >
            ×
          </button>
        </div>
      )}
      {/* 1. Control Sidebar */}
      <ControlSidebar
        datasetName={dataset ? dataset.name : 'Đang tải...'}
        totalRows={dataset ? dataset.totalRows : 0}
        rawRows={dataset?.rawRows ?? []}
        headers={dataset?.headers ?? []}
        featureColumns={dataset ? dataset.featureColumns : []}
        selectedFeatures={selectedFeatures}
        onToggleFeature={handleToggleFeature}
        onSelectAllFeatures={handleSelectAllFeatures}
        onDeselectAllFeatures={handleDeselectAllFeatures}
        reductionConfig={reductionConfig}
        onChangeReductionConfig={setReductionConfig}
        clusteringConfig={clusteringConfig}
        onChangeClusteringConfig={setClusteringConfig}
        colorMode={colorMode}
        onChangeColorMode={setColorMode}
        selectedColorFeature={selectedColorFeature}
        onChangeSelectedColorFeature={setSelectedColorFeature}
        onFileUpload={handleFileUpload}
        onLoadPreloadedDataset={handleLoadPreloaded}
        isComputing={isComputing}
        hasResults={dataPoints.length > 0}
        onActiveTabChange={setActiveArea}
        onRunComputation={runPipeline}
        onApplyClustering={handleApplyClustering}
        isDualView={isDualView}
        onToggleDualView={() => setIsDualView(!isDualView)}
        dualSecondaryMethod={dualSecondaryMethod}
        onChangeDualSecondaryMethod={(m) => {
          setDualSecondaryMethod(m);
        }}
        isLiveAnimating={isLiveAnimating}
        onToggleLiveAnimation={handleToggleLiveAnimation}
        onResetLiveAnimation={handleResetLiveAnimation}
        liveCurrentIter={liveCurrentIter}
        liveMaxIter={reductionConfig.iterations}
        onExportCSV={handleExportCSV}
        dbStatus={dbStatus}
        onOpenElbowModal={() => setIsElbowModalOpen(true)}
        onOpenSecurityModal={() => setIsSecurityModalOpen(true)}
        pcaComponents={pcaComponents}
        explainedVariance={metrics.explainedVariance}
        totalVarianceExplained={metrics.totalVarianceExplained}
        biplotVisible={biplotVisible}
        onToggleBiplot={() => setBiplotVisible(!biplotVisible)}
        biplotTopK={biplotTopK}
        onChangeBiplotTopK={setBiplotTopK}
        biplotScale={biplotScale}
        onChangeBiplotScale={setBiplotScale}
        datasetList={datasetList}
        activeDatasetId={activeDatasetId}
        onSelectDataset={handleSelectDataset}
        onRenameDataset={handleRenameDataset}
        onDeleteDataset={handleDeleteDataset}
        onOpenRawDataModal={() => setIsRawDataModalOpen(true)}
        onOpenDatasetManager={() => setIsDatasetManagerOpen(true)}
        theme={theme}
        onToggleTheme={handleToggleTheme}
      />

      {/* 2. Main 3D Viewport Area */}
      <div className="flex-1 min-w-0 relative flex flex-col h-full overflow-hidden">
        {/* Single vs Dual Viewport Rendering */}
        <div className="min-h-0 flex-1 w-full flex">
          {/* Viewport 1 (Primary) */}
          <div className="flex-1 h-full relative">
            {displayedPoints.length > 0 ? (
              <Suspense
                fallback={
                  <div className="absolute inset-0 flex items-center justify-center text-xs text-[#929EAF]" role="status">
                    Đang khởi tạo Three.js Engine...
                  </div>
                }
              >
                <ThreeCanvas
                  dataPoints={displayedPoints}
                  colorMode={colorMode}
                  selectedFeature={selectedColorFeature}
                  onSelectPoint={setSelectedPoint}
                  selectedPoint={selectedPoint}
                  reductionMethod={isRawDataArea ? 'RAW' : reductionConfig.method}
                  onCameraChange={handlePrimaryCameraChange}
                  titleBadge={isRawDataArea ? 'DỮ LIỆU GỐC (3D TRỤC ĐẶC TRƯNG)' : isDualView ? `MÀN HÌNH A: ${reductionConfig.method} 3D` : undefined}
                  axisFeatures={isRawDataArea ? selectedFeatures.slice(0, 3) : undefined}
                  pcaComponents={isRawDataArea ? undefined : pcaComponents}
                  featureNames={selectedFeatures}
                  showBiplot={biplotVisible}
                  onToggleBiplot={() => setBiplotVisible(!biplotVisible)}
                  biplotTopK={biplotTopK}
                  biplotScale={biplotScale}
                  theme={theme}
                  onToggleTheme={handleToggleTheme}
                />
              </Suspense>
            ) : (
              <div className="absolute inset-0 flex items-center justify-center px-6 text-center text-xs text-[#929EAF]">
                {dataset
                  ? 'Chọn ít nhất 3 đặc trưng số để trực quan hóa dữ liệu trên biểu đồ 3D.'
                  : 'Vui lòng chọn bộ dữ liệu để bắt đầu phân tích.'}
              </div>
            )}
          </div>

          {/* Viewport 2 (Dual Secondary with Camera Synchronization) */}
          {isDualView && !isRawDataArea && (
            <div className="flex-1 h-full relative border-l border-[#303A49]">
              <Suspense
                fallback={
                  <div className="absolute inset-0 flex items-center justify-center text-xs text-[#929EAF]" role="status">
                    Đang tải bộ máy đồ họa 3D...
                  </div>
                }
              >
                <ThreeCanvas
                  dataPoints={dualDataPoints.length > 0 ? dualDataPoints : dataPoints}
                  colorMode={colorMode}
                  selectedFeature={selectedColorFeature}
                  onSelectPoint={setSelectedPoint}
                  selectedPoint={selectedPoint}
                  reductionMethod={dualSecondaryMethod}
                  syncCameraPos={syncedCamPos}
                  syncTargetPos={syncedTargetPos}
                  titleBadge={`MÀN HÌNH B: ${dualSecondaryMethod} 3D (Đồng bộ)`}
                  theme={theme}
                  onToggleTheme={handleToggleTheme}
                />
              </Suspense>
            </div>
          )}
        </div>

        {/* Loading Overlay */}
        {isComputing && (
          <div className="absolute inset-0 z-50 bg-[#0D1117]/80 backdrop-blur-xs flex flex-col items-center justify-center gap-3">
            <div className="w-8 h-8 rounded-full border-2 border-[#303A49] border-t-[#8B9CFB] animate-spin" />
            <div className="text-xs font-semibold tracking-wide text-[#8B9CFB]">
              {loadingMessage || 'Đang xử lý tính toán...'}
            </div>
          </div>
        )}

        {/* Selected Data Point Inspector */}
        <DataInspector
          point={selectedPoint}
          onClose={() => setSelectedPoint(null)}
          theme={theme}
        />

        {/* Docked Results Panel at Bottom */}
        <div className="shrink-0 border-t border-[#303A49] bg-[#151B23]">
          <MetricsPanel
            metrics={metrics}
            reductionMethod={reductionConfig.method}
            clusteringMethod={clusteringConfig.method}
            datasetName={dataset ? dataset.name : 'Chưa chọn'}
            totalRows={dataset ? dataset.totalRows : 0}
            hasGroundTruth={Boolean(dataset?.labelColumn)}
            hasResults={dataPoints.length > 0}
            isComputing={isComputing}
            onSaveExperiment={handleSaveExperiment}
            onOpenHistory={handleOpenHistory}
            isSaving={isSavingExperiment}
            theme={theme}
          />
        </div>
      </div>

      {/* Experiments History Modal */}
      {isExperimentsModalOpen && (
        <Suspense fallback={null}>
          <ExperimentsModal
            isOpen={isExperimentsModalOpen}
            onClose={() => setIsExperimentsModalOpen(false)}
            experiments={experiments}
            onRefresh={fetchExperiments}
          />
        </Suspense>
      )}

      {/* Elbow Method & Silhouette Analysis Modal */}
      {isElbowModalOpen && (
        <Suspense fallback={null}>
          <ElbowModal
            isOpen={isElbowModalOpen}
            onClose={() => setIsElbowModalOpen(false)}
            featureMatrix={featureMatrix.length > 0 ? featureMatrix : (dataset ? extractFeatureMatrix(dataset.rawRows, selectedFeatures, reductionConfig.normalization) : [])}
            currentK={clusteringConfig.k}
            onApplyK={(k) => {
              const updatedConfig = { ...clusteringConfig, k };
              setClusteringConfig(updatedConfig);
              if (dataPoints.length > 0) handleApplyClustering(updatedConfig);
            }}
            theme={theme}
          />
        </Suspense>
      )}

      {/* Security Center & LAN Sharing Modal */}
      {isSecurityModalOpen && (
        <Suspense fallback={null}>
          <SecurityModal
            isOpen={isSecurityModalOpen}
            onClose={() => setIsSecurityModalOpen(false)}
            dbStatus={dbStatus}
            onRefreshDb={async () => {
              try {
                const res = await fetch('/api/health');
                if (res.ok) {
                  const data = await res.json();
                  setDbStatus(data.database || 'Online');
                }
              } catch (e) {
                setDbStatus('In-Memory Mode');
              }
            }}
          />
        </Suspense>
      )}

      {/* Full-Screen Raw Data Preview Modal */}
      {isRawDataModalOpen && dataset && (
        <RawDataModal
          isOpen={isRawDataModalOpen}
          onClose={() => setIsRawDataModalOpen(false)}
          datasetName={dataset.name}
          headers={dataset.headers}
          rawRows={dataset.rawRows}
          totalRows={dataset.totalRows}
          labelColumn={dataset.labelColumn ?? null}
          theme={theme}
        />
      )}

      {/* Dataset Manager Folder Modal */}
      {isDatasetManagerOpen && (
        <DatasetManagerModal
          isOpen={isDatasetManagerOpen}
          onClose={() => setIsDatasetManagerOpen(false)}
          datasetList={datasetList}
          activeDatasetId={activeDatasetId}
          onSelectDataset={(id) => {
            handleSelectDataset(id);
            setIsDatasetManagerOpen(false);
          }}
          onRenameDataset={handleRenameDataset}
          onDeleteDataset={handleDeleteDataset}
          onFileUpload={(file) => {
            handleFileUpload(file);
          }}
          isComputing={isComputing}
          theme={theme}
        />
      )}
    </div>
  );
}

export default App;
