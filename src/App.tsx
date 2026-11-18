import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  ArrowUpDown,
  BarChart3,
  Check,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Copy,
  Download,
  Eye,
  FileSpreadsheet,
  FileText,
  FileUp,
  FolderOpen,
  Gauge,
  History,
  Home,
  LayoutDashboard,
  LogOut,
  Menu,
  Moon,
  PieChart,
  Play,
  Plus,
  Printer,
  RefreshCw,
  Search,
  Settings,
  ShieldCheck,
  Sparkles,
  Sun,
  Trash2,
  Upload,
  Wand2,
  X,
} from 'lucide-react';
import {
  SAMPLE_DATASETS,
  buildAnomaliesCsv,
  buildReportCsv,
  buildReportExcelBytes,
  cleanDataset,
  computeCorrelationMatrix,
  detectAnomalies,
  optimizePrompt,
  parseCsvText,
  parseExcelArrayBuffer,
  profileDataset,
  runDataAnalysis,
  scorePromptQuality,
  validateAnalysisResult,
  type AnalysisHistoryEntry,
  type AnalysisResult,
  type AnomalyReport,
  type CleaningOptions,
  type CleaningReport,
  type DatasetProfile,
  type PromptOptimizationResult,
  type PromptQualityScore,
  type ValidationReport,
} from './engine/smartpromptEngine';
import {
  evaluateRouteAccess,
  loadStoredSession,
  logoutUserAccount,
  restoreAuthSession,
  type AuthSession,
  type PublicRoute,
} from './engine/authService';
import {
  ActivityByIntentChart,
  AnalysisChart,
  AnomalyScatterChart,
  PromptQualityComparisonChart,
} from './components/Charts';
import { HomePage, LoginPage, RegisterPage } from './components/PublicPages';
import { UnstructuredConverterView } from './components/UnstructuredConverterView';
import { SettingsView } from './components/SettingsView';
import {
  loadUserPreferences,
  saveUserPreferences,
  type UserPreferences,
} from './engine/userPreferences';

type NavSection =
  | 'dashboard'
  | 'workspace'
  | 'unstructured'
  | 'profiling'
  | 'cleaning'
  | 'optimizer'
  | 'quality'
  | 'analysis'
  | 'anomalies'
  | 'validation'
  | 'visualization'
  | 'history'
  | 'reports'
  | 'settings';

const NAV_ITEMS: { id: NavSection; label: string; icon: React.ComponentType<{ className?: string }> }[] = [
  { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { id: 'workspace', label: 'Dataset Workspace', icon: FolderOpen },
  { id: 'unstructured', label: 'Unstructured Data Converter', icon: FileUp },
  { id: 'profiling', label: 'Data Profiling', icon: FileSpreadsheet },
  { id: 'cleaning', label: 'Data Cleaning', icon: RefreshCw },
  { id: 'optimizer', label: 'Prompt Optimizer', icon: Sparkles },
  { id: 'quality', label: 'Prompt Quality Evaluator', icon: Gauge },
  { id: 'analysis', label: 'Data Analysis', icon: BarChart3 },
  { id: 'anomalies', label: 'Anomaly Detection', icon: Activity },
  { id: 'validation', label: 'Self-Validation', icon: ShieldCheck },
  { id: 'visualization', label: 'Visualization', icon: PieChart },
  { id: 'history', label: 'Analysis History', icon: History },
  { id: 'reports', label: 'Reports', icon: FileText },
  { id: 'settings', label: 'Settings', icon: Settings },
];

const HISTORY_STORAGE_KEY = 'smartprompt_ai_preview_history_v1';
const THEME_STORAGE_KEY = 'smartprompt_ai_theme_v1';

function statusTextColor(status: 'Verified' | 'Warning' | 'Unsupported'): string {
  if (status === 'Verified') return 'text-emerald-700 dark:text-emerald-400';
  if (status === 'Warning') return 'text-amber-700 dark:text-amber-400';
  return 'text-red-700 dark:text-red-400';
}

function statusDotColor(status: 'Verified' | 'Warning' | 'Unsupported'): string {
  if (status === 'Verified') return 'bg-emerald-600';
  if (status === 'Warning') return 'bg-amber-600';
  return 'bg-red-600';
}

function triggerDownload(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function parseRouteFromHash(hasSession: boolean): PublicRoute {
  try {
    const hash = window.location.hash.replace(/^#\/?/, '').toLowerCase();
    if (hash === 'login') return 'login';
    if (hash === 'register') return 'register';
    if (hash === 'dashboard' || hash === 'app') return 'app';
    if (hash === 'home' || hash === 'features' || hash === 'about') return 'home';
  } catch {
    // ignore hash errors
  }
  return hasSession ? 'app' : 'home';
}

export default function App() {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [authSession, setAuthSession] = useState<AuthSession | null>(() => loadStoredSession());
  const [authNotice, setAuthNotice] = useState<string | null>(() => {
    const sess = loadStoredSession();
    const initialReq = parseRouteFromHash(Boolean(sess));
    return evaluateRouteAccess(initialReq, sess).notice ?? null;
  });
  const [prefilledLoginId, setPrefilledLoginId] = useState<string>('');
  const [publicRoute, setPublicRoute] = useState<PublicRoute>(() => {
    const sess = loadStoredSession();
    const initialReq = parseRouteFromHash(Boolean(sess));
    return evaluateRouteAccess(initialReq, sess).allowedRoute;
  });

  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [activeSection, setActiveSection] = useState<NavSection>('dashboard');
  const [darkMode, setDarkMode] = useState<boolean>(() => {
    try {
      return localStorage.getItem(THEME_STORAGE_KEY) === 'dark';
    } catch {
      return false;
    }
  });

  // Core dataset state
  const initialSample = SAMPLE_DATASETS[0];
  const initialParsed = useMemo(() => parseCsvText(initialSample.csvText), [initialSample.csvText]);

  const [datasetName, setDatasetName] = useState<string>(initialSample.name);
  const [headers, setHeaders] = useState<string[]>(initialParsed.headers);
  const [rawRows, setRawRows] = useState<Record<string, string>[]>(initialParsed.rows);
  const [workingRows, setWorkingRows] = useState<Record<string, string>[]>(initialParsed.rows);
  const [cleaningReport, setCleaningReport] = useState<CleaningReport | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [rowFilter, setRowFilter] = useState<string>('');

  const [userPrefs, setUserPrefs] = useState<UserPreferences>(() => loadUserPreferences());
  const [notificationBanner, setNotificationBanner] = useState<string | null>(null);

  const [cleaningOptions, setCleaningOptions] = useState<CleaningOptions>(() => ({
    numericStrategy: loadUserPreferences().defaultNumericCleaning,
    categoricalStrategy: loadUserPreferences().defaultCategoricalCleaning,
    removeDuplicates: true,
    trimWhitespace: true,
  }));

  const profile: DatasetProfile = useMemo(
    () => profileDataset(datasetName, headers, workingRows),
    [datasetName, headers, workingRows]
  );

  // Prompt Optimizer & Quality Evaluator State
  const [promptInput, setPromptInput] = useState<string>('show sales');
  const [copiedPrompt, setCopiedPrompt] = useState(false);
  const [evalDraftPrompt, setEvalDraftPrompt] = useState<string>(
    'Rank category by sum of revenue across all regions'
  );

  const [optimization, setOptimization] = useState<PromptOptimizationResult>(() =>
    optimizePrompt('show sales', profileDataset(initialSample.name, initialParsed.headers, initialParsed.rows))
  );
  const [origScore, setOrigScore] = useState<PromptQualityScore>(() =>
    scorePromptQuality('show sales', profileDataset(initialSample.name, initialParsed.headers, initialParsed.rows))
  );
  const [optScore, setOptScore] = useState<PromptQualityScore>(() => {
    const p = profileDataset(initialSample.name, initialParsed.headers, initialParsed.rows);
    const o = optimizePrompt('show sales', p);
    return scorePromptQuality(o.optimizedPrompt, p);
  });

  const evaluatedDraftScore = useMemo(
    () => scorePromptQuality(evalDraftPrompt, profile),
    [evalDraftPrompt, profile]
  );

  // Analysis & Visualization State
  const [analysis, setAnalysis] = useState<AnalysisResult>(() => {
    const p = profileDataset(initialSample.name, initialParsed.headers, initialParsed.rows);
    const o = optimizePrompt('show sales', p);
    return runDataAnalysis(initialParsed.rows, p, o.taskPlan);
  });
  const [analysisViewMode, setAnalysisViewMode] = useState<'both' | 'chart' | 'table'>('both');
  const [chartOverride, setChartOverride] = useState<
    'auto' | 'bar' | 'horizontal_bar' | 'line' | 'pie' | 'histogram' | 'scatter' | 'heatmap'
  >('auto');

  // Anomaly Detection State
  const [selectedAnomalyFeatures, setSelectedAnomalyFeatures] = useState<string[]>(
    profile.numericalColumns.slice(0, 2)
  );
  const [contaminationPct, setContaminationPct] = useState<number>(8);
  const [anomalyReport, setAnomalyReport] = useState<AnomalyReport>(() =>
    detectAnomalies(initialParsed.rows, profile.numericalColumns.slice(0, 2), 8)
  );

  // Self-Validation State
  const [customClaim, setCustomClaim] = useState<string>('');
  const [validation, setValidation] = useState<ValidationReport>(() => {
    const p = profileDataset(initialSample.name, initialParsed.headers, initialParsed.rows);
    const o = optimizePrompt('show sales', p);
    const a = runDataAnalysis(initialParsed.rows, p, o.taskPlan);
    return validateAnalysisResult(initialParsed.rows, p, o.taskPlan, a);
  });

  // History & Reports State
  const [history, setHistory] = useState<AnalysisHistoryEntry[]>(() => {
    try {
      const saved = localStorage.getItem(HISTORY_STORAGE_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed)) return parsed as AnalysisHistoryEntry[];
      }
    } catch {
      // ignore storage failure
    }
    return [];
  });
  const [historySearch, setHistorySearch] = useState('');
  const [historyStatusFilter, setHistoryStatusFilter] = useState<'All' | 'Verified' | 'Warning' | 'Unsupported'>('All');
  const [historySort, setHistorySort] = useState<'date_desc' | 'date_asc' | 'dataset_asc'>('date_desc');
  const [inspectedEntry, setInspectedEntry] = useState<AnalysisHistoryEntry | null>(null);
  const [reportTitle, setReportTitle] = useState('SmartPrompt AI — Executive Analytical Report');

  const apiUrl = 'http://127.0.0.1:8000';

  const handleUpdatePreferences = (next: UserPreferences) => {
    setUserPrefs(next);
    saveUserPreferences(next);
    setCleaningOptions((prev) => ({
      ...prev,
      numericStrategy: next.defaultNumericCleaning,
      categoricalStrategy: next.defaultCategoricalCleaning,
    }));
  };

  const showNotification = (msg: string) => {
    setNotificationBanner(msg);
    setTimeout(() => {
      setNotificationBanner((cur) => (cur === msg ? null : cur));
    }, 4500);
  };

  useEffect(() => {
    try {
      localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(history));
    } catch {
      // ignore storage errors
    }
  }, [history]);

  useEffect(() => {
    try {
      localStorage.setItem(THEME_STORAGE_KEY, darkMode ? 'dark' : 'light');
    } catch {
      // ignore storage errors
    }
  }, [darkMode]);

  useEffect(() => {
    let active = true;
    restoreAuthSession({ apiUrl }).then((restored) => {
      if (!active) return;
      if (!restored && authSession) {
        setAuthSession(null);
        setPublicRoute((prev) => (prev === 'app' ? 'login' : prev));
        setAuthNotice('Your session expired. Please sign in again.');
      }
    });
    return () => {
      active = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const onHashChange = () => {
      const requested = parseRouteFromHash(Boolean(authSession));
      const check = evaluateRouteAccess(requested, authSession);
      setPublicRoute(check.allowedRoute);
      if (check.notice) setAuthNotice(check.notice);
    };
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, [authSession]);

  const navigatePublicRoute = (route: PublicRoute, sectionId?: string) => {
    const check = evaluateRouteAccess(route, authSession);
    if (check.notice) {
      setAuthNotice(check.notice);
    } else if (route !== 'login') {
      setAuthNotice(null);
    }
    setPublicRoute(check.allowedRoute);
    try {
      const targetHash =
        check.allowedRoute === 'app'
          ? '#/dashboard'
          : check.allowedRoute === 'home' && sectionId && sectionId !== 'top'
          ? `#/${sectionId}`
          : `#/${check.allowedRoute}`;
      if (window.location.hash !== targetHash) {
        window.location.hash = targetHash;
      }
    } catch {
      // ignore hash errors
    }
    if (check.allowedRoute === 'home' && sectionId) {
      setTimeout(() => {
        if (sectionId === 'top') {
          window.scrollTo({ top: 0, behavior: 'smooth' });
        } else {
          document.getElementById(sectionId)?.scrollIntoView({ behavior: 'smooth' });
        }
      }, 40);
    }
  };

  const handleLogout = async () => {
    await logoutUserAccount({ apiUrl });
    setAuthSession(null);
    setAuthNotice(null);
    setPrefilledLoginId('');
    setPublicRoute('home');
    try {
      window.location.hash = '#/home';
    } catch {
      // ignore
    }
  };

  const executeFullPipeline = (
    targetPrompt: string,
    targetRows: Record<string, string>[],
    targetProfile: DatasetProfile,
    featuresToInspect?: string[],
    claimOverride?: string,
    recordHistory = true
  ) => {
    const opt = optimizePrompt(targetPrompt, targetProfile);
    const sOrig = scorePromptQuality(targetPrompt, targetProfile);
    const sOpt = scorePromptQuality(opt.optimizedPrompt, targetProfile);
    const res = runDataAnalysis(targetRows, targetProfile, opt.taskPlan);
    const featList =
      featuresToInspect && featuresToInspect.length > 0
        ? featuresToInspect.filter((f) => targetProfile.numericalColumns.includes(f))
        : targetProfile.numericalColumns.slice(0, 2);
    const anom = detectAnomalies(targetRows, featList, contaminationPct);
    const val = validateAnalysisResult(targetRows, targetProfile, opt.taskPlan, res, claimOverride ?? customClaim);

    setOptimization(opt);
    setOrigScore(sOrig);
    setOptScore(sOpt);
    setAnalysis(res);
    setChartOverride('auto');
    setSelectedAnomalyFeatures(featList);
    setAnomalyReport(anom);
    setValidation(val);

    if (recordHistory) {
      const entry: AnalysisHistoryEntry = {
        analysisId: `ANL-${Date.now().toString(36).toUpperCase()}`,
        datasetName: targetProfile.datasetName,
        fingerprint: targetProfile.fingerprint,
        originalPrompt: targetPrompt,
        optimizedPrompt: opt.optimizedPrompt,
        detectedIntent: opt.detectedIntent,
        analysisSummary: res.summaryText,
        validationStatus: val.overallStatus,
        timestamp: new Date().toISOString(),
        originalScore: sOrig.overallScore,
        optimizedScore: sOpt.overallScore,
        anomalyCount: anom.anomalyCount,
      };
      setHistory((prev) => [entry, ...prev].slice(0, 30));
      if (userPrefs.notifyOnAnalysisComplete) {
        showNotification(
          `Analysis complete for ${targetProfile.datasetName} (${val.overallStatus}).`
        );
      }
    }
  };

  useEffect(() => {
    if (history.length === 0) {
      setHistory([
        {
          analysisId: 'ANL-DEMO-01',
          datasetName: profile.datasetName,
          fingerprint: profile.fingerprint,
          originalPrompt: 'show sales',
          optimizedPrompt: optimization.optimizedPrompt,
          detectedIntent: optimization.detectedIntent,
          analysisSummary: analysis.summaryText,
          validationStatus: validation.overallStatus,
          timestamp: new Date().toISOString(),
          originalScore: origScore.overallScore,
          optimizedScore: optScore.overallScore,
          anomalyCount: anomalyReport.anomalyCount,
        },
      ]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const applyParsedDataset = (
    name: string,
    parsedHeaders: string[],
    parsedRows: Record<string, string>[],
    defaultPrompt?: string
  ) => {
    const prof = profileDataset(name, parsedHeaders, parsedRows);
    const chosenPrompt =
      defaultPrompt ??
      (prof.numericalColumns[0] && prof.categoricalColumns[0]
        ? `Rank ${prof.categoricalColumns[0]} by ${prof.numericalColumns[0]}`
        : 'Summarize dataset columns and completeness');

    setDatasetName(name);
    setHeaders(parsedHeaders);
    setRawRows(parsedRows);
    setWorkingRows(parsedRows);
    setCleaningReport(null);
    setPromptInput(chosenPrompt);
    setCustomClaim('');
    setRowFilter('');

    executeFullPipeline(chosenPrompt, parsedRows, prof, prof.numericalColumns.slice(0, 2), '', true);
  };

  const loadNewCsv = (name: string, csvText: string, defaultPrompt?: string) => {
    try {
      setUploadError(null);
      const parsed = parseCsvText(csvText);
      applyParsedDataset(name, parsed.headers, parsed.rows, defaultPrompt);
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : 'Unable to parse CSV file.');
    }
  };

  const handleFileUpload = async (file: File) => {
    const lowerName = file.name.toLowerCase();
    const isCsv = lowerName.endsWith('.csv');
    const isExcel = lowerName.endsWith('.xlsx') || lowerName.endsWith('.xls');
    if (!isCsv && !isExcel) {
      setUploadError('Unsupported file format. Please upload a .csv, .xlsx, or .xls file.');
      return;
    }
    if (file.size > 15 * 1024 * 1024) {
      setUploadError('File exceeds 15 MB browser preview limit.');
      return;
    }
    try {
      setUploadError(null);
      if (isCsv) {
        const text = await file.text();
        loadNewCsv(file.name, text);
      } else {
        const buffer = await file.arrayBuffer();
        const parsed = parseExcelArrayBuffer(buffer);
        applyParsedDataset(file.name, parsed.headers, parsed.rows);
      }
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : 'Unable to parse uploaded file.');
    }
  };

  const handleApplyCleaning = () => {
    const rawProf = profileDataset(datasetName, headers, rawRows);
    const { cleanedRows, report } = cleanDataset(headers, rawRows, rawProf, cleaningOptions);
    const nextProf = profileDataset(datasetName, headers, cleanedRows);
    setWorkingRows(cleanedRows);
    setCleaningReport(report);
    executeFullPipeline(promptInput, cleanedRows, nextProf, selectedAnomalyFeatures, customClaim, true);
  };

  const handleResetToRaw = () => {
    const rawProf = profileDataset(datasetName, headers, rawRows);
    setWorkingRows(rawRows);
    setCleaningReport(null);
    executeFullPipeline(promptInput, rawRows, rawProf, selectedAnomalyFeatures, customClaim, false);
  };

  const handleDownloadCsvReport = () => {
    const csv = buildReportCsv(profile, optimization, analysis, validation, anomalyReport);
    const outName = `${profile.datasetName.replace(/\.(csv|xlsx|xls)$/i, '')}_smartprompt_report.csv`;
    triggerDownload(new Blob([csv], { type: 'text/csv;charset=utf-8;' }), outName);
    if (userPrefs.notifyOnReportExport) {
      showNotification(`Exported CSV report (${outName}).`);
    }
  };

  const handleDownloadExcelReport = () => {
    const bytes = buildReportExcelBytes(profile, optimization, analysis, validation, anomalyReport);
    const outName = `${profile.datasetName.replace(/\.(csv|xlsx|xls)$/i, '')}_smartprompt_report.xlsx`;
    triggerDownload(
      new Blob([new Uint8Array(bytes)], {
        type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      }),
      outName
    );
    if (userPrefs.notifyOnReportExport) {
      showNotification(`Exported Excel report (${outName}).`);
    }
  };

  const handleDownloadAnomalies = () => {
    const csv = buildAnomaliesCsv(anomalyReport);
    triggerDownload(
      new Blob([csv], { type: 'text/csv;charset=utf-8;' }),
      `${profile.datasetName.replace(/\.(csv|xlsx|xls)$/i, '')}_anomalies.csv`
    );
  };

  const handleCopyOptimizedPrompt = async () => {
    try {
      await navigator.clipboard.writeText(optimization.optimizedPrompt);
      setCopiedPrompt(true);
      setTimeout(() => setCopiedPrompt(false), 1800);
    } catch {
      setCopiedPrompt(false);
    }
  };

  // Real Dashboard Metrics derived strictly from history & active state
  const dashboardStats = useMemo(() => {
    const uniqueDatasets = new Set(history.map((h) => h.fingerprint));
    const scoredRuns = history.filter((h) => typeof h.optimizedScore === 'number');
    const avgQuality =
      scoredRuns.length > 0
        ? (scoredRuns.reduce((s, h) => s + (h.optimizedScore ?? 0), 0) / scoredRuns.length).toFixed(1)
        : optScore.overallScore.toFixed(1);

    const intentCounts = new Map<string, number>();
    for (const h of history) {
      intentCounts.set(h.detectedIntent, (intentCounts.get(h.detectedIntent) ?? 0) + 1);
    }
    const activityEntries = Array.from(intentCounts.entries())
      .map(([intent, count]) => ({ intent, count }))
      .sort((a, b) => b.count - a.count);

    const qualityRuns = history.slice(0, 5).map((h) => ({
      label: h.originalPrompt || h.datasetName,
      original: h.originalScore ?? origScore.overallScore,
      optimized: h.optimizedScore ?? optScore.overallScore,
    }));

    return {
      datasetsAnalyzed: Math.max(uniqueDatasets.size, 1),
      analysisRuns: history.length,
      avgPromptQuality: avgQuality,
      anomaliesDetected: anomalyReport.anomalyCount,
      activityEntries,
      qualityRuns,
    };
  }, [history, anomalyReport.anomalyCount, origScore.overallScore, optScore.overallScore]);

  const filteredPreviewRows = useMemo(() => {
    const q = rowFilter.trim().toLowerCase();
    if (!q) return workingRows.slice(0, 60);
    return workingRows
      .filter((row) => headers.some((h) => String(row[h] ?? '').toLowerCase().includes(q)))
      .slice(0, 60);
  }, [workingRows, headers, rowFilter]);

  const filteredHistory = useMemo(() => {
    const q = historySearch.trim().toLowerCase();
    const list = history.filter((h) => {
      const matchQuery =
        !q ||
        h.datasetName.toLowerCase().includes(q) ||
        h.originalPrompt.toLowerCase().includes(q) ||
        h.detectedIntent.toLowerCase().includes(q);
      const matchStatus = historyStatusFilter === 'All' || h.validationStatus === historyStatusFilter;
      return matchQuery && matchStatus;
    });
    return [...list].sort((a, b) => {
      if (historySort === 'date_asc') return a.timestamp.localeCompare(b.timestamp);
      if (historySort === 'dataset_asc') return a.datasetName.localeCompare(b.datasetName);
      return b.timestamp.localeCompare(a.timestamp);
    });
  }, [history, historySearch, historyStatusFilter, historySort]);

  const correlationMatrix = useMemo(
    () => computeCorrelationMatrix(workingRows, profile.numericalColumns),
    [workingRows, profile.numericalColumns]
  );

  const activeNavLabel = NAV_ITEMS.find((n) => n.id === activeSection)?.label ?? 'Dashboard';

  // Route Guard: Unauthenticated users can only see Home, Login, or Register
  if (publicRoute !== 'app' || !authSession) {
    if (publicRoute === 'login' || (publicRoute === 'app' && !authSession)) {
      return (
        <div className={darkMode ? 'dark' : ''}>
          <LoginPage
            apiUrl={apiUrl}
            darkMode={darkMode}
            notice={authNotice}
            prefilledIdentifier={prefilledLoginId}
            onToggleDarkMode={() => setDarkMode((d) => !d)}
            onNavigate={navigatePublicRoute}
            onLoginSuccess={(sess) => {
              setAuthSession(sess);
              setAuthNotice(null);
              setActiveSection('dashboard');
              setPublicRoute('app');
              try {
                window.location.hash = '#/dashboard';
              } catch {
                // ignore
              }
            }}
          />
        </div>
      );
    }

    if (publicRoute === 'register') {
      return (
        <div className={darkMode ? 'dark' : ''}>
          <RegisterPage
            apiUrl={apiUrl}
            darkMode={darkMode}
            onToggleDarkMode={() => setDarkMode((d) => !d)}
            onNavigate={navigatePublicRoute}
            onRegisterSuccess={(username, msg) => {
              setPrefilledLoginId(username);
              setAuthNotice(msg);
              setPublicRoute('login');
              try {
                window.location.hash = '#/login';
              } catch {
                // ignore
              }
            }}
          />
        </div>
      );
    }

    return (
      <div className={darkMode ? 'dark' : ''}>
        <HomePage
          session={authSession}
          darkMode={darkMode}
          onToggleDarkMode={() => setDarkMode((d) => !d)}
          onNavigate={navigatePublicRoute}
          onLogout={handleLogout}
        />
      </div>
    );
  }

  return (
    <div className={darkMode ? 'dark' : ''}>
      <div className="min-h-screen flex bg-[#F4F5F7] dark:bg-slate-950 text-slate-900 dark:text-slate-100 transition-colors">
        {/* Hidden CSV/Excel file input */}
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv,.xlsx,.xls"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) handleFileUpload(f);
            e.target.value = '';
          }}
        />

        {/* Mobile Backdrop */}
        {mobileMenuOpen && (
          <div
            className="fixed inset-0 z-20 bg-slate-900/40 lg:hidden"
            onClick={() => setMobileMenuOpen(false)}
          />
        )}

        {/* Collapsible Sidebar */}
        <aside
          className={`fixed inset-y-0 left-0 z-30 bg-white dark:bg-slate-900 border-r border-slate-200 dark:border-slate-800 flex flex-col transition-all duration-150 lg:static ${
            sidebarCollapsed ? 'lg:w-16' : 'lg:w-60'
          } w-60 ${mobileMenuOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}`}
        >
          <div className="h-14 px-3.5 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between gap-2">
            <div className="flex items-center gap-2.5 min-w-0">
              <div className="h-7 w-7 rounded bg-blue-900 dark:bg-indigo-600 flex items-center justify-center text-white shrink-0">
                <Sparkles className="h-4 w-4" />
              </div>
              {!sidebarCollapsed && (
                <span className="font-semibold text-sm tracking-tight text-slate-900 dark:text-white truncate">
                  SmartPrompt AI
                </span>
              )}
            </div>

            <button
              type="button"
              onClick={() => setSidebarCollapsed((c) => !c)}
              className="hidden lg:inline-flex p-1 rounded text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800 cursor-pointer"
              aria-label={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
              title={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            >
              {sidebarCollapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
            </button>

            <button
              type="button"
              onClick={() => setMobileMenuOpen(false)}
              className="lg:hidden p-1 text-slate-500 hover:text-slate-800 dark:hover:text-slate-200"
              aria-label="Close navigation"
            >
              <X className="h-4 w-4" />
            </button>
          </div>

          <nav className="flex-1 overflow-y-auto px-2.5 py-3 space-y-0.5">
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon;
              const isActive = activeSection === item.id;
              return (
                <button
                  key={item.id}
                  type="button"
                  title={sidebarCollapsed ? item.label : undefined}
                  onClick={() => {
                    setActiveSection(item.id);
                    setMobileMenuOpen(false);
                  }}
                  className={`w-full flex items-center gap-2.5 px-2.5 py-2 rounded-md text-[13px] font-medium transition-colors whitespace-nowrap cursor-pointer ${
                    sidebarCollapsed ? 'justify-center' : ''
                  } ${
                    isActive
                      ? 'bg-indigo-50 dark:bg-indigo-950/60 text-indigo-700 dark:text-indigo-300'
                      : 'text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800/70 hover:text-slate-900 dark:hover:text-slate-100'
                  }`}
                >
                  <Icon
                    className={`h-4 w-4 shrink-0 ${
                      isActive ? 'text-indigo-600 dark:text-indigo-400' : 'text-slate-400'
                    }`}
                  />
                  {!sidebarCollapsed && <span className="truncate">{item.label}</span>}
                </button>
              );
            })}
          </nav>

          {!sidebarCollapsed && (
            <div className="p-3.5 border-t border-slate-200 dark:border-slate-800 text-xs space-y-2.5 bg-slate-50/70 dark:bg-slate-900">
              <div className="space-y-1">
                <div className="font-medium text-slate-700 dark:text-slate-200 truncate" title={profile.datasetName}>
                  {profile.datasetName}
                </div>
                <div className="text-slate-500 dark:text-slate-400 font-mono num-tabular">
                  {profile.rowCount} × {profile.columnCount} · Q {profile.qualityReport.qualityScore}
                </div>
                <div className="text-slate-500 dark:text-slate-400 flex items-center gap-1.5">
                  <span className={`h-2 w-2 rounded-full ${statusDotColor(validation.overallStatus)}`} />
                  <span className={`font-medium ${statusTextColor(validation.overallStatus)}`}>
                    {validation.overallStatus}
                  </span>
                </div>
              </div>

              <div className="pt-2.5 border-t border-slate-200 dark:border-slate-800 space-y-2">
                <div className="truncate">
                  <div className="font-medium text-slate-800 dark:text-slate-100 truncate">
                    {authSession.user.fullName}
                  </div>
                  <div className="text-[11px] text-slate-500 dark:text-slate-400 truncate">
                    @{authSession.user.username}
                  </div>
                </div>
                <div className="flex items-center gap-1.5">
                  <button
                    type="button"
                    onClick={() => navigatePublicRoute('home', 'top')}
                    className="flex-1 inline-flex items-center justify-center gap-1 px-2 py-1.5 rounded border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-[11px] font-medium text-slate-700 dark:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-700 cursor-pointer"
                  >
                    <Home className="h-3 w-3" />
                    <span>Home</span>
                  </button>
                  <button
                    type="button"
                    onClick={handleLogout}
                    className="flex-1 inline-flex items-center justify-center gap-1 px-2 py-1.5 rounded border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-[11px] font-medium text-red-700 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-950/30 cursor-pointer"
                  >
                    <LogOut className="h-3 w-3" />
                    <span>Logout</span>
                  </button>
                </div>
              </div>
            </div>
          )}
        </aside>

        {/* Main Content Viewport */}
        <div className="flex-1 flex flex-col min-w-0">
          {/* Top Bar */}
          <header className="h-14 bg-white dark:bg-slate-900 border-b border-slate-200 dark:border-slate-800 px-4 sm:px-6 flex items-center justify-between gap-4 sticky top-0 z-10">
            <div className="flex items-center gap-3 min-w-0">
              <button
                type="button"
                onClick={() => setMobileMenuOpen(true)}
                className="lg:hidden p-1.5 rounded-md border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300"
                aria-label="Open menu"
              >
                <Menu className="h-4 w-4" />
              </button>
              <div className="flex items-center gap-2 text-sm truncate">
                <span className="text-slate-500 dark:text-slate-400">SmartPrompt AI</span>
                <span className="text-slate-300 dark:text-slate-700">/</span>
                <span className="font-semibold text-slate-900 dark:text-white truncate">{activeNavLabel}</span>
              </div>
            </div>

            <div className="flex items-center gap-2 shrink-0">
              <button
                type="button"
                onClick={() => setDarkMode((d) => !d)}
                className="p-2 rounded-md border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-700 cursor-pointer"
                aria-label={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
                title={darkMode ? 'Light mode' : 'Dark mode'}
              >
                {darkMode ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
              </button>
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-xs sm:text-sm font-medium text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-700 whitespace-nowrap shrink-0 cursor-pointer"
              >
                <Upload className="h-3.5 w-3.5" />
                <span>Upload Dataset</span>
              </button>
              <button
                type="button"
                onClick={() => setActiveSection('analysis')}
                className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-md bg-blue-900 dark:bg-indigo-600 text-xs sm:text-sm font-medium text-white hover:bg-blue-950 dark:hover:bg-indigo-700 whitespace-nowrap shrink-0 cursor-pointer"
              >
                <Plus className="h-3.5 w-3.5" />
                <span>New Analysis</span>
              </button>
              <button
                type="button"
                onClick={handleLogout}
                title={`Signed in as ${authSession.user.fullName}`}
                className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-md border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-xs font-medium text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-700 whitespace-nowrap shrink-0 cursor-pointer"
              >
                <LogOut className="h-3.5 w-3.5" />
                <span className="hidden sm:inline">Logout</span>
              </button>
            </div>
          </header>

          <main
            className={`flex-1 p-4 sm:p-6 max-w-7xl w-full mx-auto space-y-6 ${
              userPrefs.compactTables ? '[&_td]:py-1.5 [&_th]:py-1.5' : ''
            }`}
          >
            {notificationBanner && (
              <div
                role="status"
                className="no-print flex items-center justify-between gap-3 px-4 py-2.5 rounded-md border border-emerald-200 dark:border-emerald-900 bg-emerald-50 dark:bg-emerald-950/40 text-xs sm:text-sm text-emerald-800 dark:text-emerald-200"
              >
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-600 dark:text-emerald-400" />
                  <span>{notificationBanner}</span>
                </div>
                <button
                  type="button"
                  onClick={() => setNotificationBanner(null)}
                  className="text-emerald-700 dark:text-emerald-300 hover:opacity-75 cursor-pointer"
                  aria-label="Dismiss notification"
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              </div>
            )}

            {/* 1. DASHBOARD */}
            {activeSection === 'dashboard' && (
              <>
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                  <div>
                    <h1 className="text-xl font-semibold text-slate-900 dark:text-white">SmartPrompt AI</h1>
                    <p className="text-sm text-slate-600 dark:text-slate-400 mt-0.5">
                      Welcome back. Monitor dataset health, prompt optimization quality, anomaly checks, and validated runs.
                    </p>
                  </div>
                  <div className="text-xs text-slate-500 dark:text-slate-400 font-mono num-tabular">
                    Active: {profile.datasetName} ({profile.rowCount} rows)
                  </div>
                </div>

                {/* 4 KPI Cards */}
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                  <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-md p-4 shadow-[0_1px_2px_rgba(15,23,42,0.03)]">
                    <div className="text-xs font-medium text-slate-500 dark:text-slate-400">Datasets Analyzed</div>
                    <div className="text-2xl font-semibold text-slate-900 dark:text-white mt-1.5 font-mono num-tabular">
                      {dashboardStats.datasetsAnalyzed}
                    </div>
                    <div className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                      Current quality: {profile.qualityReport.qualityScore}/100
                    </div>
                  </div>

                  <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-md p-4 shadow-[0_1px_2px_rgba(15,23,42,0.03)]">
                    <div className="text-xs font-medium text-slate-500 dark:text-slate-400">Analysis Runs</div>
                    <div className="text-2xl font-semibold text-slate-900 dark:text-white mt-1.5 font-mono num-tabular">
                      {dashboardStats.analysisRuns}
                    </div>
                    <div className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                      Latest intent: {optimization.detectedIntent.replace(/_/g, ' ')}
                    </div>
                  </div>

                  <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-md p-4 shadow-[0_1px_2px_rgba(15,23,42,0.03)]">
                    <div className="text-xs font-medium text-slate-500 dark:text-slate-400">Average Prompt Quality</div>
                    <div className="text-2xl font-semibold text-indigo-700 dark:text-indigo-400 mt-1.5 font-mono num-tabular">
                      {dashboardStats.avgPromptQuality}/100
                    </div>
                    <div className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                      Active lift: {origScore.overallScore} → {optScore.overallScore}
                    </div>
                  </div>

                  <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-md p-4 shadow-[0_1px_2px_rgba(15,23,42,0.03)]">
                    <div className="text-xs font-medium text-slate-500 dark:text-slate-400">Anomalies Detected</div>
                    <div className="text-2xl font-semibold text-slate-900 dark:text-white mt-1.5 font-mono num-tabular">
                      {dashboardStats.anomaliesDetected}
                    </div>
                    <div className="text-xs text-slate-500 dark:text-slate-400 mt-1 flex items-center gap-1.5">
                      <span className={`h-2 w-2 rounded-full ${statusDotColor(validation.overallStatus)}`} />
                      <span>Validation: {validation.overallStatus}</span>
                    </div>
                  </div>
                </div>

                {/* Analysis Activity + Prompt Quality Charts */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
                  <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-md p-5 space-y-4">
                    <div className="flex items-center justify-between">
                      <h2 className="text-sm font-semibold text-slate-900 dark:text-white">Analysis Activity</h2>
                      <span className="text-xs text-slate-500 dark:text-slate-400">Runs by detected intent</span>
                    </div>
                    <ActivityByIntentChart entries={dashboardStats.activityEntries} />
                  </div>

                  <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-md p-5 space-y-4">
                    <div className="flex items-center justify-between">
                      <h2 className="text-sm font-semibold text-slate-900 dark:text-white">Prompt Quality</h2>
                      <span className="text-xs text-slate-500 dark:text-slate-400">Original vs. Optimized</span>
                    </div>
                    <PromptQualityComparisonChart runs={dashboardStats.qualityRuns} />
                  </div>
                </div>

                {/* Recent Analysis Table */}
                <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-md overflow-hidden">
                  <div className="px-5 py-3.5 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between">
                    <h2 className="text-sm font-semibold text-slate-900 dark:text-white">Recent Analysis</h2>
                    <button
                      type="button"
                      onClick={() => setActiveSection('history')}
                      className="text-xs font-medium text-indigo-700 dark:text-indigo-400 hover:underline cursor-pointer"
                    >
                      View full history →
                    </button>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs text-left">
                      <thead className="bg-slate-50 dark:bg-slate-800/60 border-b border-slate-200 dark:border-slate-800 text-slate-600 dark:text-slate-300">
                        <tr>
                          <th className="px-4 py-2.5 font-medium">Dataset</th>
                          <th className="px-4 py-2.5 font-medium">Prompt</th>
                          <th className="px-4 py-2.5 font-medium">Intent</th>
                          <th className="px-4 py-2.5 font-medium">Status</th>
                          <th className="px-4 py-2.5 font-medium text-right">Date</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                        {history.slice(0, 6).map((item) => (
                          <tr key={item.analysisId} className="hover:bg-slate-50 dark:hover:bg-slate-800/40">
                            <td className="px-4 py-2.5 font-medium text-slate-900 dark:text-slate-100">
                              {item.datasetName}
                            </td>
                            <td className="px-4 py-2.5 text-slate-600 dark:text-slate-300 max-w-xs truncate">
                              {item.originalPrompt}
                            </td>
                            <td className="px-4 py-2.5 font-mono text-slate-700 dark:text-slate-300">
                              {item.detectedIntent}
                            </td>
                            <td className={`px-4 py-2.5 font-semibold ${statusTextColor(item.validationStatus)}`}>
                              {item.validationStatus}
                            </td>
                            <td className="px-4 py-2.5 text-right font-mono num-tabular text-slate-500 dark:text-slate-400">
                              {new Date(item.timestamp).toLocaleDateString()}
                            </td>
                          </tr>
                        ))}
                        {history.length === 0 && (
                          <tr>
                            <td colSpan={5} className="px-4 py-8 text-center text-slate-500">
                              No analysis runs recorded yet. Click New Analysis to begin.
                            </td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>
              </>
            )}

            {/* 2. DATASET WORKSPACE */}
            {activeSection === 'workspace' && (
              <div className="space-y-6">
                <div>
                  <h1 className="text-xl font-semibold text-slate-900 dark:text-white">Dataset Workspace</h1>
                  <p className="text-sm text-slate-600 dark:text-slate-400 mt-0.5">
                    Upload CSV or Excel datasets, inspect record counts, missing values, duplicates, and column data types.
                  </p>
                </div>

                {uploadError && (
                  <div className="bg-red-50 dark:bg-red-950/40 border border-red-200 dark:border-red-900 rounded-md p-3.5 text-sm text-red-800 dark:text-red-300 flex items-center justify-between">
                    <span>{uploadError}</span>
                    <button type="button" onClick={() => setUploadError(null)} className="text-xs underline cursor-pointer">
                      Dismiss
                    </button>
                  </div>
                )}

                <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
                  <div className="lg:col-span-6 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-md p-5 space-y-4">
                    <h2 className="text-sm font-semibold text-slate-900 dark:text-white">Drag-and-Drop Upload</h2>
                    <div
                      onDragOver={(e) => e.preventDefault()}
                      onDrop={(e) => {
                        e.preventDefault();
                        const file = e.dataTransfer.files?.[0];
                        if (file) handleFileUpload(file);
                      }}
                      className="border border-dashed border-slate-300 dark:border-slate-700 rounded-md p-6 text-center bg-slate-50/70 dark:bg-slate-800/40 space-y-2"
                    >
                      <Upload className="h-6 w-6 text-slate-400 mx-auto" />
                      <div className="text-sm font-medium text-slate-800 dark:text-slate-200">
                        Drop CSV or Excel (.xlsx / .xls) file here
                      </div>
                      <div className="text-xs text-slate-500 dark:text-slate-400">
                        Processed locally in-browser · Max 15 MB
                      </div>
                      <button
                        type="button"
                        onClick={() => fileInputRef.current?.click()}
                        className="mt-2 px-3.5 py-1.5 rounded-md bg-blue-900 dark:bg-indigo-600 text-white text-xs font-medium hover:bg-blue-950 dark:hover:bg-indigo-700 cursor-pointer"
                      >
                        Browse Files
                      </button>
                    </div>
                  </div>

                  <div className="lg:col-span-6 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-md p-5 space-y-3">
                    <h2 className="text-sm font-semibold text-slate-900 dark:text-white">Sample Datasets</h2>
                    <div className="space-y-2">
                      {SAMPLE_DATASETS.map((ds) => {
                        const active = ds.name === datasetName;
                        return (
                          <button
                            key={ds.id}
                            type="button"
                            onClick={() => loadNewCsv(ds.name, ds.csvText, ds.suggestedPrompts[0])}
                            className={`w-full text-left p-3 rounded-md border transition-colors cursor-pointer ${
                              active
                                ? 'border-indigo-500 bg-indigo-50/50 dark:bg-indigo-950/40'
                                : 'border-slate-200 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-800/50'
                            }`}
                          >
                            <div className="flex items-center justify-between text-xs font-semibold text-slate-900 dark:text-slate-100">
                              <span>{ds.name}</span>
                              {active && <span className="text-indigo-700 dark:text-indigo-400">Active</span>}
                            </div>
                            <p className="text-xs text-slate-600 dark:text-slate-400 mt-1">{ds.description}</p>
                          </button>
                        );
                      })}
                    </div>
                  </div>
                </div>

                {/* Structural Summary Strip */}
                <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
                  <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-md p-4">
                    <div className="text-xs text-slate-500 dark:text-slate-400">Row Count</div>
                    <div className="text-xl font-semibold font-mono num-tabular mt-1">{profile.rowCount}</div>
                  </div>
                  <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-md p-4">
                    <div className="text-xs text-slate-500 dark:text-slate-400">Column Count</div>
                    <div className="text-xl font-semibold font-mono num-tabular mt-1">{profile.columnCount}</div>
                  </div>
                  <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-md p-4">
                    <div className="text-xs text-slate-500 dark:text-slate-400">Missing Values</div>
                    <div className="text-xl font-semibold font-mono num-tabular mt-1">
                      {profile.columns.reduce((s, c) => s + c.missingCount, 0)}
                    </div>
                  </div>
                  <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-md p-4">
                    <div className="text-xs text-slate-500 dark:text-slate-400">Duplicate Records</div>
                    <div className="text-xl font-semibold font-mono num-tabular mt-1">
                      {profile.qualityReport.duplicateRows}
                    </div>
                  </div>
                  <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-md p-4 col-span-2 lg:col-span-1">
                    <div className="text-xs text-slate-500 dark:text-slate-400">Data Types</div>
                    <div className="text-xs font-mono text-slate-700 dark:text-slate-300 mt-1.5">
                      {profile.numericalColumns.length} num · {profile.categoricalColumns.length} cat ·{' '}
                      {profile.datetimeColumns.length} date · {profile.booleanColumns.length} bool
                    </div>
                  </div>
                </div>

                {/* Dataset Preview Table */}
                <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-md overflow-hidden">
                  <div className="px-4 py-3 border-b border-slate-200 dark:border-slate-800 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                    <h2 className="text-sm font-semibold text-slate-900 dark:text-white">
                      Dataset Preview ({filteredPreviewRows.length} of {workingRows.length} rows)
                    </h2>
                    <div className="relative w-full sm:w-64">
                      <Search className="h-3.5 w-3.5 text-slate-400 absolute left-2.5 top-2.5" />
                      <input
                        type="text"
                        value={rowFilter}
                        onChange={(e) => setRowFilter(e.target.value)}
                        placeholder="Search rows..."
                        className="w-full pl-8 pr-3 py-1.5 text-xs border border-slate-300 dark:border-slate-700 rounded-md bg-white dark:bg-slate-800"
                      />
                    </div>
                  </div>
                  <div className="overflow-x-auto max-h-96">
                    <table className="w-full text-xs text-left">
                      <thead className="bg-slate-50 dark:bg-slate-800/70 border-b border-slate-200 dark:border-slate-800 text-slate-600 dark:text-slate-300 sticky top-0">
                        <tr>
                          <th className="px-3 py-2 font-mono text-slate-400">#</th>
                          {headers.map((h) => (
                            <th key={h} className="px-3 py-2 font-mono font-semibold whitespace-nowrap">
                              {h}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                        {filteredPreviewRows.map((row, idx) => (
                          <tr key={idx} className="hover:bg-slate-50 dark:hover:bg-slate-800/40">
                            <td className="px-3 py-2 font-mono text-slate-400 num-tabular">{idx}</td>
                            {headers.map((h) => {
                              const val = row[h];
                              const empty = val === undefined || val === '';
                              return (
                                <td
                                  key={h}
                                  className={`px-3 py-2 whitespace-nowrap font-mono num-tabular ${
                                    empty ? 'text-amber-600 italic' : 'text-slate-700 dark:text-slate-300'
                                  }`}
                                >
                                  {empty ? '(missing)' : val}
                                </td>
                              );
                            })}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}

            {/* 3. DATA PROFILING */}
            {activeSection === 'profiling' && (
              <div className="space-y-6">
                <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-2">
                  <div>
                    <h1 className="text-xl font-semibold text-slate-900 dark:text-white">Data Profiling</h1>
                    <p className="text-sm text-slate-600 dark:text-slate-400 mt-0.5">
                      Automated semantic typing, completeness metrics, and statistical IQR outlier detection for{' '}
                      <span className="font-medium text-slate-900 dark:text-slate-200">{profile.datasetName}</span>.
                    </p>
                  </div>
                  <div className="text-sm font-mono num-tabular text-slate-700 dark:text-slate-300">
                    Quality Score: <strong>{profile.qualityReport.qualityScore}/100</strong> ({profile.qualityReport.grade})
                  </div>
                </div>

                <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-md overflow-hidden">
                  <div className="px-4 py-3 border-b border-slate-200 dark:border-slate-800 flex items-center justify-between">
                    <h2 className="text-sm font-semibold text-slate-900 dark:text-white">
                      Column Statistics ({profile.columnCount} columns)
                    </h2>
                    <span className="text-xs font-mono num-tabular text-slate-500 dark:text-slate-400">
                      Completeness {profile.qualityReport.completenessPct}% · Outliers {profile.qualityReport.totalOutliers}
                    </span>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs text-left">
                      <thead className="bg-slate-50 dark:bg-slate-800/70 border-b border-slate-200 dark:border-slate-800 text-slate-600 dark:text-slate-300">
                        <tr>
                          <th className="px-4 py-2.5">Column</th>
                          <th className="px-4 py-2.5">Type</th>
                          <th className="px-4 py-2.5 text-right">Missing</th>
                          <th className="px-4 py-2.5 text-right">Unique</th>
                          <th className="px-4 py-2.5 text-right">Min / Max</th>
                          <th className="px-4 py-2.5 text-right">Mean / Median</th>
                          <th className="px-4 py-2.5 text-right">Std Dev</th>
                          <th className="px-4 py-2.5 text-right">IQR Outliers</th>
                          <th className="px-4 py-2.5">Top Value</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                        {profile.columns.map((c) => (
                          <tr key={c.name} className="hover:bg-slate-50 dark:hover:bg-slate-800/40">
                            <td className="px-4 py-2.5 font-mono font-medium text-slate-900 dark:text-slate-100">{c.name}</td>
                            <td className="px-4 py-2.5 text-slate-600 dark:text-slate-300">{c.semanticType}</td>
                            <td className="px-4 py-2.5 text-right font-mono num-tabular">
                              <span className={c.missingCount > 0 ? 'text-amber-600 font-semibold' : 'text-slate-600 dark:text-slate-400'}>
                                {c.missingCount} ({c.missingPct}%)
                              </span>
                            </td>
                            <td className="px-4 py-2.5 text-right font-mono num-tabular">{c.uniqueCount}</td>
                            <td className="px-4 py-2.5 text-right font-mono num-tabular">
                              {c.min !== undefined ? `${c.min} / ${c.max}` : '—'}
                            </td>
                            <td className="px-4 py-2.5 text-right font-mono num-tabular">
                              {c.mean !== undefined ? `${c.mean} / ${c.median}` : '—'}
                            </td>
                            <td className="px-4 py-2.5 text-right font-mono num-tabular">
                              {c.stdDev !== undefined ? c.stdDev : '—'}
                            </td>
                            <td className="px-4 py-2.5 text-right font-mono num-tabular">
                              {c.outlierCount !== undefined ? c.outlierCount : '—'}
                            </td>
                            <td className="px-4 py-2.5 text-slate-600 dark:text-slate-400 truncate max-w-[150px]">
                              {c.topValue ? `${c.topValue} (${c.topCount})` : '—'}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}

            {/* 4. DATA CLEANING */}
            {activeSection === 'cleaning' && (
              <div className="space-y-6">
                <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-2">
                  <div>
                    <h1 className="text-xl font-semibold text-slate-900 dark:text-white">Data Cleaning</h1>
                    <p className="text-sm text-slate-600 dark:text-slate-400 mt-0.5">
                      Apply non-destructive imputation, duplicate removal, and whitespace normalization while preserving the raw copy.
                    </p>
                  </div>
                  {cleaningReport && (
                    <button
                      type="button"
                      onClick={handleResetToRaw}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-xs font-medium text-slate-700 dark:text-slate-200 hover:bg-slate-50 cursor-pointer"
                    >
                      <RefreshCw className="h-3.5 w-3.5" />
                      <span>Restore Original Raw Data</span>
                    </button>
                  )}
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
                  <div className="lg:col-span-6 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-md p-5 space-y-4">
                    <h2 className="text-sm font-semibold text-slate-900 dark:text-white">Cleaning Strategy Configuration</h2>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      <label className="block space-y-1">
                        <span className="text-xs font-medium text-slate-700 dark:text-slate-300">Numerical Missing Values</span>
                        <select
                          value={cleaningOptions.numericStrategy}
                          onChange={(e) =>
                            setCleaningOptions((o) => ({
                              ...o,
                              numericStrategy: e.target.value as CleaningOptions['numericStrategy'],
                            }))
                          }
                          className="w-full px-2.5 py-2 text-sm border border-slate-300 dark:border-slate-700 rounded-md bg-white dark:bg-slate-800"
                        >
                          <option value="median">Impute with median</option>
                          <option value="mean">Impute with mean</option>
                          <option value="zero">Fill with 0</option>
                          <option value="drop_rows">Drop incomplete rows</option>
                        </select>
                      </label>

                      <label className="block space-y-1">
                        <span className="text-xs font-medium text-slate-700 dark:text-slate-300">Categorical Missing Values</span>
                        <select
                          value={cleaningOptions.categoricalStrategy}
                          onChange={(e) =>
                            setCleaningOptions((o) => ({
                              ...o,
                              categoricalStrategy: e.target.value as CleaningOptions['categoricalStrategy'],
                            }))
                          }
                          className="w-full px-2.5 py-2 text-sm border border-slate-300 dark:border-slate-700 rounded-md bg-white dark:bg-slate-800"
                        >
                          <option value="mode">Impute with mode</option>
                          <option value="unknown">Fill with &apos;Unknown&apos;</option>
                          <option value="drop_rows">Drop incomplete rows</option>
                        </select>
                      </label>
                    </div>

                    <div className="space-y-2 pt-1">
                      <label className="flex items-center gap-2 text-sm text-slate-700 dark:text-slate-300 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={cleaningOptions.removeDuplicates}
                          onChange={(e) =>
                            setCleaningOptions((o) => ({ ...o, removeDuplicates: e.target.checked }))
                          }
                          className="rounded border-slate-300 text-indigo-600"
                        />
                        <span>Remove exact duplicate rows ({profile.qualityReport.duplicateRows} detected)</span>
                      </label>

                      <label className="flex items-center gap-2 text-sm text-slate-700 dark:text-slate-300 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={cleaningOptions.trimWhitespace}
                          onChange={(e) =>
                            setCleaningOptions((o) => ({ ...o, trimWhitespace: e.target.checked }))
                          }
                          className="rounded border-slate-300 text-indigo-600"
                        />
                        <span>Trim leading and trailing cell whitespace</span>
                      </label>
                    </div>

                    <button
                      type="button"
                      onClick={handleApplyCleaning}
                      className="w-full py-2 px-4 rounded-md bg-blue-900 dark:bg-indigo-600 text-white text-sm font-medium hover:bg-blue-950 dark:hover:bg-indigo-700 cursor-pointer"
                    >
                      Apply Cleaning Strategy
                    </button>
                  </div>

                  <div className="lg:col-span-6 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-md p-5 space-y-3">
                    <h2 className="text-sm font-semibold text-slate-900 dark:text-white">Cleaning Audit Report</h2>
                    {cleaningReport ? (
                      <>
                        <div className="grid grid-cols-3 gap-3 text-xs border-b border-slate-100 dark:border-slate-800 pb-3">
                          <div>
                            <div className="text-slate-500">Rows</div>
                            <div className="font-mono font-semibold num-tabular text-sm mt-0.5">
                              {cleaningReport.rowsBefore} → {cleaningReport.rowsAfter}
                            </div>
                          </div>
                          <div>
                            <div className="text-slate-500">Duplicates Removed</div>
                            <div className="font-mono font-semibold num-tabular text-sm mt-0.5">
                              {cleaningReport.duplicatesRemoved}
                            </div>
                          </div>
                          <div>
                            <div className="text-slate-500">Missing Handled</div>
                            <div className="font-mono font-semibold num-tabular text-sm mt-0.5">
                              {cleaningReport.missingValuesHandled}
                            </div>
                          </div>
                        </div>
                        <ul className="space-y-1.5 text-xs text-slate-700 dark:text-slate-300">
                          {cleaningReport.actions.map((act, idx) => (
                            <li key={idx}>• {act}</li>
                          ))}
                        </ul>
                      </>
                    ) : (
                      <p className="text-xs text-slate-500 dark:text-slate-400">
                        Original raw dataset is currently active. Configure options on the left and click Apply Cleaning Strategy to preview non-destructive changes.
                      </p>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* 5. PROMPT OPTIMIZER (Two-Column Workspace) */}
            {activeSection === 'optimizer' && (
              <div className="space-y-6">
                <div>
                  <h1 className="text-xl font-semibold text-slate-900 dark:text-white">Prompt Optimizer</h1>
                  <p className="text-sm text-slate-600 dark:text-slate-400 mt-0.5">
                    Transform ambiguous questions into dataset-grounded analytical prompts verified against active schema columns.
                  </p>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
                  {/* Left Column: Input + Dataset Selector + Optimize Button */}
                  <div className="lg:col-span-5 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-md p-5 space-y-4">
                    <label className="block space-y-1.5">
                      <span className="text-xs font-semibold text-slate-700 dark:text-slate-300">Active Dataset</span>
                      <select
                        value={datasetName}
                        onChange={(e) => {
                          const found = SAMPLE_DATASETS.find((d) => d.name === e.target.value);
                          if (found) loadNewCsv(found.name, found.csvText, found.suggestedPrompts[0]);
                        }}
                        className="w-full px-3 py-2 text-sm border border-slate-300 dark:border-slate-700 rounded-md bg-white dark:bg-slate-800"
                      >
                        {!SAMPLE_DATASETS.some((d) => d.name === datasetName) && (
                          <option value={datasetName}>{datasetName} (Uploaded)</option>
                        )}
                        {SAMPLE_DATASETS.map((ds) => (
                          <option key={ds.id} value={ds.name}>
                            {ds.name}
                          </option>
                        ))}
                      </select>
                    </label>

                    <label className="block space-y-1.5">
                      <span className="text-xs font-semibold text-slate-700 dark:text-slate-300">User Prompt Input</span>
                      <textarea
                        rows={4}
                        value={promptInput}
                        onChange={(e) => setPromptInput(e.target.value)}
                        placeholder="e.g., show sales, or Rank category by revenue"
                        className="w-full px-3 py-2 text-sm border border-slate-300 dark:border-slate-700 rounded-md bg-white dark:bg-slate-800 focus:outline-none focus:border-indigo-600"
                      />
                    </label>

                    <button
                      type="button"
                      onClick={() =>
                        executeFullPipeline(promptInput, workingRows, profile, selectedAnomalyFeatures, customClaim, true)
                      }
                      className="w-full inline-flex items-center justify-center gap-2 py-2 px-4 rounded-md bg-blue-900 dark:bg-indigo-600 text-white text-sm font-medium hover:bg-blue-950 dark:hover:bg-indigo-700 cursor-pointer"
                    >
                      <Wand2 className="h-4 w-4" />
                      <span>Optimize Prompt</span>
                    </button>

                    <div className="space-y-1.5 pt-2 border-t border-slate-100 dark:border-slate-800">
                      <div className="text-xs text-slate-500 dark:text-slate-400">Sample prompts for {profile.datasetName}:</div>
                      <div className="flex flex-wrap gap-1.5">
                        {(
                          SAMPLE_DATASETS.find((s) => s.name === profile.datasetName)?.suggestedPrompts ?? [
                            'Summarize dataset',
                          ]
                        ).map((sp) => (
                          <button
                            key={sp}
                            type="button"
                            onClick={() => {
                              setPromptInput(sp);
                              executeFullPipeline(sp, workingRows, profile, selectedAnomalyFeatures, customClaim, true);
                            }}
                            className="text-left text-xs px-2.5 py-1 rounded border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 hover:bg-slate-100 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 cursor-pointer"
                          >
                            {sp}
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Right Column: Optimized Output + Intent + Score + Clarity/Specificity + Copy */}
                  <div className="lg:col-span-7 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-md p-5 space-y-4">
                    <div className="flex items-center justify-between gap-2">
                      <h2 className="text-sm font-semibold text-slate-900 dark:text-white">Optimized Prompt</h2>
                      <button
                        type="button"
                        onClick={handleCopyOptimizedPrompt}
                        className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded border border-slate-200 dark:border-slate-700 text-xs font-medium text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-800 cursor-pointer"
                      >
                        {copiedPrompt ? <Check className="h-3.5 w-3.5 text-emerald-600" /> : <Copy className="h-3.5 w-3.5" />}
                        <span>{copiedPrompt ? 'Copied' : 'Copy'}</span>
                      </button>
                    </div>

                    <div className="p-3.5 rounded-md bg-slate-50 dark:bg-slate-800/70 border border-slate-200 dark:border-slate-700 text-sm text-slate-900 dark:text-slate-100 leading-relaxed">
                      {optimization.optimizedPrompt}
                    </div>

                    {optimization.unavailableReferences.length > 0 && (
                      <div className="text-xs text-amber-800 dark:text-amber-300 bg-amber-50 dark:bg-amber-950/40 border border-amber-200 dark:border-amber-900 rounded-md px-3 py-2">
                        Unavailable references flagged: <span className="font-mono font-semibold">{optimization.unavailableReferences.join(', ')}</span>
                      </div>
                    )}

                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-1">
                      <div className="border border-slate-200 dark:border-slate-800 rounded p-3">
                        <div className="text-[11px] text-slate-500 dark:text-slate-400">Detected Intent</div>
                        <div className="text-xs font-mono font-semibold text-slate-900 dark:text-white mt-1">
                          {optimization.detectedIntent}
                        </div>
                      </div>
                      <div className="border border-slate-200 dark:border-slate-800 rounded p-3">
                        <div className="text-[11px] text-slate-500 dark:text-slate-400">Quality Score</div>
                        <div className="text-xs font-mono font-semibold text-indigo-700 dark:text-indigo-400 num-tabular mt-1">
                          {origScore.overallScore} → {optScore.overallScore} ({optScore.qualityClass})
                        </div>
                      </div>
                      <div className="border border-slate-200 dark:border-slate-800 rounded p-3">
                        <div className="text-[11px] text-slate-500 dark:text-slate-400">Clarity Indicator</div>
                        <div className="text-xs font-mono font-semibold text-slate-900 dark:text-white num-tabular mt-1">
                          {optScore.breakdown.intentClarity}/100
                        </div>
                      </div>
                      <div className="border border-slate-200 dark:border-slate-800 rounded p-3">
                        <div className="text-[11px] text-slate-500 dark:text-slate-400">Specificity Indicator</div>
                        <div className="text-xs font-mono font-semibold text-slate-900 dark:text-white num-tabular mt-1">
                          {optScore.breakdown.specificity}/100
                        </div>
                      </div>
                    </div>

                    <div className="border-t border-slate-100 dark:border-slate-800 pt-3 grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                      <div className="space-y-1">
                        <div className="font-semibold text-slate-700 dark:text-slate-300">Why Prompt Was Improved</div>
                        {optimization.improvementReasons.map((r, i) => (
                          <div key={i} className="text-slate-600 dark:text-slate-400">• {r}</div>
                        ))}
                      </div>
                      <div className="space-y-1">
                        <div className="font-semibold text-slate-700 dark:text-slate-300">Execution Steps</div>
                        {optimization.taskPlan.steps.map((s, idx) => (
                          <div key={idx} className="font-mono text-slate-600 dark:text-slate-400">
                            {idx + 1}. {s}
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* 6. PROMPT QUALITY EVALUATOR */}
            {activeSection === 'quality' && (
              <div className="space-y-6">
                <div>
                  <h1 className="text-xl font-semibold text-slate-900 dark:text-white">Prompt Quality Evaluator</h1>
                  <p className="text-sm text-slate-600 dark:text-slate-400 mt-0.5">
                    Evaluate any analytical prompt against {profile.datasetName} schema features (Specificity, Context, Intent Clarity, Column Relevance).
                  </p>
                </div>

                <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-md p-5 space-y-4">
                  <div className="flex flex-col sm:flex-row gap-2">
                    <input
                      type="text"
                      value={evalDraftPrompt}
                      onChange={(e) => setEvalDraftPrompt(e.target.value)}
                      placeholder="Enter any prompt to score..."
                      className="flex-1 px-3 py-2 text-sm border border-slate-300 dark:border-slate-700 rounded-md bg-white dark:bg-slate-800"
                    />
                    <button
                      type="button"
                      onClick={() => {
                        setPromptInput(evalDraftPrompt);
                        executeFullPipeline(
                          evalDraftPrompt,
                          workingRows,
                          profile,
                          selectedAnomalyFeatures,
                          customClaim,
                          true
                        );
                        setActiveSection('optimizer');
                      }}
                      className="px-4 py-2 rounded-md bg-blue-900 dark:bg-indigo-600 text-white text-xs font-medium hover:bg-blue-950 whitespace-nowrap shrink-0 cursor-pointer"
                    >
                      Send to Optimizer
                    </button>
                  </div>

                  <div className="grid grid-cols-2 lg:grid-cols-5 gap-4 pt-2">
                    <div className="border border-slate-200 dark:border-slate-800 rounded p-3.5">
                      <div className="text-xs text-slate-500 dark:text-slate-400">Overall Score</div>
                      <div className="text-2xl font-semibold font-mono num-tabular text-indigo-700 dark:text-indigo-400 mt-1">
                        {evaluatedDraftScore.overallScore}/100
                      </div>
                      <div className="text-xs text-slate-600 dark:text-slate-400 mt-0.5">
                        Class: {evaluatedDraftScore.qualityClass}
                      </div>
                    </div>
                    <div className="border border-slate-200 dark:border-slate-800 rounded p-3.5">
                      <div className="text-xs text-slate-500 dark:text-slate-400">Specificity</div>
                      <div className="text-xl font-semibold font-mono num-tabular mt-1">
                        {evaluatedDraftScore.breakdown.specificity}%
                      </div>
                    </div>
                    <div className="border border-slate-200 dark:border-slate-800 rounded p-3.5">
                      <div className="text-xs text-slate-500 dark:text-slate-400">Context Completeness</div>
                      <div className="text-xl font-semibold font-mono num-tabular mt-1">
                        {evaluatedDraftScore.breakdown.contextCompleteness}%
                      </div>
                    </div>
                    <div className="border border-slate-200 dark:border-slate-800 rounded p-3.5">
                      <div className="text-xs text-slate-500 dark:text-slate-400">Intent Clarity</div>
                      <div className="text-xl font-semibold font-mono num-tabular mt-1">
                        {evaluatedDraftScore.breakdown.intentClarity}%
                      </div>
                    </div>
                    <div className="border border-slate-200 dark:border-slate-800 rounded p-3.5">
                      <div className="text-xs text-slate-500 dark:text-slate-400">Column Relevance</div>
                      <div className="text-xl font-semibold font-mono num-tabular mt-1">
                        {evaluatedDraftScore.breakdown.columnRelevance}%
                      </div>
                    </div>
                  </div>

                  <div className="space-y-1 text-xs text-slate-600 dark:text-slate-400 border-t border-slate-100 dark:border-slate-800 pt-3">
                    {evaluatedDraftScore.notes.map((n, idx) => (
                      <div key={idx}>• {n}</div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* 7. DATA ANALYSIS */}
            {activeSection === 'analysis' && (
              <div className="space-y-6">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                  <div>
                    <h1 className="text-xl font-semibold text-slate-900 dark:text-white">Data Analysis</h1>
                    <p className="text-sm text-slate-600 dark:text-slate-400 mt-0.5">
                      Deterministic execution with inline self-validation and direct PDF / Excel export.
                    </p>
                  </div>
                  <div className="flex flex-wrap items-center gap-2">
                    <span className={`inline-flex items-center gap-1.5 text-xs font-semibold mr-2 ${statusTextColor(validation.overallStatus)}`}>
                      <span className={`h-2 w-2 rounded-full ${statusDotColor(validation.overallStatus)}`} />
                      Validation: {validation.overallStatus}
                    </span>
                    <button
                      type="button"
                      onClick={() => window.print()}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-xs font-medium text-slate-700 dark:text-slate-200 hover:bg-slate-50 cursor-pointer"
                    >
                      <Printer className="h-3.5 w-3.5" />
                      <span>Export PDF</span>
                    </button>
                    <button
                      type="button"
                      onClick={handleDownloadExcelReport}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-blue-900 dark:bg-indigo-600 text-xs font-medium text-white hover:bg-blue-950 cursor-pointer"
                    >
                      <FileSpreadsheet className="h-3.5 w-3.5" />
                      <span>Export Excel</span>
                    </button>
                  </div>
                </div>

                {/* Natural Language Input */}
                <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-md p-4 flex flex-col sm:flex-row gap-2">
                  <input
                    type="text"
                    value={promptInput}
                    onChange={(e) => setPromptInput(e.target.value)}
                    placeholder="Ask a data analysis question..."
                    className="flex-1 px-3 py-2 text-sm border border-slate-300 dark:border-slate-700 rounded-md bg-white dark:bg-slate-800"
                  />
                  <button
                    type="button"
                    onClick={() =>
                      executeFullPipeline(promptInput, workingRows, profile, selectedAnomalyFeatures, customClaim, true)
                    }
                    className="inline-flex items-center justify-center gap-1.5 px-4 py-2 rounded-md bg-blue-900 dark:bg-indigo-600 text-white text-sm font-medium hover:bg-blue-950 whitespace-nowrap shrink-0 cursor-pointer"
                  >
                    <Play className="h-3.5 w-3.5" />
                    <span>Run Analysis</span>
                  </button>
                </div>

                {/* Analysis Results KPIs */}
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                  {analysis.kpis.map((kpi) => (
                    <div key={kpi.label} className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-md p-4">
                      <div className="text-xs text-slate-500 dark:text-slate-400">{kpi.label}</div>
                      <div className="text-xl font-semibold font-mono num-tabular mt-1">{kpi.value}</div>
                    </div>
                  ))}
                </div>

                {/* View Mode Toggle + Summary */}
                <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-md p-5 space-y-4">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div>
                      <h2 className="text-sm font-semibold text-slate-900 dark:text-white">{analysis.title}</h2>
                      <p className="text-xs text-slate-600 dark:text-slate-400 mt-0.5">{analysis.summaryText}</p>
                    </div>

                    <div className="inline-flex rounded-md border border-slate-200 dark:border-slate-700 p-0.5 bg-slate-50 dark:bg-slate-800">
                      {(
                        [
                          { id: 'both', label: 'Chart + Table' },
                          { id: 'chart', label: 'Chart Only' },
                          { id: 'table', label: 'Table Only' },
                        ] as const
                      ).map((mode) => (
                        <button
                          key={mode.id}
                          type="button"
                          onClick={() => setAnalysisViewMode(mode.id)}
                          className={`px-2.5 py-1 rounded text-xs font-medium cursor-pointer ${
                            analysisViewMode === mode.id
                              ? 'bg-white dark:bg-slate-900 text-indigo-700 dark:text-indigo-400 shadow-xs'
                              : 'text-slate-600 dark:text-slate-400'
                          }`}
                        >
                          {mode.label}
                        </button>
                      ))}
                    </div>
                  </div>

                  {(analysisViewMode === 'both' || analysisViewMode === 'chart') && (
                    <AnalysisChart
                      analysis={analysis}
                      chartOverride={chartOverride}
                      correlationMatrix={correlationMatrix}
                    />
                  )}

                  {(analysisViewMode === 'both' || analysisViewMode === 'table') && (
                    <div className="overflow-x-auto border-t border-slate-100 dark:border-slate-800 pt-3">
                      <table className="w-full text-xs text-left">
                        <thead className="bg-slate-50 dark:bg-slate-800/70 border-b border-slate-200 dark:border-slate-800 text-slate-600 dark:text-slate-300">
                          <tr>
                            {analysis.columns.map((col) => (
                              <th key={col} className="px-4 py-2.5 font-mono">
                                {col}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                          {analysis.rows.map((r, i) => (
                            <tr key={i} className="hover:bg-slate-50 dark:hover:bg-slate-800/40">
                              {analysis.columns.map((col) => {
                                const val = r[col];
                                const isNum = typeof val === 'number';
                                return (
                                  <td
                                    key={col}
                                    className={`px-4 py-2.5 ${
                                      isNum ? 'font-mono num-tabular font-medium' : ''
                                    }`}
                                  >
                                    {isNum ? val.toLocaleString(undefined, { maximumFractionDigits: 2 }) : String(val ?? '')}
                                  </td>
                                );
                              })}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* 8. ANOMALY DETECTION */}
            {activeSection === 'anomalies' && (
              <div className="space-y-6">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                  <div>
                    <h1 className="text-xl font-semibold text-slate-900 dark:text-white">Anomaly Detection</h1>
                    <p className="text-sm text-slate-600 dark:text-slate-400 mt-0.5">
                      Multivariate statistical anomaly scoring across numerical columns (Isolation Forest mirror).
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={handleDownloadAnomalies}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-xs font-medium text-slate-700 dark:text-slate-200 hover:bg-slate-50 cursor-pointer"
                  >
                    <Download className="h-3.5 w-3.5" />
                    <span>Download Anomalies CSV</span>
                  </button>
                </div>

                <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-md p-5 grid grid-cols-1 lg:grid-cols-12 gap-5 items-end">
                  <div className="lg:col-span-7 space-y-2">
                    <div className="text-xs font-semibold text-slate-700 dark:text-slate-300">Numerical Features</div>
                    <div className="flex flex-wrap gap-2">
                      {profile.numericalColumns.map((col) => {
                        const checked = selectedAnomalyFeatures.includes(col);
                        return (
                          <button
                            key={col}
                            type="button"
                            onClick={() =>
                              setSelectedAnomalyFeatures((prev) =>
                                prev.includes(col) ? prev.filter((x) => x !== col) : [...prev, col]
                              )
                            }
                            className={`px-3 py-1.5 rounded border text-xs font-mono cursor-pointer ${
                              checked
                                ? 'border-indigo-600 bg-indigo-50 dark:bg-indigo-950/60 text-indigo-700 dark:text-indigo-300 font-semibold'
                                : 'border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-400'
                            }`}
                          >
                            {col}
                          </button>
                        );
                      })}
                    </div>
                  </div>
                  <div className="lg:col-span-3 space-y-1">
                    <div className="flex justify-between text-xs font-medium">
                      <span>Contamination</span>
                      <span className="font-mono num-tabular">{contaminationPct}%</span>
                    </div>
                    <input
                      type="range"
                      min={3}
                      max={25}
                      step={1}
                      value={contaminationPct}
                      onChange={(e) => setContaminationPct(Number(e.target.value))}
                      className="w-full accent-indigo-600"
                    />
                  </div>
                  <div className="lg:col-span-2">
                    <button
                      type="button"
                      onClick={() =>
                        setAnomalyReport(detectAnomalies(workingRows, selectedAnomalyFeatures, contaminationPct))
                      }
                      className="w-full py-2 px-3 rounded-md bg-blue-900 dark:bg-indigo-600 text-white text-xs font-medium hover:bg-blue-950 cursor-pointer"
                    >
                      Run Detection
                    </button>
                  </div>
                </div>

                {/* Anomaly Summary + Top Anomaly Score */}
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                  <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-md p-4">
                    <div className="text-xs text-slate-500 dark:text-slate-400">Anomalies Flagged</div>
                    <div className="text-xl font-semibold text-red-700 dark:text-red-400 font-mono num-tabular mt-1">
                      {anomalyReport.anomalyCount} ({anomalyReport.anomalyPercentage}%)
                    </div>
                  </div>
                  <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-md p-4">
                    <div className="text-xs text-slate-500 dark:text-slate-400">Peak Anomaly Score</div>
                    <div className="text-xl font-semibold font-mono num-tabular mt-1">
                      {anomalyReport.anomalies[0]?.anomalyScore ?? 0}
                    </div>
                  </div>
                  <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-md p-4">
                    <div className="text-xs text-slate-500 dark:text-slate-400">Rows Analyzed</div>
                    <div className="text-xl font-semibold font-mono num-tabular mt-1">{anomalyReport.rowsAnalyzed}</div>
                  </div>
                </div>

                <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-md p-5 space-y-3">
                  <h2 className="text-sm font-semibold text-slate-900 dark:text-white">
                    Anomaly Distribution ({anomalyReport.xFeature} vs {anomalyReport.yFeature})
                  </h2>
                  <AnomalyScatterChart report={anomalyReport} />
                </div>

                <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-md overflow-hidden">
                  <div className="px-4 py-3 border-b border-slate-200 dark:border-slate-800">
                    <h2 className="text-sm font-semibold text-slate-900 dark:text-white">Anomaly Records Table</h2>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs text-left">
                      <thead className="bg-slate-50 dark:bg-slate-800/70 border-b border-slate-200 dark:border-slate-800 text-slate-600 dark:text-slate-300">
                        <tr>
                          <th className="px-4 py-2.5">Row</th>
                          <th className="px-4 py-2.5 text-right">Anomaly Score</th>
                          <th className="px-4 py-2.5 text-right">Max Z-Score</th>
                          <th className="px-4 py-2.5">Deviating Feature</th>
                          <th className="px-4 py-2.5 text-right">Value</th>
                          <th className="px-4 py-2.5">Context</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                        {anomalyReport.anomalies.map((a) => (
                          <tr key={a.rowIndex} className="hover:bg-slate-50 dark:hover:bg-slate-800/40">
                            <td className="px-4 py-2.5 font-mono font-semibold text-red-700 dark:text-red-400">#{a.rowIndex}</td>
                            <td className="px-4 py-2.5 text-right font-mono num-tabular">{a.anomalyScore}</td>
                            <td className="px-4 py-2.5 text-right font-mono num-tabular">{a.zMax}σ</td>
                            <td className="px-4 py-2.5 font-mono">{a.topFeature}</td>
                            <td className="px-4 py-2.5 text-right font-mono num-tabular font-semibold">
                              {a.topFeatureValue.toLocaleString()}
                            </td>
                            <td className="px-4 py-2.5 font-mono text-slate-500 dark:text-slate-400 truncate max-w-sm">
                              {Object.entries(a.rowPreview)
                                .slice(0, 4)
                                .map(([k, v]) => `${k}=${v}`)
                                .join(' · ')}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}

            {/* 9. SELF-VALIDATION */}
            {activeSection === 'validation' && (
              <div className="space-y-6">
                <div>
                  <h1 className="text-xl font-semibold text-slate-900 dark:text-white">Self-Validation</h1>
                  <p className="text-sm text-slate-600 dark:text-slate-400 mt-0.5">
                    Independently verifies numerical totals and filters unsupported claims before report export.
                  </p>
                </div>

                <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-md p-4 flex flex-col sm:flex-row gap-2">
                  <input
                    type="text"
                    value={customClaim}
                    onChange={(e) => setCustomClaim(e.target.value)}
                    placeholder="Test a numerical claim (e.g. 'Total revenue is 999999')"
                    className="flex-1 px-3 py-2 text-sm border border-slate-300 dark:border-slate-700 rounded-md bg-white dark:bg-slate-800"
                  />
                  <button
                    type="button"
                    onClick={() =>
                      setValidation(
                        validateAnalysisResult(workingRows, profile, optimization.taskPlan, analysis, customClaim)
                      )
                    }
                    className="px-4 py-2 rounded-md bg-blue-900 dark:bg-indigo-600 text-white text-sm font-medium hover:bg-blue-950 whitespace-nowrap shrink-0 cursor-pointer"
                  >
                    Verify Claim
                  </button>
                </div>

                <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
                  <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-md p-4">
                    <div className="text-xs text-slate-500 dark:text-slate-400">Overall Status</div>
                    <div className={`text-xl font-semibold mt-1 flex items-center gap-2 ${statusTextColor(validation.overallStatus)}`}>
                      <span className={`h-2.5 w-2.5 rounded-full ${statusDotColor(validation.overallStatus)}`} />
                      <span>{validation.overallStatus}</span>
                    </div>
                  </div>
                  <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-md p-4">
                    <div className="text-xs text-slate-500 dark:text-slate-400">Verified</div>
                    <div className="text-xl font-semibold text-emerald-700 dark:text-emerald-400 font-mono num-tabular mt-1">
                      {validation.verifiedCount}
                    </div>
                  </div>
                  <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-md p-4">
                    <div className="text-xs text-slate-500 dark:text-slate-400">Warning</div>
                    <div className="text-xl font-semibold text-amber-700 dark:text-amber-400 font-mono num-tabular mt-1">
                      {validation.warningCount}
                    </div>
                  </div>
                  <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-md p-4">
                    <div className="text-xs text-slate-500 dark:text-slate-400">Unsupported</div>
                    <div className="text-xl font-semibold text-red-700 dark:text-red-400 font-mono num-tabular mt-1">
                      {validation.unsupportedCount}
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
                  <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-md p-5 space-y-2.5">
                    <h2 className="text-sm font-semibold text-slate-900 dark:text-white">Verified Evidence</h2>
                    <ul className="space-y-2 text-xs text-slate-700 dark:text-slate-300">
                      {validation.verifiedInsights.map((v, i) => (
                        <li key={i} className="flex items-start gap-2">
                          <CheckCircle2 className="h-4 w-4 text-emerald-600 shrink-0 mt-0.5" />
                          <span>{v}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                  <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-md p-5 space-y-2.5">
                    <h2 className="text-sm font-semibold text-slate-900 dark:text-white">Excluded Unsupported Claims</h2>
                    {validation.excludedClaims.length > 0 ? (
                      <ul className="space-y-2 text-xs text-red-700 dark:text-red-400">
                        {validation.excludedClaims.map((c, i) => (
                          <li key={i} className="flex items-start gap-2">
                            <AlertTriangle className="h-4 w-4 text-red-600 shrink-0 mt-0.5" />
                            <span>{c}</span>
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <p className="text-xs text-slate-500 dark:text-slate-400">
                        No unsupported claims flagged in the active run.
                      </p>
                    )}
                  </div>
                </div>

                <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-md overflow-hidden">
                  <div className="px-4 py-3 border-b border-slate-200 dark:border-slate-800">
                    <h2 className="text-sm font-semibold text-slate-900 dark:text-white">Validation Audit Details</h2>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs text-left">
                      <thead className="bg-slate-50 dark:bg-slate-800/70 border-b border-slate-200 dark:border-slate-800 text-slate-600 dark:text-slate-300">
                        <tr>
                          <th className="px-4 py-2.5">Check</th>
                          <th className="px-4 py-2.5">Status</th>
                          <th className="px-4 py-2.5">Expected</th>
                          <th className="px-4 py-2.5">Observed</th>
                          <th className="px-4 py-2.5">Detail</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                        {validation.checks.map((chk, idx) => (
                          <tr key={idx}>
                            <td className="px-4 py-2.5 font-medium">{chk.checkName}</td>
                            <td className={`px-4 py-2.5 font-semibold ${statusTextColor(chk.status)}`}>{chk.status}</td>
                            <td className="px-4 py-2.5 font-mono">{chk.expected}</td>
                            <td className="px-4 py-2.5 font-mono">{chk.observed}</td>
                            <td className="px-4 py-2.5 text-slate-600 dark:text-slate-400">{chk.detail}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}

            {/* 10. VISUALIZATION */}
            {activeSection === 'visualization' && (
              <div className="space-y-6">
                <div>
                  <h1 className="text-xl font-semibold text-slate-900 dark:text-white">Visualization Studio</h1>
                  <p className="text-sm text-slate-600 dark:text-slate-400 mt-0.5">
                    Interactive analytical figures with automatic chart selection and restrained enterprise styling.
                  </p>
                </div>

                <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-md p-5 space-y-4">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div>
                      <h2 className="text-sm font-semibold text-slate-900 dark:text-white">{analysis.title}</h2>
                      <p className="text-xs text-slate-500 dark:text-slate-400">
                        Recommended chart: {analysis.recommendedChart} · X: {analysis.xColumn} · Y: {analysis.yColumn}
                      </p>
                    </div>
                    <div className="flex flex-wrap gap-1">
                      {(
                        [
                          { id: 'auto', label: 'Auto' },
                          { id: 'bar', label: 'Bar' },
                          { id: 'horizontal_bar', label: 'Ranking' },
                          { id: 'line', label: 'Line' },
                          { id: 'pie', label: 'Pie' },
                          { id: 'histogram', label: 'Histogram' },
                          { id: 'scatter', label: 'Scatter' },
                          { id: 'heatmap', label: 'Heatmap' },
                        ] as const
                      ).map((opt) => (
                        <button
                          key={opt.id}
                          type="button"
                          onClick={() => setChartOverride(opt.id)}
                          className={`px-2.5 py-1 rounded text-xs font-medium cursor-pointer ${
                            chartOverride === opt.id
                              ? 'bg-blue-900 dark:bg-indigo-600 text-white'
                              : 'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300'
                          }`}
                        >
                          {opt.label}
                        </button>
                      ))}
                    </div>
                  </div>

                  <AnalysisChart
                    analysis={analysis}
                    chartOverride={chartOverride}
                    correlationMatrix={correlationMatrix}
                  />
                </div>
              </div>
            )}

            {/* 11. ANALYSIS HISTORY */}
            {activeSection === 'history' && (
              <div className="space-y-6">
                <div>
                  <h1 className="text-xl font-semibold text-slate-900 dark:text-white">Analysis History</h1>
                  <p className="text-sm text-slate-600 dark:text-slate-400 mt-0.5">
                    Searchable, filterable, metadata-only log of past analytical executions.
                  </p>
                </div>

                <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-md overflow-hidden">
                  <div className="px-4 py-3 border-b border-slate-200 dark:border-slate-800 flex flex-col md:flex-row md:items-center justify-between gap-2">
                    <div className="relative w-full md:w-72">
                      <Search className="h-3.5 w-3.5 text-slate-400 absolute left-2.5 top-2.5" />
                      <input
                        type="text"
                        value={historySearch}
                        onChange={(e) => setHistorySearch(e.target.value)}
                        placeholder="Search dataset, prompt, or intent..."
                        className="w-full pl-8 pr-3 py-1.5 text-xs border border-slate-300 dark:border-slate-700 rounded-md bg-white dark:bg-slate-800"
                      />
                    </div>
                    <div className="flex flex-wrap items-center gap-2">
                      <select
                        value={historyStatusFilter}
                        onChange={(e) =>
                          setHistoryStatusFilter(e.target.value as 'All' | 'Verified' | 'Warning' | 'Unsupported')
                        }
                        className="px-2.5 py-1.5 text-xs border border-slate-300 dark:border-slate-700 rounded-md bg-white dark:bg-slate-800"
                      >
                        <option value="All">All Statuses</option>
                        <option value="Verified">Verified</option>
                        <option value="Warning">Warning</option>
                        <option value="Unsupported">Unsupported</option>
                      </select>
                      <button
                        type="button"
                        onClick={() =>
                          setHistorySort((s) =>
                            s === 'date_desc' ? 'date_asc' : s === 'date_asc' ? 'dataset_asc' : 'date_desc'
                          )
                        }
                        className="inline-flex items-center gap-1 px-2.5 py-1.5 text-xs border border-slate-300 dark:border-slate-700 rounded-md bg-white dark:bg-slate-800 cursor-pointer"
                      >
                        <ArrowUpDown className="h-3 w-3" />
                        <span>
                          Sort:{' '}
                          {historySort === 'date_desc'
                            ? 'Newest'
                            : historySort === 'date_asc'
                            ? 'Oldest'
                            : 'Dataset A–Z'}
                        </span>
                      </button>
                    </div>
                  </div>

                  <div className="overflow-x-auto">
                    <table className="w-full text-xs text-left">
                      <thead className="bg-slate-50 dark:bg-slate-800/70 border-b border-slate-200 dark:border-slate-800 text-slate-600 dark:text-slate-300">
                        <tr>
                          <th className="px-4 py-2.5">Dataset Name</th>
                          <th className="px-4 py-2.5">Analysis Type</th>
                          <th className="px-4 py-2.5">Date</th>
                          <th className="px-4 py-2.5">Status</th>
                          <th className="px-4 py-2.5 text-right">Actions</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                        {filteredHistory.map((h) => (
                          <tr key={h.analysisId} className="hover:bg-slate-50 dark:hover:bg-slate-800/40">
                            <td className="px-4 py-2.5">
                              <div className="font-medium text-slate-900 dark:text-slate-100">{h.datasetName}</div>
                              <div className="font-mono text-[11px] text-slate-400 truncate max-w-xs">{h.originalPrompt}</div>
                            </td>
                            <td className="px-4 py-2.5 font-mono">{h.detectedIntent}</td>
                            <td className="px-4 py-2.5 font-mono num-tabular text-slate-500 dark:text-slate-400">
                              {new Date(h.timestamp).toLocaleString()}
                            </td>
                            <td className={`px-4 py-2.5 font-semibold ${statusTextColor(h.validationStatus)}`}>
                              {h.validationStatus}
                            </td>
                            <td className="px-4 py-2.5 text-right space-x-2 whitespace-nowrap">
                              <button
                                type="button"
                                onClick={() => setInspectedEntry(h)}
                                className="inline-flex items-center gap-1 px-2 py-1 rounded border border-slate-200 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800 cursor-pointer"
                              >
                                <Eye className="h-3 w-3" />
                                <span>View details</span>
                              </button>
                              <button
                                type="button"
                                onClick={handleDownloadExcelReport}
                                className="inline-flex items-center gap-1 px-2 py-1 rounded border border-slate-200 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800 cursor-pointer"
                              >
                                <Download className="h-3 w-3" />
                                <span>Download</span>
                              </button>
                              <button
                                type="button"
                                onClick={() => setHistory((prev) => prev.filter((x) => x.analysisId !== h.analysisId))}
                                className="p-1 text-slate-400 hover:text-red-600 cursor-pointer"
                                aria-label="Delete"
                              >
                                <Trash2 className="h-3.5 w-3.5" />
                              </button>
                            </td>
                          </tr>
                        ))}
                        {filteredHistory.length === 0 && (
                          <tr>
                            <td colSpan={5} className="px-4 py-8 text-center text-slate-500">
                              No history entries match your filters.
                            </td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>

                {inspectedEntry && (
                  <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-md p-5 space-y-3">
                    <div className="flex items-center justify-between">
                      <h2 className="text-sm font-semibold text-slate-900 dark:text-white">
                        Run Details · {inspectedEntry.analysisId}
                      </h2>
                      <button
                        type="button"
                        onClick={() => setInspectedEntry(null)}
                        className="text-xs text-slate-500 hover:text-slate-900 dark:hover:text-white cursor-pointer"
                      >
                        Close
                      </button>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                      <div>
                        <div className="font-semibold text-slate-500">Original Prompt</div>
                        <p className="mt-1 text-slate-800 dark:text-slate-200">{inspectedEntry.originalPrompt}</p>
                      </div>
                      <div>
                        <div className="font-semibold text-slate-500">Optimized Prompt</div>
                        <p className="mt-1 text-slate-800 dark:text-slate-200">{inspectedEntry.optimizedPrompt}</p>
                      </div>
                    </div>
                    <div className="text-xs text-slate-600 dark:text-slate-400">
                      <strong>Summary:</strong> {inspectedEntry.analysisSummary}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* 12. REPORTS */}
            {activeSection === 'reports' && (
              <div className="space-y-6">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                  <div>
                    <h1 className="text-xl font-semibold text-slate-900 dark:text-white">Reports</h1>
                    <p className="text-sm text-slate-600 dark:text-slate-400 mt-0.5">
                      Generate executive analytical deliverables in Excel (.xlsx), CSV, or PDF format with unsupported claims excluded.
                    </p>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <button
                      type="button"
                      onClick={handleDownloadExcelReport}
                      className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium cursor-pointer ${
                        userPrefs.defaultExportFormat === 'xlsx'
                          ? 'bg-blue-900 dark:bg-indigo-600 text-white hover:bg-blue-950'
                          : 'border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800'
                      }`}
                    >
                      <FileSpreadsheet className="h-3.5 w-3.5" />
                      <span>Download Excel (.xlsx)</span>
                    </button>
                    <button
                      type="button"
                      onClick={handleDownloadCsvReport}
                      className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium cursor-pointer ${
                        userPrefs.defaultExportFormat === 'csv'
                          ? 'bg-blue-900 dark:bg-indigo-600 text-white hover:bg-blue-950'
                          : 'border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800'
                      }`}
                    >
                      <Download className="h-3.5 w-3.5" />
                      <span>Download CSV</span>
                    </button>
                    <button
                      type="button"
                      onClick={() => window.print()}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-xs font-medium cursor-pointer"
                    >
                      <Printer className="h-3.5 w-3.5" />
                      <span>Download / Print PDF</span>
                    </button>
                  </div>
                </div>

                <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-md p-6 space-y-5">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-200 dark:border-slate-800 pb-4">
                    <input
                      type="text"
                      value={reportTitle}
                      onChange={(e) => setReportTitle(e.target.value)}
                      className="text-base font-semibold bg-transparent border-b border-transparent focus:border-indigo-500 focus:outline-none w-full sm:w-2/3"
                    />
                    <span className="text-xs font-mono text-slate-500">{profile.datasetName}</span>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-5 text-xs">
                    <div className="space-y-1">
                      <div className="font-semibold text-slate-500">Dataset & Validation</div>
                      <div>Shape: {profile.rowCount} rows × {profile.columnCount} columns</div>
                      <div>Quality Score: {profile.qualityReport.qualityScore}/100 ({profile.qualityReport.grade})</div>
                      <div>
                        Validation Status:{' '}
                        <span className={`font-semibold ${statusTextColor(validation.overallStatus)}`}>
                          {validation.overallStatus}
                        </span>
                      </div>
                    </div>
                    <div className="space-y-1">
                      <div className="font-semibold text-slate-500">Optimized Prompt & Finding</div>
                      <p className="text-slate-700 dark:text-slate-300">{optimization.optimizedPrompt}</p>
                      <p className="text-slate-600 dark:text-slate-400 pt-1">{analysis.summaryText}</p>
                    </div>
                  </div>

                  <div className="space-y-1.5 border-t border-slate-100 dark:border-slate-800 pt-4 text-xs">
                    <div className="font-semibold text-slate-700 dark:text-slate-300">Verified Statistical Insights</div>
                    {validation.verifiedInsights.map((v, i) => (
                      <div key={i} className="text-slate-600 dark:text-slate-400">• {v}</div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* 13. SETTINGS */}
            {activeSection === 'settings' && (
              <SettingsView
                session={authSession}
                darkMode={darkMode}
                preferences={userPrefs}
                onUpdateDarkMode={setDarkMode}
                onUpdatePreferences={handleUpdatePreferences}
                onSessionUpdated={setAuthSession}
                onLogout={handleLogout}
              />
            )}

            {/* 14. UNSTRUCTURED DATA CONVERTER */}
            {activeSection === 'unstructured' && (
              <UnstructuredConverterView onNavigateToWorkspace={() => setActiveSection('workspace')} />
            )}
          </main>
        </div>
      </div>
    </div>
  );
}
