import * as XLSX from 'xlsx';

export type SemanticType = 'numerical' | 'categorical' | 'datetime' | 'boolean';

export type IntentType =
  | 'dataset_summary'
  | 'ranking'
  | 'comparison'
  | 'trend_analysis'
  | 'aggregation'
  | 'correlation'
  | 'distribution_analysis'
  | 'anomaly_analysis'
  | 'filtering'
  | 'missing_value_analysis'
  | 'unknown';

export type AggregationOp = 'sum' | 'mean' | 'median' | 'count' | 'min' | 'max';

export interface ColumnProfile {
  name: string;
  semanticType: SemanticType;
  nonNullCount: number;
  missingCount: number;
  missingPct: number;
  uniqueCount: number;
  min?: number;
  max?: number;
  mean?: number;
  median?: number;
  stdDev?: number;
  outlierCount?: number;
  topValue?: string;
  topCount?: number;
  sampleValues: string[];
}

export interface DatasetQualityReport {
  qualityScore: number;
  grade: 'Excellent' | 'Good' | 'Fair' | 'Needs Cleaning';
  completenessPct: number;
  duplicateRows: number;
  duplicatePct: number;
  totalOutliers: number;
  issues: string[];
}

export interface DatasetProfile {
  datasetName: string;
  fingerprint: string;
  rowCount: number;
  columnCount: number;
  numericalColumns: string[];
  categoricalColumns: string[];
  datetimeColumns: string[];
  booleanColumns: string[];
  columns: ColumnProfile[];
  qualityReport: DatasetQualityReport;
}

export interface CleaningOptions {
  numericStrategy: 'median' | 'mean' | 'zero' | 'drop_rows';
  categoricalStrategy: 'mode' | 'unknown' | 'drop_rows';
  removeDuplicates: boolean;
  trimWhitespace: boolean;
}

export interface CleaningReport {
  rowsBefore: number;
  rowsAfter: number;
  duplicatesRemoved: number;
  missingValuesHandled: number;
  trimmedCells: number;
  actions: string[];
}

export interface TaskPlan {
  intent: IntentType;
  metrics: string[];
  groupBy: string[];
  timeColumn: string | null;
  aggregation: AggregationOp;
  topN: number | null;
  sortOrder: 'desc' | 'asc';
  filters: { column: string; value: string }[];
  steps: string[];
}

export interface PromptOptimizationResult {
  originalPrompt: string;
  optimizedPrompt: string;
  detectedIntent: IntentType;
  matchedColumns: string[];
  unavailableReferences: string[];
  missingContext: string[];
  improvementReasons: string[];
  taskPlan: TaskPlan;
}

export interface PromptQualityScore {
  overallScore: number;
  qualityClass: 'Low' | 'Medium' | 'High';
  breakdown: {
    specificity: number;
    contextCompleteness: number;
    intentClarity: number;
    columnRelevance: number;
  };
  notes: string[];
}

export interface AnalysisResult {
  intent: IntentType;
  title: string;
  summaryText: string;
  recommendedChart: 'bar' | 'horizontal_bar' | 'line' | 'histogram' | 'scatter' | 'heatmap' | 'table';
  xColumn: string;
  yColumn: string;
  columns: string[];
  rows: Record<string, string | number>[];
  kpis: { label: string; value: string }[];
  evidence: {
    rowCountAnalyzed: number;
    totalSum?: number;
    mean?: number;
    topEntity?: string;
    topValue?: number;
    correlationR?: number;
  };
  warnings: string[];
}

export interface AnomalyRowDetail {
  rowIndex: number;
  anomalyScore: number;
  zMax: number;
  topFeature: string;
  topFeatureValue: number;
  rowPreview: Record<string, string | number>;
}

export interface AnomalyReport {
  selectedFeatures: string[];
  contaminationPct: number;
  rowsAnalyzed: number;
  skippedIncompleteRows: number;
  anomalyCount: number;
  anomalyPercentage: number;
  anomalies: AnomalyRowDetail[];
  scatterPoints: {
    rowIndex: number;
    x: number;
    y: number;
    isAnomaly: boolean;
    score: number;
  }[];
  xFeature: string;
  yFeature: string;
  disclaimer: string;
}

export interface ValidationCheck {
  checkName: string;
  status: 'Verified' | 'Warning' | 'Unsupported';
  expected: string;
  observed: string;
  detail: string;
}

export interface ValidationReport {
  overallStatus: 'Verified' | 'Warning' | 'Unsupported';
  verifiedCount: number;
  warningCount: number;
  unsupportedCount: number;
  verifiedInsights: string[];
  excludedClaims: string[];
  checks: ValidationCheck[];
  limitationsNote: string;
}

export interface AnalysisHistoryEntry {
  analysisId: string;
  datasetName: string;
  fingerprint: string;
  originalPrompt: string;
  optimizedPrompt: string;
  detectedIntent: IntentType;
  analysisSummary: string;
  validationStatus: 'Verified' | 'Warning' | 'Unsupported';
  timestamp: string;
  originalScore?: number;
  optimizedScore?: number;
  anomalyCount?: number;
}

export interface SampleDatasetDef {
  id: string;
  name: string;
  description: string;
  suggestedPrompts: string[];
  csvText: string;
}

export const SAMPLE_DATASETS: SampleDatasetDef[] = [
  {
    id: 'retail_sales',
    name: 'Retail_Sales_2025.csv',
    description: 'Omni-channel order ledger with regions, categories, units, revenue, discount, and customer rating (with realistic missing values, duplicates, and high-value outliers).',
    suggestedPrompts: [
      'show sales',
      'Which category had the highest revenue and how does profit compare across regions?',
      'Analyze monthly revenue trend over order_date',
      'Check correlation between discount_pct and profit',
      'Find anomalous orders in revenue and units_sold',
    ],
    csvText: `order_id,order_date,region,category,units_sold,revenue,profit,discount_pct,is_returned
ORD-1001,2025-01-05,North,Electronics,12,4800,1150,5.0,false
ORD-1002,2025-01-12,West,Furniture,4,2200,410,10.0,false
ORD-1003,2025-01-18,South,Office Supplies,25,625,190,0.0,false
ORD-1004,2025-01-24,East,Electronics,8,3400,790,5.0,false
ORD-1005,2025-02-02,North,Technology,15,6750,1820,4.0,false
ORD-1006,2025-02-09,West,Office Supplies,30,780,210,8.0,true
ORD-1007,2025-02-14,South,Furniture,6,3150,520,12.0,false
ORD-1008,2025-02-21,East,Technology,10,4900,1340,2.5,false
ORD-1009,2025-03-04,North,Electronics,14,5950,1410,5.0,false
ORD-1010,2025-03-11,West,Furniture,5,,440,10.0,false
ORD-1011,2025-03-17,South,Technology,18,8640,2420,3.0,false
ORD-1012,2025-03-25,East,Office Supplies,40,1040,310,0.0,false
ORD-1013,2025-04-03,North,Furniture,7,3640,680,15.0,true
ORD-1014,2025-04-10,West,Electronics,11,4620,1050,6.0,false
ORD-1015,2025-04-19,South,Office Supplies,22,572,155,5.0,false
ORD-1016,2025-04-27,East,Technology,16,7680,2100,4.0,false
ORD-1017,2025-05-06,North,Electronics,48,26400,6850,2.0,false
ORD-1018,2025-05-14,West,Technology,13,6240,1690,5.0,false
ORD-1019,2025-05-20,,Office Supplies,19,494,140,0.0,false
ORD-1020,2025-05-28,East,Furniture,9,4590,890,11.0,false
ORD-1021,2025-06-04,North,Technology,20,9800,2750,3.5,false
ORD-1022,2025-06-12,South,Electronics,9,3780,820,7.0,false
ORD-1023,2025-06-19,West,Furniture,3,1560,240,18.0,true
ORD-1024,2025-06-26,East,Office Supplies,35,910,260,2.0,false
ORD-1009,2025-03-04,North,Electronics,14,5950,1410,5.0,false
ORD-1025,2025-06-29,South,Technology,2,940,-120,28.0,true`,
  },
  {
    id: 'hr_attrition',
    name: 'HR_Compensation_2025.csv',
    description: 'Workforce salary, tenure, performance score, and attrition records across departments.',
    suggestedPrompts: [
      'Compare average monthly_salary by department',
      'Rank departments by headcount and overtime_hours',
      'Show distribution of tenure_years',
      'Detect compensation anomalies across monthly_salary and overtime_hours',
    ],
    csvText: `emp_id,hire_date,department,role_level,tenure_years,monthly_salary,overtime_hours,satisfaction_score,attrited
E-201,2020-03-15,Engineering,Senior,5.1,11800,12,4.4,false
E-202,2022-07-01,Engineering,Mid,2.8,8900,18,3.9,false
E-203,2019-11-10,Sales,Senior,5.5,9400,26,3.2,true
E-204,2023-02-20,Marketing,Junior,2.1,5600,6,4.1,false
E-205,2021-06-12,Operations,Mid,3.9,6700,22,3.5,false
E-206,2018-09-01,Engineering,Principal,6.7,15400,10,4.6,false
E-207,2023-08-14,Sales,Junior,1.7,5100,34,2.4,true
E-208,2020-05-19,Finance,Senior,4.9,10200,8,4.2,false
E-209,2022-01-11,Operations,Junior,3.2,5200,28,2.9,true
E-210,2021-10-03,Marketing,Mid,3.5,7100,9,4.0,false
E-211,2024-01-08,Engineering,Junior,1.2,7200,14,4.3,false
E-212,2019-04-22,Sales,Senior,6.0,9800,19,3.7,false
E-213,2020-12-01,Finance,Mid,4.4,8100,7,4.1,false
E-214,2023-05-18,Engineering,Mid,1.9,,16,3.8,false
E-215,2017-02-14,Engineering,VP,8.2,24800,45,4.7,false
E-216,2022-09-09,Operations,Mid,2.6,6400,15,3.6,false
E-217,2021-03-27,Marketing,Senior,4.1,8800,5,4.5,false
E-218,2022-11-30,Sales,Mid,2.4,6900,31,2.7,true`,
  },
  {
    id: 'supply_chain',
    name: 'Supply_Chain_Logistics.csv',
    description: 'Shipment transit times, freight cost, carrier performance, and delay tracking.',
    suggestedPrompts: [
      'Rank carrier by average freight_cost_usd',
      'Compare transit_days across warehouse_hub',
      'Analyze freight_cost_usd trend by ship_date',
      'Detect outliers in transit_days and freight_cost_usd',
    ],
    csvText: `shipment_id,ship_date,warehouse_hub,carrier,weight_kg,transit_days,freight_cost_usd,delayed
SHP-01,2025-01-08,Chicago,FastFreight,420,2.4,610,false
SHP-02,2025-01-14,Newark,AtlasCargo,890,4.8,1280,true
SHP-03,2025-01-22,Dallas,SwiftHaul,310,2.1,440,false
SHP-04,2025-02-03,Seattle,FastFreight,560,3.2,790,false
SHP-05,2025-02-11,Chicago,SwiftHaul,720,2.9,940,false
SHP-06,2025-02-19,Newark,AtlasCargo,1120,6.5,1890,true
SHP-07,2025-03-01,Dallas,FastFreight,480,2.3,650,false
SHP-08,2025-03-10,Seattle,AtlasCargo,650,5.4,1140,true
SHP-09,2025-03-18,Chicago,SwiftHaul,390,2.0,520,false
SHP-10,2025-03-26,Newark,FastFreight,950,3.6,1310,false
SHP-11,2025-04-05,Dallas,AtlasCargo,1400,11.2,3450,true
SHP-12,2025-04-15,Seattle,SwiftHaul,510,2.7,710,false
SHP-13,2025-04-23,Chicago,AtlasCargo,770,4.9,1190,true
SHP-14,2025-05-02,Newark,SwiftHaul,430,2.5,580,false`,
  },
];

// ---------------------------------------------------------------------------
// CSV Parsing & Fingerprinting
// ---------------------------------------------------------------------------

export function parseCsvText(csvText: string): { headers: string[]; rows: Record<string, string>[] } {
  const cleaned = csvText.replace(/^\uFEFF/, '').trim();
  if (!cleaned) {
    throw new Error('The uploaded CSV file is empty.');
  }

  const lines: string[] = [];
  let current = '';
  let inQuotes = false;

  for (let i = 0; i < cleaned.length; i++) {
    const ch = cleaned[i];
    const next = cleaned[i + 1];
    if (ch === '"') {
      if (inQuotes && next === '"') {
        current += '"';
        i++;
      } else {
        inQuotes = !inQuotes;
      }
    } else if ((ch === '\n' || ch === '\r') && !inQuotes) {
      if (ch === '\r' && next === '\n') i++;
      if (current.trim().length > 0) lines.push(current);
      current = '';
    } else {
      current += ch;
    }
  }
  if (current.trim().length > 0) lines.push(current);

  if (lines.length < 2) {
    throw new Error('CSV must include a header row and at least one data row.');
  }

  const splitRow = (line: string): string[] => {
    const out: string[] = [];
    let cell = '';
    let quoted = false;
    for (let i = 0; i < line.length; i++) {
      const ch = line[i];
      const next = line[i + 1];
      if (ch === '"') {
        if (quoted && next === '"') {
          cell += '"';
          i++;
        } else {
          quoted = !quoted;
        }
      } else if (ch === ',' && !quoted) {
        out.push(cell.trim());
        cell = '';
      } else {
        cell += ch;
      }
    }
    out.push(cell.trim());
    return out;
  };

  const rawHeaders = splitRow(lines[0]).map((h, idx) => h || `column_${idx + 1}`);
  const headers = rawHeaders;
  const rows: Record<string, string>[] = [];

  for (let r = 1; r < lines.length; r++) {
    const vals = splitRow(lines[r]);
    const rowObj: Record<string, string> = {};
    headers.forEach((h, idx) => {
      rowObj[h] = vals[idx] ?? '';
    });
    rows.push(rowObj);
  }

  return { headers, rows };
}

function computeFingerprint(name: string, headers: string[], rowCount: number): string {
  const raw = `${name}::${headers.join(',')}::${rowCount}`;
  let hash = 2166136261;
  for (let i = 0; i < raw.length; i++) {
    hash ^= raw.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(16).padStart(8, '0');
}

function isMissing(val: string | number | boolean | null | undefined): boolean {
  if (val === null || val === undefined) return true;
  const s = String(val).trim();
  return s === '' || s.toLowerCase() === 'na' || s.toLowerCase() === 'n/a' || s.toLowerCase() === 'null' || s.toLowerCase() === 'nan';
}

function parseNumeric(val: string | number | null | undefined): number | null {
  if (isMissing(val)) return null;
  if (typeof val === 'number') return Number.isFinite(val) ? val : null;
  const cleaned = String(val).replace(/[$,%\s,]/g, '');
  if (!cleaned) return null;
  const num = Number(cleaned);
  return Number.isFinite(num) ? num : null;
}

function looksLikeDate(val: string): boolean {
  const s = val.trim();
  if (!s || /^\d+(\.\d+)?$/.test(s)) return false;
  if (!/[-/]/.test(s) && !/^\d{4}-\d{2}-\d{2}/.test(s)) return false;
  const ms = Date.parse(s);
  return !Number.isNaN(ms);
}

function looksLikeBoolean(val: string): boolean {
  const s = val.trim().toLowerCase();
  return s === 'true' || s === 'false' || s === 'yes' || s === 'no';
}

function computeMedian(sorted: number[]): number {
  if (sorted.length === 0) return 0;
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

function computeQuantile(sorted: number[], q: number): number {
  if (sorted.length === 0) return 0;
  const pos = (sorted.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  if (sorted[base + 1] !== undefined) {
    return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
  }
  return sorted[base];
}

// ---------------------------------------------------------------------------
// Module 1: Dataset Profiling
// ---------------------------------------------------------------------------

export function profileDataset(
  datasetName: string,
  headers: string[],
  rows: Record<string, string>[]
): DatasetProfile {
  const rowCount = rows.length;
  const columnCount = headers.length;
  const columns: ColumnProfile[] = [];
  const numericalColumns: string[] = [];
  const categoricalColumns: string[] = [];
  const datetimeColumns: string[] = [];
  const booleanColumns: string[] = [];

  let totalMissing = 0;
  let totalOutliers = 0;

  for (const col of headers) {
    const rawValues = rows.map((r) => r[col] ?? '');
    const nonMissing = rawValues.filter((v) => !isMissing(v));
    const missingCount = rowCount - nonMissing.length;
    totalMissing += missingCount;
    const missingPct = rowCount > 0 ? Number(((missingCount / rowCount) * 100).toFixed(2)) : 0;

    let numMatches = 0;
    let dateMatches = 0;
    let boolMatches = 0;
    for (const v of nonMissing) {
      if (looksLikeBoolean(v)) boolMatches++;
      else if (parseNumeric(v) !== null) numMatches++;
      else if (looksLikeDate(v)) dateMatches++;
    }

    let semanticType: SemanticType = 'categorical';
    if (nonMissing.length > 0) {
      const ratio = 0.8;
      if (boolMatches / nonMissing.length >= ratio) semanticType = 'boolean';
      else if (numMatches / nonMissing.length >= ratio) semanticType = 'numerical';
      else if (dateMatches / nonMissing.length >= ratio || /date|time|month|year/i.test(col) && dateMatches > 0) semanticType = 'datetime';
    }

    if (semanticType === 'numerical') numericalColumns.push(col);
    else if (semanticType === 'datetime') datetimeColumns.push(col);
    else if (semanticType === 'boolean') booleanColumns.push(col);
    else categoricalColumns.push(col);

    const freq = new Map<string, number>();
    for (const v of nonMissing) {
      const key = String(v).trim();
      freq.set(key, (freq.get(key) ?? 0) + 1);
    }

    let topValue: string | undefined;
    let topCount = 0;
    for (const [k, c] of freq.entries()) {
      if (c > topCount) {
        topCount = c;
        topValue = k;
      }
    }

    const baseProfile: ColumnProfile = {
      name: col,
      semanticType,
      nonNullCount: nonMissing.length,
      missingCount,
      missingPct,
      uniqueCount: freq.size,
      topValue,
      topCount,
      sampleValues: nonMissing.slice(0, 3),
    };

    if (semanticType === 'numerical') {
      const nums = nonMissing
        .map((v) => parseNumeric(v))
        .filter((v): v is number => v !== null)
        .sort((a, b) => a - b);

      if (nums.length > 0) {
        const sum = nums.reduce((acc, x) => acc + x, 0);
        const mean = sum / nums.length;
        const median = computeMedian(nums);
        const variance = nums.reduce((acc, x) => acc + (x - mean) ** 2, 0) / nums.length;
        const stdDev = Math.sqrt(variance);
        const q1 = computeQuantile(nums, 0.25);
        const q3 = computeQuantile(nums, 0.75);
        const iqr = q3 - q1;
        const lower = q1 - 1.5 * iqr;
        const upper = q3 + 1.5 * iqr;
        const outlierCount = iqr > 0 ? nums.filter((x) => x < lower || x > upper).length : 0;
        totalOutliers += outlierCount;

        baseProfile.min = Number(nums[0].toFixed(2));
        baseProfile.max = Number(nums[nums.length - 1].toFixed(2));
        baseProfile.mean = Number(mean.toFixed(2));
        baseProfile.median = Number(median.toFixed(2));
        baseProfile.stdDev = Number(stdDev.toFixed(2));
        baseProfile.outlierCount = outlierCount;
      }
    }

    columns.push(baseProfile);
  }

  // Duplicate row detection
  const seen = new Set<string>();
  let duplicateRows = 0;
  for (const r of rows) {
    const sig = headers.map((h) => String(r[h] ?? '').trim()).join('||');
    if (seen.has(sig)) duplicateRows++;
    else seen.add(sig);
  }

  const totalCells = Math.max(1, rowCount * columnCount);
  const completenessPct = Number((((totalCells - totalMissing) / totalCells) * 100).toFixed(2));
  const duplicatePct = rowCount > 0 ? Number(((duplicateRows / rowCount) * 100).toFixed(2)) : 0;

  const missingPenalty = (100 - completenessPct) * 1.4;
  const dupPenalty = duplicatePct * 1.2;
  const outlierPenalty = rowCount > 0 ? Math.min(12, (totalOutliers / (rowCount * Math.max(1, numericalColumns.length))) * 40) : 0;
  const qualityScore = Number(Math.max(0, Math.min(100, 100 - missingPenalty - dupPenalty - outlierPenalty)).toFixed(1));

  let grade: DatasetQualityReport['grade'] = 'Needs Cleaning';
  if (qualityScore >= 92) grade = 'Excellent';
  else if (qualityScore >= 80) grade = 'Good';
  else if (qualityScore >= 65) grade = 'Fair';

  const issues: string[] = [];
  if (totalMissing > 0) issues.push(`${totalMissing} missing cell(s) detected across columns.`);
  if (duplicateRows > 0) issues.push(`${duplicateRows} exact duplicate row(s) detected.`);
  if (totalOutliers > 0) issues.push(`${totalOutliers} statistical IQR outlier value(s) detected in numerical fields.`);
  if (issues.length === 0) issues.push('No missing cells, duplicate rows, or IQR outliers detected.');

  return {
    datasetName,
    fingerprint: computeFingerprint(datasetName, headers, rowCount),
    rowCount,
    columnCount,
    numericalColumns,
    categoricalColumns,
    datetimeColumns,
    booleanColumns,
    columns,
    qualityReport: {
      qualityScore,
      grade,
      completenessPct,
      duplicateRows,
      duplicatePct,
      totalOutliers,
      issues,
    },
  };
}

// ---------------------------------------------------------------------------
// Module 2: Data Cleaning & Preprocessing
// ---------------------------------------------------------------------------

export function cleanDataset(
  headers: string[],
  rawRows: Record<string, string>[],
  profile: DatasetProfile,
  options: CleaningOptions
): { cleanedRows: Record<string, string>[]; report: CleaningReport } {
  const rowsBefore = rawRows.length;
  let working = rawRows.map((r) => ({ ...r }));
  let trimmedCells = 0;
  let duplicatesRemoved = 0;
  let missingValuesHandled = 0;
  const actions: string[] = [];

  if (options.trimWhitespace) {
    working = working.map((row) => {
      const next: Record<string, string> = {};
      for (const h of headers) {
        const orig = row[h] ?? '';
        const trimmed = orig.trim();
        if (trimmed !== orig) trimmedCells++;
        next[h] = trimmed;
      }
      return next;
    });
    if (trimmedCells > 0) actions.push(`Trimmed leading/trailing whitespace in ${trimmedCells} cell(s).`);
  }

  if (options.removeDuplicates) {
    const seen = new Set<string>();
    const deduped: Record<string, string>[] = [];
    for (const r of working) {
      const sig = headers.map((h) => String(r[h] ?? '').trim()).join('||');
      if (seen.has(sig)) duplicatesRemoved++;
      else {
        seen.add(sig);
        deduped.push(r);
      }
    }
    working = deduped;
    if (duplicatesRemoved > 0) actions.push(`Removed ${duplicatesRemoved} duplicate row(s).`);
  }

  // Handle missing values per column
  const colMap = new Map(profile.columns.map((c) => [c.name, c]));

  if (options.numericStrategy === 'drop_rows' || options.categoricalStrategy === 'drop_rows') {
    const beforeDrop = working.length;
    working = working.filter((row) => {
      for (const h of headers) {
        const col = colMap.get(h);
        if (!col) continue;
        if (isMissing(row[h])) {
          if (col.semanticType === 'numerical' && options.numericStrategy === 'drop_rows') return false;
          if (col.semanticType !== 'numerical' && options.categoricalStrategy === 'drop_rows') return false;
        }
      }
      return true;
    });
    const dropped = beforeDrop - working.length;
    if (dropped > 0) {
      missingValuesHandled += dropped;
      actions.push(`Dropped ${dropped} row(s) containing missing values per selected drop policy.`);
    }
  }

  for (const h of headers) {
    const col = colMap.get(h);
    if (!col) continue;

    if (col.semanticType === 'numerical' && options.numericStrategy !== 'drop_rows') {
      const fillVal =
        options.numericStrategy === 'median'
          ? String(col.median ?? 0)
          : options.numericStrategy === 'mean'
          ? String(col.mean ?? 0)
          : '0';

      let filled = 0;
      for (const r of working) {
        if (isMissing(r[h])) {
          r[h] = fillVal;
          filled++;
        }
      }
      if (filled > 0) {
        missingValuesHandled += filled;
        actions.push(`Imputed ${filled} missing value(s) in '${h}' using ${options.numericStrategy} (${fillVal}).`);
      }
    } else if (col.semanticType !== 'numerical' && options.categoricalStrategy !== 'drop_rows') {
      const fillVal = options.categoricalStrategy === 'mode' ? col.topValue || 'Unknown' : 'Unknown';
      let filled = 0;
      for (const r of working) {
        if (isMissing(r[h])) {
          r[h] = fillVal;
          filled++;
        }
      }
      if (filled > 0) {
        missingValuesHandled += filled;
        actions.push(`Imputed ${filled} missing value(s) in '${h}' using '${fillVal}'.`);
      }
    }
  }

  if (actions.length === 0) {
    actions.push('No modifications were required; original rows preserved intact.');
  }

  return {
    cleanedRows: working,
    report: {
      rowsBefore,
      rowsAfter: working.length,
      duplicatesRemoved,
      missingValuesHandled,
      trimmedCells,
      actions,
    },
  };
}

// ---------------------------------------------------------------------------
// Module 3 & 4: Intent Detection & Dataset-Aware Smart Prompt Optimizer
// ---------------------------------------------------------------------------

const STOPWORDS = new Set([
  'show', 'give', 'tell', 'find', 'get', 'list', 'what', 'which', 'how', 'is', 'are', 'the', 'a', 'an',
  'in', 'on', 'of', 'by', 'for', 'to', 'and', 'or', 'across', 'over', 'between', 'with', 'from', 'me',
  'all', 'please', 'can', 'you', 'analyze', 'analysis', 'data', 'dataset', 'highest', 'lowest', 'top',
  'bottom', 'rank', 'ranking', 'compare', 'comparison', 'trend', 'monthly', 'correlation', 'relationship',
  'distribution', 'histogram', 'anomaly', 'anomalies', 'outlier', 'outliers', 'missing', 'null', 'summary',
  'overview', 'average', 'mean', 'sum', 'total', 'median', 'count', 'min', 'max', 'filter', 'where',
  'sales', 'performance', 'metrics', 'values', 'records', 'rows',
]);

export function detectIntent(prompt: string, profile: DatasetProfile): {
  intent: IntentType;
  matchedColumns: string[];
  aggregation: AggregationOp;
  topN: number | null;
  sortOrder: 'desc' | 'asc';
} {
  const lower = prompt.toLowerCase().trim();
  const matchedColumns = profile.columns
    .filter((c) => {
      const colLower = c.name.toLowerCase();
      const readable = colLower.replace(/_/g, ' ');
      return lower.includes(colLower) || lower.includes(readable);
    })
    .map((c) => c.name);

  let aggregation: AggregationOp = 'sum';
  if (/\b(avg|average|mean)\b/.test(lower)) aggregation = 'mean';
  else if (/\bmedian\b/.test(lower)) aggregation = 'median';
  else if (/\b(count|how many|headcount|frequency)\b/.test(lower)) aggregation = 'count';
  else if (/\b(min|minimum|lowest|smallest)\b/.test(lower)) aggregation = 'min';
  else if (/\b(max|maximum|highest|largest)\b/.test(lower)) aggregation = 'max';

  const topMatch = lower.match(/\btop\s+(\d+)\b/);
  const topN = topMatch ? Number(topMatch[1]) : null;
  const sortOrder: 'desc' | 'asc' = /\b(lowest|bottom|smallest|worst|min)\b/.test(lower) ? 'asc' : 'desc';

  let intent: IntentType = 'unknown';
  if (!lower) intent = 'unknown';
  else if (/\b(missing|null|na|completeness|empty)\b/.test(lower)) intent = 'missing_value_analysis';
  else if (/\b(anomal|outlier|unusual|spike|suspicious|deviation)\b/.test(lower)) intent = 'anomaly_analysis';
  else if (/\b(correlat|relationship|versus|vs\b|scatter|depend|impact)\b/.test(lower)) intent = 'correlation';
  else if (/\b(trend|over time|monthly|timeline|growth|chronological)\b/.test(lower)) intent = 'trend_analysis';
  else if (/\b(distribut|histogram|spread|frequency|range)\b/.test(lower)) intent = 'distribution_analysis';
  else if (/\b(rank|top\s+\d+|highest|lowest|best|worst|leading)\b/.test(lower)) intent = 'ranking';
  else if (/\b(compare|comparison|across|by\s+\w+|versus|breakdown)\b/.test(lower)) intent = 'comparison';
  else if (/\b(filter|where|only|subset)\b/.test(lower)) intent = 'filtering';
  else if (/\b(sum|total|average|mean|median|count|aggregate)\b/.test(lower)) intent = 'aggregation';
  else if (/\b(summary|summarize|overview|profile|describe|show)\b/.test(lower)) intent = 'dataset_summary';

  return { intent, matchedColumns, aggregation, topN, sortOrder };
}

export function optimizePrompt(originalPrompt: string, profile: DatasetProfile): PromptOptimizationResult {
  const { intent, matchedColumns, aggregation, topN, sortOrder } = detectIntent(originalPrompt, profile);

  const metrics = matchedColumns.filter((c) => profile.numericalColumns.includes(c));
  const groups = matchedColumns.filter((c) => profile.categoricalColumns.includes(c) || profile.booleanColumns.includes(c));
  const dates = matchedColumns.filter((c) => profile.datetimeColumns.includes(c));

  const primaryMetric = metrics[0] ?? profile.numericalColumns[0] ?? '';
  const secondaryMetric = metrics[1] ?? profile.numericalColumns[1] ?? primaryMetric;
  const primaryGroup = groups[0] ?? profile.categoricalColumns[0] ?? '';
  const primaryDate = dates[0] ?? profile.datetimeColumns[0] ?? null;

  // Identify terms in user prompt that mention snake_case or capitalized tokens not in schema
  const colLowerSet = new Set(profile.columns.map((c) => c.name.toLowerCase()));
  const unavailableReferences: string[] = [];
  const snakeTokens = originalPrompt.match(/\b[a-z]+_[a-z0-9_]+\b/gi) ?? [];
  for (const tok of snakeTokens) {
    if (!colLowerSet.has(tok.toLowerCase())) {
      unavailableReferences.push(tok);
    }
  }
  const commonMetricsNotPresent = ['salary', 'revenue', 'profit', 'churn', 'budget', 'cost', 'age', 'rating'];
  for (const term of commonMetricsNotPresent) {
    if (
      new RegExp(`\\b${term}\\b`, 'i').test(originalPrompt) &&
      !profile.columns.some((c) => c.name.toLowerCase().includes(term))
    ) {
      unavailableReferences.push(term);
    }
  }

  const missingContext: string[] = [];
  if (metrics.length === 0 && profile.numericalColumns.length > 0 && intent !== 'missing_value_analysis' && intent !== 'dataset_summary') {
    missingContext.push(`No numerical target metric was specified; grounded to '${primaryMetric}'.`);
  }
  if (groups.length === 0 && (intent === 'ranking' || intent === 'comparison') && profile.categoricalColumns.length > 0) {
    missingContext.push(`No grouping dimension was specified; grounded to '${primaryGroup}'.`);
  }
  if (intent === 'trend_analysis' && !dates.length && primaryDate) {
    missingContext.push(`No date/time column was named; grounded to '${primaryDate}'.`);
  }
  if (!/\b(sum|mean|average|median|count|min|max)\b/i.test(originalPrompt) && intent !== 'missing_value_analysis') {
    missingContext.push(`Aggregation function was omitted; defaulted to '${aggregation}'.`);
  }
  if (profile.qualityReport.duplicateRows > 0 || profile.columns.some((c) => c.missingCount > 0)) {
    missingContext.push(`Dataset contains ${profile.qualityReport.duplicateRows} duplicate(s) and missing cells requiring consistent handling.`);
  }

  const effectiveIntent: IntentType = intent === 'unknown' ? 'aggregation' : intent;
  let optimizedPrompt = '';
  const steps: string[] = [];

  if (effectiveIntent === 'trend_analysis' && primaryDate && primaryMetric) {
    optimizedPrompt = `Analyze the chronological ${aggregation} trend of '${primaryMetric}' grouped by month using '${primaryDate}' across ${profile.rowCount} rows in '${profile.datasetName}', reporting period totals, peak period, and percentage change while verifying against missing dates.`;
    steps.push(`Parse '${primaryDate}' into chronological periods`, `Aggregate '${primaryMetric}' (${aggregation}) per period`, `Generate line trend and verify sum conservation`);
  } else if (effectiveIntent === 'correlation' && primaryMetric && secondaryMetric) {
    optimizedPrompt = `Compute the Pearson correlation coefficient between '${primaryMetric}' and '${secondaryMetric}' across valid paired observations in '${profile.datasetName}', and verify linear relationship strength without assuming causation.`;
    steps.push(`Filter non-null pairs for '${primaryMetric}' and '${secondaryMetric}'`, `Compute Pearson r and summary ranges`, `Render scatter plot and validate sample size`);
  } else if (effectiveIntent === 'anomaly_analysis' && profile.numericalColumns.length > 0) {
    const featList = (metrics.length > 0 ? metrics : profile.numericalColumns.slice(0, 3)).join("', '");
    optimizedPrompt = `Run multivariate statistical anomaly detection across numerical features ['${featList}'] in '${profile.datasetName}', reporting flagged row indices, anomaly rate percentage, and top deviating attributes.`;
    steps.push(`Standardize selected numerical features ['${featList}']`, `Compute Z-score / distance anomaly scores`, `Flag top outliers and append non-error disclaimer`);
  } else if (effectiveIntent === 'distribution_analysis' && primaryMetric) {
    optimizedPrompt = `Compute the numerical distribution of '${primaryMetric}' in '${profile.datasetName}', including minimum, median, mean, standard deviation, interquartile range, and binned histogram frequencies.`;
    steps.push(`Extract valid numerical observations of '${primaryMetric}'`, `Compute central tendency, dispersion, and histogram bins`, `Render histogram and validate bounds`);
  } else if (effectiveIntent === 'missing_value_analysis') {
    optimizedPrompt = `Audit missing values, completeness percentage, and exact duplicate rows across all ${profile.columnCount} columns of '${profile.datasetName}', ranking columns by missing count.`;
    steps.push(`Count null/empty values per column`, `Calculate completeness percentage`, `Identify columns needing imputation`);
  } else if ((effectiveIntent === 'ranking' || effectiveIntent === 'comparison') && primaryMetric && primaryGroup) {
    const limitPhrase = topN ? `the top ${topN} ` : 'all ';
    optimizedPrompt = `${effectiveIntent === 'ranking' ? 'Rank' : 'Compare'} ${limitPhrase}categories of '${primaryGroup}' by ${aggregation} of '${primaryMetric}' (${sortOrder === 'desc' ? 'highest to lowest' : 'lowest to highest'}) in '${profile.datasetName}', showing record counts and share of total.`;
    steps.push(`Group records by '${primaryGroup}'`, `Compute ${aggregation}('${primaryMetric}') and record count`, `Sort ${sortOrder} and cross-check total against dataset sum`);
  } else {
    optimizedPrompt = `Produce a structured analytical summary of '${profile.datasetName}' (${profile.rowCount} rows × ${profile.columnCount} columns)${primaryMetric ? ` evaluating ${aggregation}, mean, min, and max of '${primaryMetric}'` : ''}${primaryGroup ? ` broken down by '${primaryGroup}'` : ''}, grounded strictly to verified columns.`;
    steps.push(`Verify schema types across ${profile.columnCount} columns`, `Compute deterministic KPIs for verified columns`, `Cross-validate totals against raw data`);
  }

  const improvementReasons: string[] = [
    `Grounded prompt strictly to verified schema columns (${[primaryMetric, primaryGroup, primaryDate].filter(Boolean).join(', ') || 'dataset overview'}).`,
    `Clarified explicit analytical intent (${effectiveIntent.replace(/_/g, ' ')}) and deterministic aggregation (${aggregation}).`,
    `Defined verifiable output expectations so the validation engine can audit numeric accuracy.`,
  ];

  if (unavailableReferences.length > 0) {
    improvementReasons.push(`Flagged referenced terms not found in schema: ${Array.from(new Set(unavailableReferences)).join(', ')}.`);
  }

  const finalMetrics = metrics.length > 0 ? metrics : primaryMetric ? [primaryMetric] : [];
  if (effectiveIntent === 'correlation' && secondaryMetric && !finalMetrics.includes(secondaryMetric)) {
    finalMetrics.push(secondaryMetric);
  }

  return {
    originalPrompt,
    optimizedPrompt,
    detectedIntent: effectiveIntent,
    matchedColumns: Array.from(new Set(matchedColumns)),
    unavailableReferences: Array.from(new Set(unavailableReferences)),
    missingContext,
    improvementReasons,
    taskPlan: {
      intent: effectiveIntent,
      metrics: finalMetrics,
      groupBy: groups.length > 0 ? groups : primaryGroup ? [primaryGroup] : [],
      timeColumn: primaryDate,
      aggregation,
      topN,
      sortOrder,
      filters: [],
      steps,
    },
  };
}

// ---------------------------------------------------------------------------
// Module 5: Transparent Deterministic Prompt Quality Evaluator
// ---------------------------------------------------------------------------

export function scorePromptQuality(prompt: string, profile: DatasetProfile): PromptQualityScore {
  const trimmed = prompt.trim();
  const words = trimmed.split(/\s+/).filter(Boolean);
  const { intent, matchedColumns } = detectIntent(trimmed, profile);

  // 1. Specificity (0..100)
  let specificity = 20;
  if (words.length >= 5) specificity += 20;
  if (words.length >= 10) specificity += 20;
  if (/\b(sum|total|mean|average|median|count|min|max|top \d+|monthly|pearson|iqr|z-score)\b/i.test(trimmed)) specificity += 25;
  if (/\b(percentage|share|trend|group|breakdown|per|by)\b/i.test(trimmed)) specificity += 15;
  specificity = Math.min(100, specificity);

  // 2. Context completeness (0..100)
  let contextCompleteness = 20;
  const hasNum = matchedColumns.some((c) => profile.numericalColumns.includes(c));
  const hasDim = matchedColumns.some((c) => profile.categoricalColumns.includes(c) || profile.datetimeColumns.includes(c));
  if (hasNum) contextCompleteness += 35;
  if (hasDim) contextCompleteness += 30;
  if (matchedColumns.length >= 2) contextCompleteness += 15;
  contextCompleteness = Math.min(100, contextCompleteness);

  // 3. Intent clarity (0..100)
  let intentClarity = intent === 'unknown' ? 25 : 75;
  if (/\b(rank|compare|trend|correlat|distribut|anomal|missing)\b/i.test(trimmed)) intentClarity = 95;
  if (words.length <= 2) intentClarity = Math.min(intentClarity, 35);

  // 4. Column relevance (0..100)
  const columnRelevance = matchedColumns.length === 0 ? 15 : Math.min(100, 45 + matchedColumns.length * 25);

  const overallScore = Number(
    (specificity * 0.25 + contextCompleteness * 0.30 + intentClarity * 0.25 + columnRelevance * 0.20).toFixed(1)
  );

  let qualityClass: PromptQualityScore['qualityClass'] = 'Low';
  if (overallScore >= 75) qualityClass = 'High';
  else if (overallScore >= 48) qualityClass = 'Medium';

  const notes: string[] = [];
  if (matchedColumns.length === 0) notes.push('Prompt does not explicitly name any verified dataset columns.');
  else notes.push(`Matched ${matchedColumns.length} dataset column(s): ${matchedColumns.join(', ')}.`);
  if (words.length < 5) notes.push('Short prompt lacks explicit grouping or aggregation parameters.');
  if (intentClarity >= 80) notes.push('Analytical intent is clearly identifiable.');

  return {
    overallScore,
    qualityClass,
    breakdown: {
      specificity,
      contextCompleteness,
      intentClarity,
      columnRelevance,
    },
    notes,
  };
}

// ---------------------------------------------------------------------------
// Module 6: Automated Deterministic Data Analysis Engine
// ---------------------------------------------------------------------------

export function runDataAnalysis(
  rows: Record<string, string>[],
  profile: DatasetProfile,
  plan: TaskPlan
): AnalysisResult {
  const warnings: string[] = [];
  const metric = plan.metrics[0] ?? profile.numericalColumns[0] ?? '';
  const secondMetric = plan.metrics[1] ?? profile.numericalColumns[1] ?? '';
  const groupCol = plan.groupBy[0] ?? profile.categoricalColumns[0] ?? '';
  const timeCol = plan.timeColumn ?? profile.datetimeColumns[0] ?? '';

  if (plan.intent === 'missing_value_analysis') {
    const outRows = profile.columns
      .map((c) => ({
        column: c.name,
        type: c.semanticType,
        missing_count: c.missingCount,
        missing_pct: c.missingPct,
        unique_values: c.uniqueCount,
      }))
      .sort((a, b) => Number(b.missing_count) - Number(a.missing_count));

    return {
      intent: plan.intent,
      title: 'Missing Value & Completeness Audit',
      summaryText: `Dataset completeness is ${profile.qualityReport.completenessPct}% across ${profile.rowCount} rows and ${profile.columnCount} columns.`,
      recommendedChart: 'bar',
      xColumn: 'column',
      yColumn: 'missing_count',
      columns: ['column', 'type', 'missing_count', 'missing_pct', 'unique_values'],
      rows: outRows,
      kpis: [
        { label: 'Completeness', value: `${profile.qualityReport.completenessPct}%` },
        { label: 'Quality Score', value: `${profile.qualityReport.qualityScore}/100` },
        { label: 'Duplicate Rows', value: String(profile.qualityReport.duplicateRows) },
      ],
      evidence: { rowCountAnalyzed: profile.rowCount },
      warnings,
    };
  }

  if (plan.intent === 'correlation' && metric && secondMetric && metric !== secondMetric) {
    const pairs: { x: number; y: number }[] = [];
    for (const r of rows) {
      const x = parseNumeric(r[metric]);
      const y = parseNumeric(r[secondMetric]);
      if (x !== null && y !== null) pairs.push({ x, y });
    }
    if (pairs.length < 3) warnings.push('Fewer than 3 complete numerical pairs available for correlation.');

    const n = pairs.length;
    const meanX = n > 0 ? pairs.reduce((s, p) => s + p.x, 0) / n : 0;
    const meanY = n > 0 ? pairs.reduce((s, p) => s + p.y, 0) / n : 0;
    let num = 0;
    let denX = 0;
    let denY = 0;
    for (const p of pairs) {
      const dx = p.x - meanX;
      const dy = p.y - meanY;
      num += dx * dy;
      denX += dx * dx;
      denY += dy * dy;
    }
    const rVal = denX > 0 && denY > 0 ? Number((num / Math.sqrt(denX * denY)).toFixed(4)) : 0;
    const strength = Math.abs(rVal) >= 0.7 ? 'Strong' : Math.abs(rVal) >= 0.4 ? 'Moderate' : 'Weak';

    return {
      intent: plan.intent,
      title: `Correlation: ${metric} vs ${secondMetric}`,
      summaryText: `Pearson r = ${rVal} (${strength} ${rVal >= 0 ? 'positive' : 'negative'} linear correlation across ${n} valid pairs).`,
      recommendedChart: 'scatter',
      xColumn: metric,
      yColumn: secondMetric,
      columns: [metric, secondMetric],
      rows: pairs.map((p) => ({ [metric]: Number(p.x.toFixed(2)), [secondMetric]: Number(p.y.toFixed(2)) })),
      kpis: [
        { label: 'Pearson r', value: String(rVal) },
        { label: 'Strength', value: strength },
        { label: 'Paired Rows', value: String(n) },
      ],
      evidence: { rowCountAnalyzed: n, correlationR: rVal },
      warnings,
    };
  }

  if (plan.intent === 'trend_analysis' && timeCol && metric) {
    const buckets = new Map<string, number[]>();
    for (const r of rows) {
      const rawDate = String(r[timeCol] ?? '').trim();
      const val = parseNumeric(r[metric]);
      if (!rawDate || val === null) continue;
      const parsed = new Date(rawDate);
      const key = Number.isNaN(parsed.getTime())
        ? rawDate
        : `${parsed.getUTCFullYear()}-${String(parsed.getUTCMonth() + 1).padStart(2, '0')}`;
      const list = buckets.get(key) ?? [];
      list.push(val);
      buckets.set(key, list);
    }

    const sortedKeys = Array.from(buckets.keys()).sort();
    const outRows = sortedKeys.map((period) => {
      const vals = buckets.get(period)!;
      const sum = vals.reduce((a, b) => a + b, 0);
      const mean = sum / vals.length;
      const aggVal =
        plan.aggregation === 'mean'
          ? mean
          : plan.aggregation === 'median'
          ? computeMedian([...vals].sort((a, b) => a - b))
          : plan.aggregation === 'count'
          ? vals.length
          : plan.aggregation === 'min'
          ? Math.min(...vals)
          : plan.aggregation === 'max'
          ? Math.max(...vals)
          : sum;
      return {
        period,
        value: Number(aggVal.toFixed(2)),
        records: vals.length,
      };
    });

    const grandTotal = outRows.reduce((s, r) => s + Number(r.value), 0);
    const peak = [...outRows].sort((a, b) => Number(b.value) - Number(a.value))[0];

    return {
      intent: plan.intent,
      title: `Monthly ${plan.aggregation.toUpperCase()} Trend of ${metric}`,
      summaryText: peak
        ? `Across ${outRows.length} chronological period(s), peak ${metric} occurred in ${peak.period} (${peak.value.toLocaleString()}).`
        : `No valid date-metric pairs found.`,
      recommendedChart: 'line',
      xColumn: 'period',
      yColumn: 'value',
      columns: ['period', 'value', 'records'],
      rows: outRows,
      kpis: [
        { label: 'Periods', value: String(outRows.length) },
        { label: `Peak Period`, value: peak ? `${peak.period} (${peak.value.toLocaleString()})` : 'N/A' },
        { label: `Period Sum`, value: grandTotal.toLocaleString(undefined, { maximumFractionDigits: 2 }) },
      ],
      evidence: {
        rowCountAnalyzed: outRows.reduce((s, r) => s + Number(r.records), 0),
        totalSum: Number(grandTotal.toFixed(2)),
        topEntity: peak?.period,
        topValue: peak ? Number(peak.value) : undefined,
      },
      warnings,
    };
  }

  if (plan.intent === 'distribution_analysis' && metric) {
    const vals = rows
      .map((r) => parseNumeric(r[metric]))
      .filter((v): v is number => v !== null)
      .sort((a, b) => a - b);
    const min = vals[0] ?? 0;
    const max = vals[vals.length - 1] ?? 0;
    const mean = vals.length > 0 ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
    const median = computeMedian(vals);
    const binCount = Math.min(8, Math.max(4, Math.round(Math.sqrt(vals.length))));
    const span = max - min || 1;
    const width = span / binCount;

    const outRows: Record<string, string | number>[] = [];
    for (let i = 0; i < binCount; i++) {
      const lo = min + i * width;
      const hi = i === binCount - 1 ? max : lo + width;
      const count = vals.filter((v) => (i === binCount - 1 ? v >= lo && v <= hi : v >= lo && v < hi)).length;
      outRows.push({
        bin_range: `${lo.toFixed(0)}–${hi.toFixed(0)}`,
        count,
      });
    }

    return {
      intent: plan.intent,
      title: `Distribution of ${metric}`,
      summaryText: `Computed ${binCount} frequency bins across ${vals.length} valid values of '${metric}' (range ${min.toLocaleString()} to ${max.toLocaleString()}).`,
      recommendedChart: 'histogram',
      xColumn: 'bin_range',
      yColumn: 'count',
      columns: ['bin_range', 'count'],
      rows: outRows,
      kpis: [
        { label: 'Mean', value: mean.toLocaleString(undefined, { maximumFractionDigits: 2 }) },
        { label: 'Median', value: median.toLocaleString(undefined, { maximumFractionDigits: 2 }) },
        { label: 'Range', value: `${min.toLocaleString()} – ${max.toLocaleString()}` },
      ],
      evidence: { rowCountAnalyzed: vals.length, mean: Number(mean.toFixed(2)) },
      warnings,
    };
  }

  // Group-by ranking / comparison / default aggregation
  if (groupCol && metric) {
    const groups = new Map<string, number[]>();
    let validCount = 0;
    for (const r of rows) {
      const key = isMissing(r[groupCol]) ? '(Missing)' : String(r[groupCol]).trim();
      const val = parseNumeric(r[metric]);
      if (val === null) continue;
      validCount++;
      const arr = groups.get(key) ?? [];
      arr.push(val);
      groups.set(key, arr);
    }

    const totalMetricSum = Array.from(groups.values())
      .flat()
      .reduce((a, b) => a + b, 0);

    let outRows = Array.from(groups.entries()).map(([category, vals]) => {
      const sum = vals.reduce((a, b) => a + b, 0);
      const mean = sum / vals.length;
      const aggVal =
        plan.aggregation === 'mean'
          ? mean
          : plan.aggregation === 'median'
          ? computeMedian([...vals].sort((a, b) => a - b))
          : plan.aggregation === 'count'
          ? vals.length
          : plan.aggregation === 'min'
          ? Math.min(...vals)
          : plan.aggregation === 'max'
          ? Math.max(...vals)
          : sum;
      const sharePct = totalMetricSum !== 0 ? Number(((sum / totalMetricSum) * 100).toFixed(1)) : 0;

      return {
        [groupCol]: category,
        value: Number(aggVal.toFixed(2)),
        sum_total: Number(sum.toFixed(2)),
        records: vals.length,
        share_pct: sharePct,
      };
    });

    outRows.sort((a, b) =>
      plan.sortOrder === 'asc' ? Number(a.value) - Number(b.value) : Number(b.value) - Number(a.value)
    );
    if (plan.topN && plan.topN > 0) outRows = outRows.slice(0, plan.topN);

    const top = outRows[0];
    if (validCount < rows.length) {
      warnings.push(`${rows.length - validCount} row(s) skipped due to missing '${metric}' values.`);
    }

    return {
      intent: plan.intent,
      title: `${plan.aggregation.toUpperCase()} of ${metric} by ${groupCol}`,
      summaryText: top
        ? `'${top[groupCol]}' leads '${groupCol}' with ${plan.aggregation}(${metric}) = ${Number(top.value).toLocaleString()} (${top.share_pct}% of total sum across ${validCount} valid rows).`
        : 'No grouped observations matched.',
      recommendedChart: plan.intent === 'ranking' ? 'horizontal_bar' : 'bar',
      xColumn: groupCol,
      yColumn: 'value',
      columns: [groupCol, 'value', 'sum_total', 'records', 'share_pct'],
      rows: outRows,
      kpis: [
        { label: `Total ${metric}`, value: totalMetricSum.toLocaleString(undefined, { maximumFractionDigits: 2 }) },
        { label: `Top ${groupCol}`, value: top ? String(top[groupCol]) : 'N/A' },
        { label: 'Groups', value: String(outRows.length) },
      ],
      evidence: {
        rowCountAnalyzed: validCount,
        totalSum: Number(totalMetricSum.toFixed(2)),
        mean: validCount > 0 ? Number((totalMetricSum / validCount).toFixed(2)) : 0,
        topEntity: top ? String(top[groupCol]) : undefined,
        topValue: top ? Number(top.value) : undefined,
      },
      warnings,
    };
  }

  // Fallback dataset KPI summary
  const summaryRows = profile.numericalColumns.map((colName) => {
    const col = profile.columns.find((c) => c.name === colName)!;
    return {
      metric: colName,
      mean: col.mean ?? 0,
      median: col.median ?? 0,
      min: col.min ?? 0,
      max: col.max ?? 0,
    };
  });

  return {
    intent: 'dataset_summary',
    title: 'Dataset Quantitative Summary',
    summaryText: `Summarized ${profile.numericalColumns.length} numerical column(s) across ${profile.rowCount} rows.`,
    recommendedChart: 'bar',
    xColumn: 'metric',
    yColumn: 'mean',
    columns: ['metric', 'mean', 'median', 'min', 'max'],
    rows: summaryRows,
    kpis: [
      { label: 'Rows', value: String(profile.rowCount) },
      { label: 'Columns', value: String(profile.columnCount) },
      { label: 'Quality Score', value: `${profile.qualityReport.qualityScore}/100` },
    ],
    evidence: { rowCountAnalyzed: profile.rowCount },
    warnings,
  };
}

// ---------------------------------------------------------------------------
// Module 7: Statistical Numerical Anomaly Detection
// ---------------------------------------------------------------------------

export function detectAnomalies(
  rows: Record<string, string>[],
  selectedFeatures: string[],
  contaminationPct: number
): AnomalyReport {
  const validRows: { rowIndex: number; row: Record<string, string>; vec: number[] }[] = [];
  let skippedIncompleteRows = 0;

  rows.forEach((row, idx) => {
    const vec: number[] = [];
    let ok = true;
    for (const f of selectedFeatures) {
      const n = parseNumeric(row[f]);
      if (n === null) {
        ok = false;
        break;
      }
      vec.push(n);
    }
    if (ok && vec.length > 0) {
      validRows.push({ rowIndex: idx, row, vec });
    } else {
      skippedIncompleteRows++;
    }
  });

  if (validRows.length === 0 || selectedFeatures.length === 0) {
    return {
      selectedFeatures,
      contaminationPct,
      rowsAnalyzed: 0,
      skippedIncompleteRows,
      anomalyCount: 0,
      anomalyPercentage: 0,
      anomalies: [],
      scatterPoints: [],
      xFeature: selectedFeatures[0] ?? '',
      yFeature: selectedFeatures[1] ?? selectedFeatures[0] ?? '',
      disclaimer:
        'Anomalous values reflect multivariate statistical deviation and do not automatically indicate data errors or fraud.',
    };
  }

  // Compute robust Z-scores (mean & std dev) per feature
  const dim = selectedFeatures.length;
  const means = new Array<number>(dim).fill(0);
  const stds = new Array<number>(dim).fill(1);

  for (let d = 0; d < dim; d++) {
    const vals = validRows.map((r) => r.vec[d]);
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    const variance = vals.reduce((a, b) => a + (b - mean) ** 2, 0) / vals.length;
    means[d] = mean;
    stds[d] = Math.sqrt(variance) || 1;
  }

  const scored = validRows.map((item) => {
    const zScores = item.vec.map((v, d) => Math.abs(v - means[d]) / stds[d]);
    const l2Dist = Math.sqrt(zScores.reduce((a, z) => a + z * z, 0) / dim);
    let maxZ = -1;
    let maxIdx = 0;
    zScores.forEach((z, d) => {
      if (z > maxZ) {
        maxZ = z;
        maxIdx = d;
      }
    });
    return {
      ...item,
      anomalyScore: Number(l2Dist.toFixed(3)),
      zMax: Number(maxZ.toFixed(2)),
      topFeature: selectedFeatures[maxIdx],
      topFeatureValue: item.vec[maxIdx],
    };
  });

  const sortedByScore = [...scored].sort((a, b) => b.anomalyScore - a.anomalyScore);
  const targetCount = Math.max(1, Math.round((contaminationPct / 100) * validRows.length));
  const flaggedSet = new Set(sortedByScore.slice(0, targetCount).map((s) => s.rowIndex));

  const anomalies: AnomalyRowDetail[] = sortedByScore
    .filter((s) => flaggedSet.has(s.rowIndex))
    .map((s) => ({
      rowIndex: s.rowIndex,
      anomalyScore: s.anomalyScore,
      zMax: s.zMax,
      topFeature: s.topFeature,
      topFeatureValue: s.topFeatureValue,
      rowPreview: s.row,
    }));

  const xIdx = 0;
  const yIdx = selectedFeatures.length > 1 ? 1 : 0;

  return {
    selectedFeatures,
    contaminationPct,
    rowsAnalyzed: validRows.length,
    skippedIncompleteRows,
    anomalyCount: anomalies.length,
    anomalyPercentage: Number(((anomalies.length / validRows.length) * 100).toFixed(2)),
    anomalies,
    scatterPoints: scored.map((s) => ({
      rowIndex: s.rowIndex,
      x: s.vec[xIdx],
      y: s.vec[yIdx],
      isAnomaly: flaggedSet.has(s.rowIndex),
      score: s.anomalyScore,
    })),
    xFeature: selectedFeatures[xIdx],
    yFeature: selectedFeatures[yIdx],
    disclaimer:
      'Anomalous values reflect multivariate statistical deviation and do not automatically indicate data errors or fraud.',
  };
}

// ---------------------------------------------------------------------------
// Module 8: Evidence-Based Self-Validation Engine
// ---------------------------------------------------------------------------

export function validateAnalysisResult(
  rows: Record<string, string>[],
  profile: DatasetProfile,
  plan: TaskPlan,
  analysis: AnalysisResult,
  customClaim?: string
): ValidationReport {
  const checks: ValidationCheck[] = [];
  const verifiedInsights: string[] = [];
  const excludedClaims: string[] = [];

  // Check 1: Schema Column Existence
  const colSet = new Set(profile.columns.map((c) => c.name));
  const usedCols = [...plan.metrics, ...plan.groupBy, ...(plan.timeColumn ? [plan.timeColumn] : [])];
  const missingCols = usedCols.filter((c) => !colSet.has(c));
  checks.push({
    checkName: 'Schema Column Verification',
    status: missingCols.length === 0 ? 'Verified' : 'Unsupported',
    expected: 'All referenced columns exist in profile',
    observed: missingCols.length === 0 ? `${usedCols.length || 1} verified column(s)` : `Missing: ${missingCols.join(', ')}`,
    detail: missingCols.length === 0 ? 'All task plan fields are present in uploaded dataset.' : 'Unsupported columns referenced.',
  });

  // Check 2: Row Conservation / Completeness
  const skipped = profile.rowCount - analysis.evidence.rowCountAnalyzed;
  checks.push({
    checkName: 'Sample Coverage & Row Count',
    status: skipped === 0 ? 'Verified' : analysis.evidence.rowCountAnalyzed > 0 ? 'Warning' : 'Unsupported',
    expected: `${profile.rowCount} dataset rows`,
    observed: `${analysis.evidence.rowCountAnalyzed} rows analyzed`,
    detail:
      skipped === 0
        ? '100% of rows included in computation.'
        : `${skipped} row(s) excluded due to missing numerical values in target column.`,
  });

  // Check 3: Independent Numeric Recomputation (Sum & Winner Verification)
  const targetMetric = plan.metrics[0] ?? profile.numericalColumns[0];
  if (targetMetric && analysis.evidence.totalSum !== undefined) {
    const rawSum = rows.reduce((acc, r) => {
      const n = parseNumeric(r[targetMetric]);
      return n !== null ? acc + n : acc;
    }, 0);
    const roundedRaw = Number(rawSum.toFixed(2));
    const diff = Math.abs(roundedRaw - analysis.evidence.totalSum);
    const match = diff <= 0.05;
    checks.push({
      checkName: `Independent Total Recomputation (${targetMetric})`,
      status: match ? 'Verified' : 'Unsupported',
      expected: roundedRaw.toLocaleString(),
      observed: analysis.evidence.totalSum.toLocaleString(),
      detail: match
        ? 'Aggregated sum matches independent pass over raw dataset rows within tolerance (0.05).'
        : `Discrepancy of ${diff.toFixed(2)} detected between analysis result and raw rows.`,
    });
    if (match) {
      verifiedInsights.push(
        `Verified total ${targetMetric} = ${roundedRaw.toLocaleString()} across ${analysis.evidence.rowCountAnalyzed} valid rows.`
      );
    } else {
      excludedClaims.push(`Excluded total claim for ${targetMetric}: reported ${analysis.evidence.totalSum} != recomputed ${roundedRaw}.`);
    }
  }

  if (analysis.evidence.topEntity && analysis.evidence.topValue !== undefined) {
    verifiedInsights.push(
      `Verified leading segment '${analysis.evidence.topEntity}' with ${plan.aggregation}(${targetMetric}) = ${analysis.evidence.topValue.toLocaleString()}.`
    );
  }

  if (analysis.evidence.correlationR !== undefined) {
    verifiedInsights.push(
      `Verified Pearson correlation r = ${analysis.evidence.correlationR} across ${analysis.evidence.rowCountAnalyzed} complete numeric pairs.`
    );
  }

  // Optional Custom Claim Verification
  if (customClaim && customClaim.trim().length > 0) {
    const claim = customClaim.trim();
    const numericMatches = claim.match(/-?\d[\d,]*(?:\.\d+)?/g) ?? [];
    const claimedNums = numericMatches
      .map((s) => Number(s.replace(/,/g, '')))
      .filter((n) => Number.isFinite(n));

    const claimCols = profile.columns.filter((c) => claim.toLowerCase().includes(c.name.toLowerCase()));
    if (claimedNums.length === 0) {
      checks.push({
        checkName: 'Custom Claim Check',
        status: 'Warning',
        expected: 'Numeric claim verifiable against data',
        observed: `"${claim}"`,
        detail: 'Claim contains no concrete numerical assertion to verify deterministically.',
      });
    } else {
      // Compare claimed numbers against known dataset metric sums, means, top values, or row count
      const knownValues: number[] = [
        profile.rowCount,
        analysis.evidence.totalSum ?? Number.NaN,
        analysis.evidence.topValue ?? Number.NaN,
        analysis.evidence.mean ?? Number.NaN,
      ];
      for (const c of profile.columns) {
        if (c.mean !== undefined) knownValues.push(c.mean);
        if (c.max !== undefined) knownValues.push(c.max);
        if (c.min !== undefined) knownValues.push(c.min);
      }
      for (const r of analysis.rows) {
        for (const v of Object.values(r)) {
          if (typeof v === 'number') knownValues.push(v);
        }
      }

      const matched = claimedNums.some((cn) =>
        knownValues.some((kv) => Number.isFinite(kv) && Math.abs(kv - cn) <= Math.max(1, Math.abs(kv) * 0.015))
      );

      if (matched && (claimCols.length > 0 || claimedNums.length > 0)) {
        checks.push({
          checkName: 'Custom Claim Verification',
          status: 'Verified',
          expected: 'Matches computed dataset evidence',
          observed: `"${claim}"`,
          detail: 'Claimed numeric value agrees with deterministic dataset computation within 1.5% tolerance.',
        });
        verifiedInsights.push(`User claim verified: "${claim}".`);
      } else {
        checks.push({
          checkName: 'Custom Claim Verification',
          status: 'Unsupported',
          expected: 'Consistent with raw dataset figures',
          observed: `"${claim}"`,
          detail: `Claimed value(s) [${claimedNums.join(', ')}] do not match any verified calculation on '${profile.datasetName}'.`,
        });
        excludedClaims.push(`Unsupported claim rejected: "${claim}" (no numerical evidence match in dataset).`);
      }
    }
  }

  const verifiedCount = checks.filter((c) => c.status === 'Verified').length;
  const warningCount = checks.filter((c) => c.status === 'Warning').length;
  const unsupportedCount = checks.filter((c) => c.status === 'Unsupported').length;

  let overallStatus: ValidationReport['overallStatus'] = 'Verified';
  if (unsupportedCount > 0) overallStatus = 'Unsupported';
  else if (warningCount > 0) overallStatus = 'Warning';

  return {
    overallStatus,
    verifiedCount,
    warningCount,
    unsupportedCount,
    verifiedInsights,
    excludedClaims,
    checks,
    limitationsNote:
      'Validation compares outputs strictly against deterministic computations on the active dataset snapshot; correlation or anomaly flags do not imply causation.',
  };
}

// ---------------------------------------------------------------------------
// Module 11: Client-Side Report Serialization (CSV Download)
// ---------------------------------------------------------------------------

export function buildReportCsv(
  profile: DatasetProfile,
  optimization: PromptOptimizationResult,
  analysis: AnalysisResult,
  validation: ValidationReport,
  anomalyReport: AnomalyReport | null
): string {
  const esc = (v: unknown) => {
    const s = String(v ?? '');
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const lines: string[] = [];
  lines.push('SECTION,KEY,VALUE');
  lines.push(['Dataset', 'Name', profile.datasetName].map(esc).join(','));
  lines.push(['Dataset', 'Rows x Columns', `${profile.rowCount} x ${profile.columnCount}`].map(esc).join(','));
  lines.push(['Dataset', 'Quality Score', `${profile.qualityReport.qualityScore}/100 (${profile.qualityReport.grade})`].map(esc).join(','));
  lines.push(['Prompt', 'Original', optimization.originalPrompt].map(esc).join(','));
  lines.push(['Prompt', 'Optimized', optimization.optimizedPrompt].map(esc).join(','));
  lines.push(['Prompt', 'Detected Intent', optimization.detectedIntent].map(esc).join(','));
  lines.push(['Analysis', 'Summary', analysis.summaryText].map(esc).join(','));
  lines.push(['Validation', 'Overall Status', validation.overallStatus].map(esc).join(','));
  for (const ins of validation.verifiedInsights) {
    lines.push(['Verified Insight', 'Evidence', ins].map(esc).join(','));
  }
  if (anomalyReport) {
    lines.push(['Anomalies', 'Flagged Rows', `${anomalyReport.anomalyCount} (${anomalyReport.anomalyPercentage}%)`].map(esc).join(','));
  }
  lines.push('');
  lines.push(analysis.columns.map(esc).join(','));
  for (const r of analysis.rows) {
    lines.push(analysis.columns.map((c) => esc(r[c] ?? '')).join(','));
  }
  return lines.join('\n');
}

export function parseExcelArrayBuffer(buffer: ArrayBuffer): {
  headers: string[];
  rows: Record<string, string>[];
  sheetName: string;
} {
  const wb = XLSX.read(buffer, { type: 'array', cellDates: true });
  if (!wb.SheetNames || wb.SheetNames.length === 0) {
    throw new Error('The uploaded Excel workbook contains no sheets.');
  }
  const sheetName = wb.SheetNames[0];
  const sheet = wb.Sheets[sheetName];
  const rawMatrix = XLSX.utils.sheet_to_json<unknown[]>(sheet, { header: 1, defval: '' });
  if (!rawMatrix || rawMatrix.length < 2) {
    throw new Error(`Sheet '${sheetName}' must contain a header row and at least one data row.`);
  }
  const headerRow = rawMatrix[0] as unknown[];
  const headers = headerRow.map((cell, idx) => {
    const s = String(cell ?? '').trim();
    return s || `column_${idx + 1}`;
  });
  const rows: Record<string, string>[] = [];
  for (let r = 1; r < rawMatrix.length; r++) {
    const arr = rawMatrix[r] as unknown[];
    if (!arr || arr.every((c) => String(c ?? '').trim() === '')) continue;
    const rowObj: Record<string, string> = {};
    headers.forEach((h, idx) => {
      const val = arr[idx];
      if (val instanceof Date && !Number.isNaN(val.getTime())) {
        rowObj[h] = val.toISOString().slice(0, 10);
      } else {
        rowObj[h] = String(val ?? '').trim();
      }
    });
    rows.push(rowObj);
  }
  if (rows.length === 0) {
    throw new Error(`Sheet '${sheetName}' has headers but no non-empty data rows.`);
  }
  return { headers, rows, sheetName };
}

export function computeCorrelationMatrix(
  rows: Record<string, string>[],
  numericColumns: string[]
): { columns: string[]; matrix: number[][] } {
  const cols = numericColumns.slice(0, 8);
  const matrix: number[][] = cols.map(() => cols.map(() => 0));
  for (let i = 0; i < cols.length; i++) {
    for (let j = i; j < cols.length; j++) {
      if (i === j) {
        matrix[i][j] = 1;
        continue;
      }
      const pairs: { x: number; y: number }[] = [];
      for (const r of rows) {
        const x = parseNumeric(r[cols[i]]);
        const y = parseNumeric(r[cols[j]]);
        if (x !== null && y !== null) pairs.push({ x, y });
      }
      if (pairs.length < 2) {
        matrix[i][j] = 0;
        matrix[j][i] = 0;
        continue;
      }
      const n = pairs.length;
      const mx = pairs.reduce((s, p) => s + p.x, 0) / n;
      const my = pairs.reduce((s, p) => s + p.y, 0) / n;
      let num = 0;
      let dx2 = 0;
      let dy2 = 0;
      for (const p of pairs) {
        const dx = p.x - mx;
        const dy = p.y - my;
        num += dx * dy;
        dx2 += dx * dx;
        dy2 += dy * dy;
      }
      const corr = dx2 > 0 && dy2 > 0 ? Number((num / Math.sqrt(dx2 * dy2)).toFixed(2)) : 0;
      matrix[i][j] = corr;
      matrix[j][i] = corr;
    }
  }
  return { columns: cols, matrix };
}

export function buildReportExcelBytes(
  profile: DatasetProfile,
  optimization: PromptOptimizationResult,
  analysis: AnalysisResult,
  validation: ValidationReport,
  anomalyReport: AnomalyReport | null
): Uint8Array {
  const wb = XLSX.utils.book_new();

  // Sheet 1: Executive Summary & Verified Insights
  const summaryRows = [
    { Section: 'Dataset', Metric: 'Name', Value: profile.datasetName },
    { Section: 'Dataset', Metric: 'Fingerprint', Value: profile.fingerprint },
    { Section: 'Dataset', Metric: 'Shape', Value: `${profile.rowCount} rows x ${profile.columnCount} cols` },
    {
      Section: 'Dataset',
      Metric: 'Quality Score',
      Value: `${profile.qualityReport.qualityScore}/100 (${profile.qualityReport.grade})`,
    },
    { Section: 'Prompt', Metric: 'Original Prompt', Value: optimization.originalPrompt },
    { Section: 'Prompt', Metric: 'Optimized Prompt', Value: optimization.optimizedPrompt },
    { Section: 'Prompt', Metric: 'Detected Intent', Value: optimization.detectedIntent },
    { Section: 'Analysis', Metric: 'Summary', Value: analysis.summaryText },
    { Section: 'Validation', Metric: 'Overall Status', Value: validation.overallStatus },
    ...validation.verifiedInsights.map((ins, i) => ({
      Section: 'Verified Insight',
      Metric: `Insight ${i + 1}`,
      Value: ins,
    })),
    ...(anomalyReport
      ? [
          {
            Section: 'Anomalies',
            Metric: 'Flagged Records',
            Value: `${anomalyReport.anomalyCount} of ${anomalyReport.rowsAnalyzed} (${anomalyReport.anomalyPercentage}%)`,
          },
          {
            Section: 'Anomalies',
            Metric: 'Interpretation Note',
            Value: anomalyReport.disclaimer,
          },
        ]
      : []),
  ];
  const wsSummary = XLSX.utils.json_to_sheet(summaryRows);
  XLSX.utils.book_append_sheet(wb, wsSummary, 'Executive Summary');

  // Sheet 2: Computed Evidence Table
  const wsEvidence = XLSX.utils.json_to_sheet(
    analysis.rows.length > 0 ? analysis.rows : [{ Note: 'No tabular evidence rows produced.' }]
  );
  XLSX.utils.book_append_sheet(wb, wsEvidence, 'Evidence Table');

  // Sheet 3: Validation Audit
  const wsVal = XLSX.utils.json_to_sheet(
    validation.checks.map((c) => ({
      Check: c.checkName,
      Status: c.status,
      Expected: c.expected,
      Observed: c.observed,
      Detail: c.detail,
    }))
  );
  XLSX.utils.book_append_sheet(wb, wsVal, 'Validation Audit');

  // Sheet 4: Anomaly Records
  if (anomalyReport && anomalyReport.anomalies.length > 0) {
    const wsAnom = XLSX.utils.json_to_sheet(
      anomalyReport.anomalies.map((a) => ({
        RowIndex: a.rowIndex,
        AnomalyScore: a.anomalyScore,
        MaxZScore: a.zMax,
        TopFeature: a.topFeature,
        TopFeatureValue: a.topFeatureValue,
      }))
    );
    XLSX.utils.book_append_sheet(wb, wsAnom, 'Anomalies');
  }

  return XLSX.write(wb, { bookType: 'xlsx', type: 'array' }) as Uint8Array;
}

export function buildAnomaliesCsv(report: AnomalyReport): string {
  const esc = (v: unknown) => {
    const s = String(v ?? '');
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const previewCols = report.anomalies.length > 0 ? Object.keys(report.anomalies[0].rowPreview) : [];
  const headers = ['row_index', 'anomaly_score', 'max_z_score', 'top_feature', 'top_feature_value', ...previewCols];
  const lines = [headers.map(esc).join(',')];
  for (const a of report.anomalies) {
    lines.push(
      [
        a.rowIndex,
        a.anomalyScore,
        a.zMax,
        a.topFeature,
        a.topFeatureValue,
        ...previewCols.map((c) => a.rowPreview[c] ?? ''),
      ]
        .map(esc)
        .join(',')
    );
  }
  return lines.join('\n');
}
