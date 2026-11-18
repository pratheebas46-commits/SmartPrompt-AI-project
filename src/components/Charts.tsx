import React, { useState } from 'react';
import type { AnalysisResult, AnomalyReport } from '../engine/smartpromptEngine';

interface AnalysisChartProps {
  analysis: AnalysisResult;
  chartOverride?: 'auto' | 'bar' | 'horizontal_bar' | 'line' | 'pie' | 'histogram' | 'scatter' | 'heatmap';
  correlationMatrix?: { columns: string[]; matrix: number[][] };
}

function formatAxisNumber(n: number): string {
  if (Math.abs(n) >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (Math.abs(n) >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return Number.isInteger(n) ? String(n) : n.toFixed(1);
}

const PIE_PALETTE = ['#1E3A8A', '#4F46E5', '#0F766E', '#64748B', '#4338CA', '#0369A1', '#94A3B8'];

export const AnalysisChart: React.FC<AnalysisChartProps> = ({
  analysis,
  chartOverride = 'auto',
  correlationMatrix,
}) => {
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);

  const chartType =
    chartOverride === 'auto'
      ? analysis.recommendedChart === 'table'
        ? 'bar'
        : analysis.recommendedChart
      : chartOverride;

  if (!analysis.rows || analysis.rows.length === 0) {
    return (
      <div className="flex h-64 items-center justify-center border border-dashed border-slate-200 rounded-lg bg-slate-50/50 text-sm text-slate-500">
        No chartable observations available for this query.
      </div>
    );
  }

  // Scatter Plot
  if (chartType === 'scatter') {
    const pts = analysis.rows
      .map((r, i) => ({
        idx: i,
        x: Number(r[analysis.xColumn]),
        y: Number(r[analysis.yColumn]),
      }))
      .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));

    if (pts.length === 0) {
      return (
        <div className="flex h-64 items-center justify-center border border-dashed border-slate-200 rounded-lg text-sm text-slate-500">
          Select two numerical columns to render scatter correlation.
        </div>
      );
    }

    const minX = Math.min(...pts.map((p) => p.x));
    const maxX = Math.max(...pts.map((p) => p.x));
    const minY = Math.min(...pts.map((p) => p.y));
    const maxY = Math.max(...pts.map((p) => p.y));
    const spanX = maxX - minX || 1;
    const spanY = maxY - minY || 1;

    const W = 720;
    const H = 310;
    const padL = 62;
    const padR = 24;
    const padT = 22;
    const padB = 46;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;

    const toSvgX = (x: number) => padL + ((x - minX) / spanX) * plotW;
    const toSvgY = (y: number) => padT + plotH - ((y - minY) / spanY) * plotH;

    const hovered = hoverIdx !== null ? pts[hoverIdx] : null;

    return (
      <div className="relative">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-76 overflow-visible">
          {[0, 0.25, 0.5, 0.75, 1].map((t) => {
            const y = padT + plotH * (1 - t);
            const val = minY + spanY * t;
            return (
              <g key={t}>
                <line x1={padL} y1={y} x2={W - padR} y2={y} stroke="#E2E8F0" strokeDasharray="3 3" />
                <text x={padL - 8} y={y + 4} textAnchor="end" className="fill-slate-500 text-[11px] font-mono">
                  {formatAxisNumber(val)}
                </text>
              </g>
            );
          })}

          {pts.map((p, idx) => (
            <circle
              key={idx}
              cx={toSvgX(p.x)}
              cy={toSvgY(p.y)}
              r={hoverIdx === idx ? 7 : 5}
              className="fill-blue-600/80 stroke-white cursor-pointer transition-all"
              strokeWidth={1.5}
              onMouseEnter={() => setHoverIdx(idx)}
              onMouseLeave={() => setHoverIdx(null)}
            />
          ))}

          <text x={padL + plotW / 2} y={H - 8} textAnchor="middle" className="fill-slate-600 text-[12px] font-medium">
            {analysis.xColumn}
          </text>
          <text
            x={16}
            y={padT + plotH / 2}
            textAnchor="middle"
            transform={`rotate(-90 16 ${padT + plotH / 2})`}
            className="fill-slate-600 text-[12px] font-medium"
          >
            {analysis.yColumn}
          </text>
        </svg>

        {hovered && (
          <div className="mt-2 flex items-center justify-between border-t border-slate-100 pt-2 text-xs text-slate-600 font-mono num-tabular">
            <span>
              {analysis.xColumn}: <strong className="text-slate-900">{hovered.x.toLocaleString()}</strong>
            </span>
            <span>
              {analysis.yColumn}: <strong className="text-slate-900">{hovered.y.toLocaleString()}</strong>
            </span>
          </div>
        )}
      </div>
    );
  }

  // Correlation Matrix Heatmap
  if (chartType === 'heatmap') {
    if (!correlationMatrix || correlationMatrix.columns.length < 2) {
      return (
        <div className="flex h-60 items-center justify-center border border-dashed border-slate-200 rounded-lg text-sm text-slate-500">
          At least two numerical columns are required to compute a correlation heatmap.
        </div>
      );
    }
    const { columns, matrix } = correlationMatrix;
    const cellBg = (v: number) => {
      if (v >= 0.7) return 'bg-blue-600 text-white';
      if (v >= 0.35) return 'bg-blue-200 text-blue-950';
      if (v > -0.35) return 'bg-slate-100 text-slate-700';
      if (v > -0.7) return 'bg-amber-200 text-amber-950';
      return 'bg-red-600 text-white';
    };
    return (
      <div className="overflow-x-auto py-2">
        <table className="w-full text-xs border-collapse">
          <thead>
            <tr>
              <th className="p-2 text-left text-slate-500 font-medium">Feature</th>
              {columns.map((c) => (
                <th key={c} className="p-2 font-mono text-center text-slate-700 whitespace-nowrap">
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {columns.map((rowCol, rIdx) => (
              <tr key={rowCol}>
                <td className="p-2 font-mono font-medium text-slate-800 whitespace-nowrap">{rowCol}</td>
                {columns.map((colCol, cIdx) => {
                  const val = matrix[rIdx]?.[cIdx] ?? 0;
                  return (
                    <td key={colCol} className="p-1">
                      <div
                        className={`rounded px-2 py-2 text-center font-mono num-tabular font-semibold ${cellBg(val)}`}
                        title={`${rowCol} vs ${colCol}: r = ${val}`}
                      >
                        {val.toFixed(2)}
                      </div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }

  // Extract label + numeric value series
  const items = analysis.rows.map((r) => ({
    label: String(r[analysis.xColumn] ?? ''),
    value: Number(r[analysis.yColumn]) || 0,
  }));

  // Pie / Share Donut Chart
  if (chartType === 'pie') {
    const positiveItems = items.filter((d) => d.value > 0).slice(0, 7);
    const total = positiveItems.reduce((s, d) => s + d.value, 0);
    if (total <= 0) {
      return (
        <div className="flex h-60 items-center justify-center border border-dashed border-slate-200 rounded-lg text-sm text-slate-500">
          Pie chart requires positive category values.
        </div>
      );
    }
    let accAngle = -Math.PI / 2;
    const cx = 130;
    const cy = 130;
    const outerR = 105;
    const innerR = 56;
    const slices = positiveItems.map((item, idx) => {
      const frac = item.value / total;
      const start = accAngle;
      const end = accAngle + frac * Math.PI * 2;
      accAngle = end;
      const largeArc = frac > 0.5 ? 1 : 0;
      const x1 = cx + outerR * Math.cos(start);
      const y1 = cy + outerR * Math.sin(start);
      const x2 = cx + outerR * Math.cos(end);
      const y2 = cy + outerR * Math.sin(end);
      const ix1 = cx + innerR * Math.cos(end);
      const iy1 = cy + innerR * Math.sin(end);
      const ix2 = cx + innerR * Math.cos(start);
      const iy2 = cy + innerR * Math.sin(start);
      const d =
        positiveItems.length === 1
          ? `M ${cx} ${cy - outerR} A ${outerR} ${outerR} 0 1 1 ${cx - 0.01} ${cy - outerR} Z`
          : `M ${x1} ${y1} A ${outerR} ${outerR} 0 ${largeArc} 1 ${x2} ${y2} L ${ix1} ${iy1} A ${innerR} ${innerR} 0 ${largeArc} 0 ${ix2} ${iy2} Z`;
      return {
        ...item,
        pct: Number((frac * 100).toFixed(1)),
        color: PIE_PALETTE[idx % PIE_PALETTE.length],
        d,
      };
    });

    return (
      <div className="grid grid-cols-1 md:grid-cols-12 items-center gap-6 py-2">
        <div className="md:col-span-5 flex justify-center">
          <svg viewBox="0 0 260 260" className="w-60 h-60">
            {slices.map((sl, idx) => (
              <path
                key={sl.label}
                d={sl.d}
                fill={sl.color}
                opacity={hoverIdx === null || hoverIdx === idx ? 1 : 0.45}
                className="cursor-pointer transition-opacity stroke-white"
                strokeWidth={2}
                onMouseEnter={() => setHoverIdx(idx)}
                onMouseLeave={() => setHoverIdx(null)}
              />
            ))}
            <text x={cx} y={cy - 4} textAnchor="middle" className="fill-slate-500 text-[10px]">
              Total
            </text>
            <text x={cx} y={cy + 14} textAnchor="middle" className="fill-slate-900 text-xs font-mono font-semibold">
              {formatAxisNumber(total)}
            </text>
          </svg>
        </div>
        <div className="md:col-span-7 space-y-2">
          {slices.map((sl, idx) => (
            <div
              key={sl.label}
              onMouseEnter={() => setHoverIdx(idx)}
              onMouseLeave={() => setHoverIdx(null)}
              className={`flex items-center justify-between px-3 py-1.5 rounded text-xs cursor-pointer ${
                hoverIdx === idx ? 'bg-slate-100' : ''
              }`}
            >
              <div className="flex items-center gap-2 min-w-0">
                <span className="h-2.5 w-2.5 rounded-sm shrink-0" style={{ backgroundColor: sl.color }} />
                <span className="font-medium text-slate-800 truncate">{sl.label}</span>
              </div>
              <div className="font-mono num-tabular text-slate-700 shrink-0">
                {sl.value.toLocaleString(undefined, { maximumFractionDigits: 2 })} ({sl.pct}%)
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  // Horizontal Ranking Bar Chart
  if (chartType === 'horizontal_bar') {
    const maxVal = Math.max(...items.map((d) => Math.abs(d.value)), 1);
    return (
      <div className="space-y-2.5 py-2">
        {items.slice(0, 12).map((item, idx) => {
          const pct = Math.max(2, (Math.abs(item.value) / maxVal) * 100);
          return (
            <div
              key={idx}
              className="grid grid-cols-12 items-center gap-3 text-sm"
              onMouseEnter={() => setHoverIdx(idx)}
              onMouseLeave={() => setHoverIdx(null)}
            >
              <div className="col-span-3 truncate text-right font-medium text-slate-700" title={item.label}>
                {item.label}
              </div>
              <div className="col-span-7 h-7 rounded bg-slate-100 overflow-hidden flex items-center">
                <div
                  className={`h-full rounded transition-all ${
                    hoverIdx === idx ? 'bg-blue-700' : idx === 0 ? 'bg-blue-600' : 'bg-blue-500/85'
                  }`}
                  style={{ width: `${pct}%` }}
                />
              </div>
              <div className="col-span-2 text-right font-mono text-xs font-semibold text-slate-800 num-tabular">
                {item.value.toLocaleString(undefined, { maximumFractionDigits: 2 })}
              </div>
            </div>
          );
        })}
      </div>
    );
  }

  // Vertical Bar / Histogram / Line Trend
  const W = 720;
  const H = 300;
  const padL = 62;
  const padR = 20;
  const padT = 22;
  const padB = 54;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const vals = items.map((d) => d.value);
  const minV = Math.min(0, ...vals);
  const maxV = Math.max(...vals, 1);
  const spanV = maxV - minV || 1;
  const zeroY = padT + plotH - ((0 - minV) / spanV) * plotH;

  if (chartType === 'line') {
    const stepX = items.length > 1 ? plotW / (items.length - 1) : plotW / 2;
    const coords = items.map((d, i) => ({
      ...d,
      cx: items.length > 1 ? padL + i * stepX : padL + plotW / 2,
      cy: padT + plotH - ((d.value - minV) / spanV) * plotH,
    }));
    const polylinePoints = coords.map((c) => `${c.cx},${c.cy}`).join(' ');
    const hovered = hoverIdx !== null ? coords[hoverIdx] : null;

    return (
      <div>
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-76 overflow-visible">
          {[0, 0.25, 0.5, 0.75, 1].map((t) => {
            const y = padT + plotH * (1 - t);
            const v = minV + spanV * t;
            return (
              <g key={t}>
                <line x1={padL} y1={y} x2={W - padR} y2={y} stroke="#E2E8F0" strokeDasharray="3 3" />
                <text x={padL - 8} y={y + 4} textAnchor="end" className="fill-slate-500 text-[11px] font-mono">
                  {formatAxisNumber(v)}
                </text>
              </g>
            );
          })}

          {coords.length > 1 && (
            <polyline fill="none" stroke="#2563EB" strokeWidth={2.5} points={polylinePoints} />
          )}

          {coords.map((c, idx) => (
            <g key={idx}>
              <circle
                cx={c.cx}
                cy={c.cy}
                r={hoverIdx === idx ? 6.5 : 4.5}
                className="fill-blue-600 stroke-white cursor-pointer"
                strokeWidth={2}
                onMouseEnter={() => setHoverIdx(idx)}
                onMouseLeave={() => setHoverIdx(null)}
              />
              <text
                x={c.cx}
                y={H - 24}
                textAnchor="middle"
                className="fill-slate-600 text-[10px] font-mono"
              >
                {c.label.length > 10 ? `${c.label.slice(0, 9)}…` : c.label}
              </text>
            </g>
          ))}
        </svg>

        {hovered && (
          <div className="mt-2 flex items-center justify-between border-t border-slate-100 pt-2 text-xs text-slate-600 font-mono num-tabular">
            <span>
              {analysis.xColumn}: <strong className="text-slate-900">{hovered.label}</strong>
            </span>
            <span>
              {analysis.yColumn}: <strong className="text-slate-900">{hovered.value.toLocaleString()}</strong>
            </span>
          </div>
        )}
      </div>
    );
  }

  // Default Vertical Bar / Histogram
  const barSlot = plotW / Math.max(1, items.length);
  const barWidth = Math.min(48, Math.max(12, barSlot * (chartType === 'histogram' ? 0.88 : 0.62)));
  const hovered = hoverIdx !== null ? items[hoverIdx] : null;

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-76 overflow-visible">
        {[0, 0.25, 0.5, 0.75, 1].map((t) => {
          const y = padT + plotH * (1 - t);
          const v = minV + spanV * t;
          return (
            <g key={t}>
              <line x1={padL} y1={y} x2={W - padR} y2={y} stroke="#E2E8F0" strokeDasharray="3 3" />
              <text x={padL - 8} y={y + 4} textAnchor="end" className="fill-slate-500 text-[11px] font-mono">
                {formatAxisNumber(v)}
              </text>
            </g>
          );
        })}

        <line x1={padL} y1={zeroY} x2={W - padR} y2={zeroY} stroke="#94A3B8" strokeWidth={1} />

        {items.slice(0, 14).map((d, idx) => {
          const cx = padL + idx * barSlot + barSlot / 2;
          const x = cx - barWidth / 2;
          const valY = padT + plotH - ((d.value - minV) / spanV) * plotH;
          const barY = Math.min(valY, zeroY);
          const barH = Math.max(2, Math.abs(zeroY - valY));

          return (
            <g
              key={idx}
              className="cursor-pointer"
              onMouseEnter={() => setHoverIdx(idx)}
              onMouseLeave={() => setHoverIdx(null)}
            >
              <rect
                x={x}
                y={barY}
                width={barWidth}
                height={barH}
                rx={3}
                className={
                  hoverIdx === idx
                    ? 'fill-indigo-700'
                    : chartType === 'histogram'
                    ? 'fill-indigo-600/85'
                    : 'fill-blue-900/85 dark:fill-indigo-500/85'
                }
              />
              <text x={cx} y={H - 22} textAnchor="middle" className="fill-slate-600 dark:fill-slate-400 text-[10px] font-medium">
                {d.label.length > 11 ? `${d.label.slice(0, 10)}…` : d.label}
              </text>
            </g>
          );
        })}
      </svg>

      {hovered && (
        <div className="mt-2 flex items-center justify-between border-t border-slate-100 dark:border-slate-800 pt-2 text-xs text-slate-600 dark:text-slate-300 font-mono num-tabular">
          <span>
            {analysis.xColumn}: <strong className="text-slate-900 dark:text-slate-100">{hovered.label}</strong>
          </span>
          <span>
            {analysis.yColumn}: <strong className="text-slate-900 dark:text-slate-100">{hovered.value.toLocaleString()}</strong>
          </span>
        </div>
      )}
    </div>
  );
};

interface AnomalyScatterProps {
  report: AnomalyReport;
}

export const AnomalyScatterChart: React.FC<AnomalyScatterProps> = ({ report }) => {
  const [hovered, setHovered] = useState<AnomalyReport['scatterPoints'][number] | null>(null);
  const pts = report.scatterPoints;
  if (pts.length === 0) {
    return (
      <div className="flex h-60 items-center justify-center border border-dashed border-slate-200 rounded-lg text-sm text-slate-500">
        Select at least one numerical feature to visualize anomaly distribution.
      </div>
    );
  }

  const minX = Math.min(...pts.map((p) => p.x));
  const maxX = Math.max(...pts.map((p) => p.x));
  const minY = Math.min(...pts.map((p) => p.y));
  const maxY = Math.max(...pts.map((p) => p.y));
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;

  const W = 720;
  const H = 290;
  const padL = 62;
  const padR = 24;
  const padT = 20;
  const padB = 46;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-72 overflow-visible">
        {[0, 0.25, 0.5, 0.75, 1].map((t) => {
          const y = padT + plotH * (1 - t);
          const v = minY + spanY * t;
          return (
            <g key={t}>
              <line x1={padL} y1={y} x2={W - padR} y2={y} stroke="#E2E8F0" strokeDasharray="3 3" />
              <text x={padL - 8} y={y + 4} textAnchor="end" className="fill-slate-500 text-[11px] font-mono">
                {formatAxisNumber(v)}
              </text>
            </g>
          );
        })}

        {pts.map((p) => {
          const cx = padL + ((p.x - minX) / spanX) * plotW;
          const cy = padT + plotH - ((p.y - minY) / spanY) * plotH;
          return (
            <circle
              key={p.rowIndex}
              cx={cx}
              cy={cy}
              r={p.isAnomaly ? 6.5 : 4.5}
              className={`cursor-pointer stroke-white transition-all ${
                p.isAnomaly ? 'fill-red-600' : 'fill-slate-400/80'
              }`}
              strokeWidth={1.5}
              onMouseEnter={() => setHovered(p)}
              onMouseLeave={() => setHovered(null)}
            />
          );
        })}

        <text x={padL + plotW / 2} y={H - 8} textAnchor="middle" className="fill-slate-600 text-[12px] font-medium">
          {report.xFeature}
        </text>
        <text
          x={16}
          y={padT + plotH / 2}
          textAnchor="middle"
          transform={`rotate(-90 16 ${padT + plotH / 2})`}
          className="fill-slate-600 text-[12px] font-medium"
        >
          {report.yFeature}
        </text>
      </svg>

      <div className="mt-2 flex flex-wrap items-center justify-between gap-2 border-t border-slate-100 pt-2 text-xs text-slate-600">
        <div className="flex items-center gap-4">
          <span className="inline-flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-full bg-slate-400" />
            Normal observation
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="h-2.5 w-2.5 rounded-full bg-red-600" />
            Flagged statistical anomaly
          </span>
        </div>
        {hovered && (
          <span className="font-mono num-tabular text-slate-800">
            Row #{hovered.rowIndex} · {report.xFeature}={hovered.x.toLocaleString()} · {report.yFeature}=
            {hovered.y.toLocaleString()} · Score={hovered.score}
          </span>
        )}
      </div>
    </div>
  );
};

export const ActivityByIntentChart: React.FC<{
  entries: { intent: string; count: number }[];
}> = ({ entries }) => {
  if (entries.length === 0) {
    return (
      <div className="flex h-48 items-center justify-center text-xs text-slate-500 dark:text-slate-400">
        Run an analysis to populate activity distribution.
      </div>
    );
  }
  const maxVal = Math.max(...entries.map((e) => e.count), 1);
  return (
    <div className="space-y-2.5">
      {entries.slice(0, 6).map((item) => {
        const pct = Math.max(6, (item.count / maxVal) * 100);
        return (
          <div key={item.intent} className="grid grid-cols-12 items-center gap-3 text-xs">
            <div className="col-span-5 truncate text-slate-700 dark:text-slate-300 font-medium">
              {item.intent.replace(/_/g, ' ')}
            </div>
            <div className="col-span-5 h-5 rounded bg-slate-100 dark:bg-slate-800 overflow-hidden">
              <div
                className="h-full rounded bg-blue-900 dark:bg-indigo-500 transition-all"
                style={{ width: `${pct}%` }}
              />
            </div>
            <div className="col-span-2 text-right font-mono num-tabular font-semibold text-slate-800 dark:text-slate-200">
              {item.count}
            </div>
          </div>
        );
      })}
    </div>
  );
};

export const PromptQualityComparisonChart: React.FC<{
  runs: { label: string; original: number; optimized: number }[];
}> = ({ runs }) => {
  if (runs.length === 0) {
    return (
      <div className="flex h-48 items-center justify-center text-xs text-slate-500 dark:text-slate-400">
        No prompt optimization runs recorded yet.
      </div>
    );
  }
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-4 text-[11px] text-slate-600 dark:text-slate-400">
        <span className="inline-flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-sm bg-slate-300 dark:bg-slate-600" />
          Original Score
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-sm bg-indigo-600 dark:bg-indigo-400" />
          Optimized Score
        </span>
      </div>
      <div className="space-y-2.5">
        {runs.slice(0, 5).map((r, i) => (
          <div key={i} className="space-y-1">
            <div className="flex justify-between text-xs">
              <span className="text-slate-700 dark:text-slate-300 truncate max-w-[68%] font-medium" title={r.label}>
                {r.label}
              </span>
              <span className="font-mono num-tabular text-slate-700 dark:text-slate-300">
                {r.original} → <strong className="text-indigo-700 dark:text-indigo-400">{r.optimized}</strong>
              </span>
            </div>
            <div className="h-2 rounded bg-slate-100 dark:bg-slate-800 overflow-hidden relative">
              <div
                className="h-full bg-indigo-600 dark:bg-indigo-400 rounded"
                style={{ width: `${Math.min(100, Math.max(2, r.optimized))}%` }}
              />
              <div
                className="h-full bg-slate-400/70 dark:bg-slate-600/80 absolute top-0 left-0 rounded-l"
                style={{ width: `${Math.min(100, Math.max(2, r.original))}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
