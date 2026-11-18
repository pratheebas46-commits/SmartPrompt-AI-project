import React, { useRef, useState } from 'react';
import {
  AlertCircle,
  CheckCircle2,
  Download,
  FileText,
  FileUp,
  RefreshCw,
  Table as TableIcon,
} from 'lucide-react';
import {
  MAX_UNSTRUCTURED_FILE_MB,
  buildConvertedExcelBytes,
  convertUnstructuredFile,
  type ConvertedDatasetResult,
} from '../engine/unstructuredConverter';

interface UnstructuredConverterViewProps {
  onNavigateToWorkspace: () => void;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export function UnstructuredConverterView({ onNavigateToWorkspace }: UnstructuredConverterViewProps) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [converting, setConverting] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [result, setResult] = useState<ConvertedDatasetResult | null>(null);
  const [showRawText, setShowRawText] = useState(false);
  const [downloadedFileName, setDownloadedFileName] = useState<string | null>(null);

  const handleFilePicked = (file: File | null) => {
    setErrorMessage(null);
    setDownloadedFileName(null);
    if (!file) return;
    const lower = file.name.toLowerCase();
    if (!lower.endsWith('.pdf') && !lower.endsWith('.docx') && !lower.endsWith('.txt')) {
      setSelectedFile(null);
      setErrorMessage('Unsupported file format. Please select a .pdf, .docx, or .txt document.');
      return;
    }
    if (file.size === 0) {
      setSelectedFile(null);
      setErrorMessage('The selected file is empty (0 bytes).');
      return;
    }
    if (file.size > MAX_UNSTRUCTURED_FILE_MB * 1024 * 1024) {
      setSelectedFile(null);
      setErrorMessage(`File exceeds maximum size of ${MAX_UNSTRUCTURED_FILE_MB} MB.`);
      return;
    }
    setSelectedFile(file);
    setResult(null);
  };

  const handleExtractAndConvert = async () => {
    if (!selectedFile) {
      setErrorMessage('Please upload a .pdf, .docx, or .txt file first.');
      return;
    }
    setErrorMessage(null);
    setDownloadedFileName(null);
    setConverting(true);
    try {
      const converted = await convertUnstructuredFile(selectedFile);
      setResult(converted);
    } catch (err) {
      setResult(null);
      setErrorMessage(
        err instanceof Error ? err.message : 'Unable to extract structured data from this document.'
      );
    } finally {
      setConverting(false);
    }
  };

  const handleDownloadExcel = () => {
    if (!result) return;
    const bytes = buildConvertedExcelBytes(result);
    const baseName = result.sourceFileName.replace(/\.(pdf|docx|txt)$/i, '');
    const outName = `${baseName}_structured.xlsx`;
    const blob = new Blob([new Uint8Array(bytes)], {
      type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = outName;
    a.click();
    URL.revokeObjectURL(url);
    setDownloadedFileName(outName);
  };

  const handleReset = () => {
    setSelectedFile(null);
    setResult(null);
    setErrorMessage(null);
    setDownloadedFileName(null);
    setShowRawText(false);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold text-slate-900 dark:text-white">
            Unstructured Data to Structured Data Converter
          </h1>
          <p className="text-sm text-slate-600 dark:text-slate-400 mt-0.5">
            Upload a PDF, Word (.docx), or Text (.txt) document, extract information into rows and columns, preview the
            table, and download as Excel (.xlsx).
          </p>
        </div>
        {result && (
          <button
            type="button"
            onClick={handleReset}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-xs font-medium text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-700 self-start cursor-pointer"
          >
            <RefreshCw className="h-3.5 w-3.5" />
            <span>Upload Another Document</span>
          </button>
        )}
      </div>

      {/* Upload & Controls Card */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-md p-6 space-y-5 shadow-[0_1px_2px_rgba(15,23,42,0.03)]">
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.docx,.txt"
          className="hidden"
          onChange={(e) => handleFilePicked(e.target.files?.[0] ?? null)}
        />

        <div
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            handleFilePicked(e.dataTransfer.files?.[0] ?? null);
          }}
          className="border border-dashed border-slate-300 dark:border-slate-700 rounded-md p-8 text-center bg-slate-50/60 dark:bg-slate-800/30 space-y-3"
        >
          <div className="mx-auto h-10 w-10 rounded-md bg-indigo-50 dark:bg-indigo-950/60 border border-indigo-100 dark:border-indigo-900 flex items-center justify-center text-indigo-700 dark:text-indigo-300">
            <FileUp className="h-5 w-5" />
          </div>
          <div className="space-y-1">
            <div className="text-sm font-medium text-slate-900 dark:text-white">
              {selectedFile ? selectedFile.name : 'Select or drag and drop an unstructured document'}
            </div>
            <div className="text-xs text-slate-500 dark:text-slate-400">
              Supported formats: <span className="font-mono font-medium">.pdf</span>,{' '}
              <span className="font-mono font-medium">.docx</span>,{' '}
              <span className="font-mono font-medium">.txt</span> · Max size {MAX_UNSTRUCTURED_FILE_MB} MB
              {selectedFile ? ` · Selected: ${formatBytes(selectedFile.size)}` : ''}
            </div>
          </div>

          <div className="flex flex-wrap items-center justify-center gap-2.5 pt-1">
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="px-4 py-2 rounded-md border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 text-xs sm:text-sm font-medium text-slate-800 dark:text-slate-100 hover:bg-slate-50 dark:hover:bg-slate-700 cursor-pointer"
            >
              Choose File
            </button>
            <button
              type="button"
              disabled={!selectedFile || converting}
              onClick={handleExtractAndConvert}
              className="inline-flex items-center gap-1.5 px-4 py-2 rounded-md bg-blue-900 dark:bg-indigo-600 text-white text-xs sm:text-sm font-medium hover:bg-blue-950 dark:hover:bg-indigo-700 disabled:opacity-50 cursor-pointer"
            >
              <TableIcon className="h-4 w-4" />
              <span>{converting ? 'Extracting & Converting...' : 'Extract and Convert'}</span>
            </button>
          </div>
        </div>

        {errorMessage && (
          <div
            role="alert"
            className="flex items-start gap-2.5 p-3 rounded-md border border-red-200 dark:border-red-900 bg-red-50 dark:bg-red-950/40 text-xs sm:text-sm text-red-700 dark:text-red-300"
          >
            <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
            <span>{errorMessage}</span>
          </div>
        )}

        {result && (
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-3.5 rounded-md border border-emerald-200 dark:border-emerald-900 bg-emerald-50/70 dark:bg-emerald-950/30">
            <div className="flex items-start gap-2.5 text-xs sm:text-sm text-emerald-900 dark:text-emerald-200">
              <CheckCircle2 className="h-4 w-4 shrink-0 mt-0.5 text-emerald-700 dark:text-emerald-400" />
              <div>
                <div className="font-medium">
                  Extracted {result.rows.length} rows across {result.headers.length} columns ({result.extractionStrategy})
                </div>
                <div className="text-xs text-emerald-800/80 dark:text-emerald-300/80 mt-0.5">
                  Source: {result.sourceFileName} ({result.sourceFileType}) · Blank/unavailable cells preserved:{' '}
                  <span className="font-mono">{result.missingCellCount}</span>
                </div>
              </div>
            </div>

            <button
              type="button"
              onClick={handleDownloadExcel}
              className="inline-flex items-center justify-center gap-1.5 px-4 py-2 rounded-md bg-blue-900 dark:bg-indigo-600 text-white text-xs sm:text-sm font-medium hover:bg-blue-950 dark:hover:bg-indigo-700 whitespace-nowrap shrink-0 cursor-pointer"
            >
              <Download className="h-4 w-4" />
              <span>Download as Excel (.xlsx)</span>
            </button>
          </div>
        )}

        {result?.warnings && result.warnings.length > 0 && (
          <div className="space-y-1 text-xs text-amber-800 dark:text-amber-300 bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-900 rounded-md p-3">
            {result.warnings.map((w, i) => (
              <div key={i}>Note: {w}</div>
            ))}
          </div>
        )}

        {downloadedFileName && (
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-3.5 rounded-md border border-indigo-200 dark:border-indigo-900 bg-indigo-50/60 dark:bg-indigo-950/30 text-xs sm:text-sm text-slate-800 dark:text-slate-200">
            <div>
              Downloaded <span className="font-mono font-medium">{downloadedFileName}</span>. Your existing workspace
              dataset was not modified. Open Dataset Workspace to upload this file and run analytics.
            </div>
            <button
              type="button"
              onClick={onNavigateToWorkspace}
              className="px-3 py-1.5 rounded-md border border-indigo-300 dark:border-indigo-700 bg-white dark:bg-slate-800 text-xs font-medium text-indigo-700 dark:text-indigo-300 hover:bg-indigo-50 whitespace-nowrap shrink-0 cursor-pointer"
            >
              Go to Dataset Workspace
            </button>
          </div>
        )}
      </div>

      {/* Preview Table & Extracted Text Toggle */}
      {result && (
        <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-md overflow-hidden shadow-[0_1px_2px_rgba(15,23,42,0.03)]">
          <div className="px-4 py-3 border-b border-slate-200 dark:border-slate-800 flex flex-wrap items-center justify-between gap-3">
            <div>
              <h2 className="text-sm font-semibold text-slate-900 dark:text-white">
                Structured Data Preview ({result.rows.length} rows × {result.headers.length} columns)
              </h2>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                Review extracted columns and rows before downloading. Missing values remain blank and are never invented.
              </p>
            </div>
            <button
              type="button"
              onClick={() => setShowRawText((v) => !v)}
              className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded border border-slate-200 dark:border-slate-700 text-xs font-medium text-slate-700 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-800 cursor-pointer"
            >
              <FileText className="h-3.5 w-3.5" />
              <span>{showRawText ? 'Hide Extracted Raw Text' : 'Inspect Extracted Raw Text'}</span>
            </button>
          </div>

          {showRawText && (
            <div className="p-4 border-b border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950">
              <div className="text-xs font-medium text-slate-500 dark:text-slate-400 mb-1.5">
                Extracted Raw Text ({result.rawCharacterCount} characters)
              </div>
              <pre className="text-xs font-mono text-slate-700 dark:text-slate-300 whitespace-pre-wrap max-h-56 overflow-y-auto">
                {result.rawTextPreview}
              </pre>
            </div>
          )}

          <div className="overflow-x-auto max-h-[460px]">
            <table className="w-full text-xs text-left">
              <thead className="bg-slate-50 dark:bg-slate-800/70 border-b border-slate-200 dark:border-slate-800 text-slate-600 dark:text-slate-300 sticky top-0">
                <tr>
                  <th className="px-3 py-2.5 font-mono text-slate-400">#</th>
                  {result.headers.map((h) => (
                    <th key={h} className="px-3 py-2.5 font-semibold whitespace-nowrap">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                {result.rows.slice(0, 200).map((row, rIdx) => (
                  <tr key={rIdx} className="hover:bg-slate-50 dark:hover:bg-slate-800/40">
                    <td className="px-3 py-2 font-mono text-slate-400 num-tabular">{rIdx + 1}</td>
                    {result.headers.map((h) => {
                      const val = row[h] ?? '';
                      return (
                        <td key={h} className="px-3 py-2 text-slate-800 dark:text-slate-200 max-w-xs truncate" title={val}>
                          {val !== '' ? val : <span className="text-slate-300 dark:text-slate-600">—</span>}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {result.rows.length > 200 && (
            <div className="px-4 py-2 border-t border-slate-200 dark:border-slate-800 text-xs text-slate-500 dark:text-slate-400">
              Showing first 200 of {result.rows.length} rows in browser preview. All {result.rows.length} rows will be
              included in the downloaded Excel file.
            </div>
          )}
        </div>
      )}
    </div>
  );
}
