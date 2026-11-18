import JSZip from 'jszip';
import * as XLSX from 'xlsx';

export type SupportedUnstructuredExt = '.pdf' | '.docx' | '.txt';

export const SUPPORTED_UNSTRUCTURED_EXTENSIONS: SupportedUnstructuredExt[] = ['.pdf', '.docx', '.txt'];
export const MAX_UNSTRUCTURED_FILE_MB = 15;

export interface ConvertedDatasetResult {
  sourceFileName: string;
  sourceFileType: 'PDF' | 'DOCX' | 'TXT';
  fileSizeBytes: number;
  extractionStrategy: string;
  headers: string[];
  rows: Record<string, string>[];
  rawTextPreview: string;
  rawCharacterCount: number;
  missingCellCount: number;
  warnings: string[];
  extractedAt: string;
}

export class UnstructuredConversionError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'UnstructuredConversionError';
  }
}

interface RawExtractedContent {
  fileType: 'PDF' | 'DOCX' | 'TXT';
  rawText: string;
  tables: string[][][];
  warnings: string[];
}

function getFileExtension(filename: string): string {
  const idx = filename.lastIndexOf('.');
  return idx >= 0 ? filename.slice(idx).toLowerCase() : '';
}

function normalizeHeaderName(raw: string, fallbackIdx: number): string {
  const cleaned = raw
    .replace(/[\r\n\t]+/g, ' ')
    .replace(/[^\w\s-]/g, ' ')
    .trim()
    .replace(/\s+/g, '_');
  return cleaned || `Column_${fallbackIdx + 1}`;
}

function deduplicateHeaders(rawHeaders: string[]): string[] {
  const seen = new Map<string, number>();
  return rawHeaders.map((raw, idx) => {
    const base = normalizeHeaderName(raw, idx);
    const count = seen.get(base) ?? 0;
    seen.set(base, count + 1);
    return count === 0 ? base : `${base}_${count + 1}`;
  });
}

function sanitizeCellValue(val: string | undefined | null): string {
  if (val === undefined || val === null) return '';
  return String(val).replace(/\s+/g, ' ').trim();
}

function decodeTextBuffer(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  if (bytes.length === 0) {
    throw new UnstructuredConversionError('The uploaded text file is empty (0 bytes).');
  }

  // Detect UTF-16 BOM vs corrupted binary null bytes
  const hasUtf16LeBom = bytes.length >= 2 && bytes[0] === 0xff && bytes[1] === 0xfe;
  const hasUtf16BeBom = bytes.length >= 2 && bytes[0] === 0xfe && bytes[1] === 0xff;

  if (!hasUtf16LeBom && !hasUtf16BeBom) {
    let nulCount = 0;
    const checkLen = Math.min(bytes.length, 4096);
    for (let i = 0; i < checkLen; i++) {
      if (bytes[i] === 0) nulCount++;
    }
    if (nulCount > 2) {
      throw new UnstructuredConversionError(
        'The uploaded .txt file appears to contain binary or corrupted data rather than readable text.'
      );
    }
  }

  const decoder = hasUtf16LeBom
    ? new TextDecoder('utf-16le')
    : hasUtf16BeBom
    ? new TextDecoder('utf-16be')
    : new TextDecoder('utf-8', { fatal: false });

  const text = decoder.decode(bytes).replace(/^\uFEFF/, '');
  if (!text.trim()) {
    throw new UnstructuredConversionError('The uploaded text file contains no readable text content.');
  }
  return text;
}

function decodeXmlEntities(str: string): string {
  return str
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'")
    .replace(/&#(\d+);/g, (_, num) => String.fromCharCode(Number(num)))
    .replace(/&#x([0-9a-fA-F]+);/g, (_, hex) => String.fromCharCode(parseInt(hex, 16)));
}

function extractTextFromParagraphXml(pXml: string): string {
  const withBreaks = pXml
    .replace(/<w:tab\b[^>]*\/>/g, '\t')
    .replace(/<w:br\b[^>]*\/>/g, '\n');
  const parts: string[] = [];
  const textTagRegex = /<w:t\b[^>]*>([\s\S]*?)<\/w:t>/g;
  let m: RegExpExecArray | null;
  while ((m = textTagRegex.exec(withBreaks)) !== null) {
    parts.push(decodeXmlEntities(m[1]));
  }
  return parts.join('').trim();
}

async function extractDocxContent(buffer: ArrayBuffer): Promise<RawExtractedContent> {
  let zip: JSZip;
  try {
    zip = await JSZip.loadAsync(buffer);
  } catch {
    throw new UnstructuredConversionError(
      'Unable to read this .docx file. It may be corrupted or not a valid Word (.docx) archive.'
    );
  }

  if (zip.file('EncryptedPackage') || zip.file('EncryptionInfo')) {
    throw new UnstructuredConversionError(
      'This Word (.docx) document is password-protected / encrypted and cannot be extracted automatically.'
    );
  }

  const docEntry = zip.file('word/document.xml');
  if (!docEntry) {
    throw new UnstructuredConversionError(
      'Invalid .docx structure: missing word/document.xml payload.'
    );
  }

  const xml = await docEntry.async('string');
  const tables: string[][][] = [];
  const warnings: string[] = [];

  // Extract Word tables (<w:tbl>...</w:tbl>)
  const tblRegex = /<w:tbl\b[\s\S]*?<\/w:tbl>/g;
  let tblMatch: RegExpExecArray | null;
  while ((tblMatch = tblRegex.exec(xml)) !== null) {
    const tblXml = tblMatch[0];
    const rows: string[][] = [];
    const trRegex = /<w:tr\b[\s\S]*?<\/w:tr>/g;
    let trMatch: RegExpExecArray | null;
    while ((trMatch = trRegex.exec(tblXml)) !== null) {
      const trXml = trMatch[0];
      const cells: string[] = [];
      const tcRegex = /<w:tc\b[\s\S]*?<\/w:tc>/g;
      let tcMatch: RegExpExecArray | null;
      while ((tcMatch = tcRegex.exec(trXml)) !== null) {
        const tcXml = tcMatch[0];
        const cellParagraphs: string[] = [];
        const pRegex = /<w:p\b[\s\S]*?<\/w:p>/g;
        let pMatch: RegExpExecArray | null;
        while ((pMatch = pRegex.exec(tcXml)) !== null) {
          const txt = extractTextFromParagraphXml(pMatch[0]);
          if (txt) cellParagraphs.push(txt);
        }
        cells.push(cellParagraphs.join(' ').trim());
      }
      if (cells.some((c) => c !== '')) {
        rows.push(cells);
      }
    }
    if (rows.length >= 2) {
      tables.push(rows);
    }
  }

  // Remove tables before extracting standalone paragraphs so table cells are not duplicated
  const xmlWithoutTables = xml.replace(/<w:tbl\b[\s\S]*?<\/w:tbl>/g, '');
  const paragraphs: string[] = [];
  const pRegex = /<w:p\b[\s\S]*?<\/w:p>/g;
  let pMatch: RegExpExecArray | null;
  while ((pMatch = pRegex.exec(xmlWithoutTables)) !== null) {
    const txt = extractTextFromParagraphXml(pMatch[0]);
    if (txt) paragraphs.push(txt);
  }

  const tableLines = tables.flatMap((tbl) => tbl.map((r) => r.join('\t')));
  const combinedText = [...paragraphs, ...tableLines].join('\n').trim();
  if (!combinedText) {
    throw new UnstructuredConversionError('The uploaded Word (.docx) document contains no readable text or tables.');
  }

  return {
    fileType: 'DOCX',
    rawText: [...paragraphs, ...tableLines].join('\n'),
    tables,
    warnings,
  };
}

async function extractPdfContent(buffer: ArrayBuffer): Promise<RawExtractedContent> {
  const pdfjs = await import('pdfjs-dist/legacy/build/pdf.mjs');
  if (typeof window !== 'undefined' && pdfjs.GlobalWorkerOptions) {
    pdfjs.GlobalWorkerOptions.workerSrc = new URL(
      'pdfjs-dist/legacy/build/pdf.worker.min.mjs',
      import.meta.url
    ).href;
  }

  const dataCopy = new Uint8Array(buffer.slice(0));
  const loadingTask = pdfjs.getDocument({
    data: dataCopy,
    verbosity: 0,
  });

  let pdfDoc: Awaited<typeof loadingTask.promise>;
  try {
    pdfDoc = await loadingTask.promise;
  } catch (err: unknown) {
    const name = (err as { name?: string })?.name ?? '';
    const msg = err instanceof Error ? err.message : String(err);
    if (name === 'PasswordException' || /password/i.test(msg)) {
      throw new UnstructuredConversionError(
        'This PDF document is password-protected. Please unlock it before uploading.'
      );
    }
    throw new UnstructuredConversionError(
      'Unable to read this PDF document. It may be corrupted or use an unsupported PDF format.'
    );
  }

  const pageLines: string[] = [];
  const warnings: string[] = [];
  const maxPages = Math.min(pdfDoc.numPages, 150);
  if (pdfDoc.numPages > maxPages) {
    warnings.push(`Processed first ${maxPages} of ${pdfDoc.numPages} PDF pages for browser safety.`);
  }

  for (let p = 1; p <= maxPages; p++) {
    const page = await pdfDoc.getPage(p);
    const content = await page.getTextContent();
    interface PosItem {
      str: string;
      x: number;
      y: number;
      width: number;
    }
    const items: PosItem[] = [];
    for (const rawItem of content.items) {
      const candidate = rawItem as { str?: string; transform?: number[]; width?: number };
      if (!candidate.str || !candidate.str.trim()) continue;
      const tr = candidate.transform ?? [1, 0, 0, 1, 0, 0];
      items.push({
        str: candidate.str,
        x: Number(tr[4] ?? 0),
        y: Number(tr[5] ?? 0),
        width: Number(candidate.width ?? 0),
      });
    }

    // Group items by Y coordinate (within 3pt tolerance) sorted top-to-bottom, left-to-right
    items.sort((a, b) => (Math.abs(b.y - a.y) > 3 ? b.y - a.y : a.x - b.x));
    const lines: PosItem[][] = [];
    for (const item of items) {
      const current = lines[lines.length - 1];
      if (!current || Math.abs(current[0].y - item.y) > 3) {
        lines.push([item]);
      } else {
        current.push(item);
      }
    }

    for (const lineItems of lines) {
      lineItems.sort((a, b) => a.x - b.x);
      let lineStr = '';
      for (let i = 0; i < lineItems.length; i++) {
        const it = lineItems[i];
        if (i > 0) {
          const prev = lineItems[i - 1];
          const gap = it.x - (prev.x + prev.width);
          lineStr += gap > 18 ? '   ' : ' ';
        }
        lineStr += it.str.trim();
      }
      const trimmed = lineStr.trim();
      if (trimmed) pageLines.push(trimmed);
    }
  }

  const rawText = pageLines.join('\n').trim();
  if (!rawText) {
    throw new UnstructuredConversionError(
      'No readable text could be extracted from this PDF. It may be empty or consist of scanned images.'
    );
  }

  return {
    fileType: 'PDF',
    rawText,
    tables: [],
    warnings,
  };
}

function tryParseTextTable(rawText: string): { headers: string[]; rows: Record<string, string>[]; strategy: string } | null {
  const nonEmptyLines = rawText
    .split(/\r?\n/)
    .map((l) => l.trimEnd())
    .filter((l) => l.trim().length > 0 && !/^[-=_|+ ]+$/.test(l.trim()));

  if (nonEmptyLines.length < 3) return null;

  // Candidate 1: Pipe-delimited Markdown / ASCII tables
  const pipeLines = nonEmptyLines.filter((l) => l.includes('|'));
  if (pipeLines.length >= 3 && pipeLines.length >= nonEmptyLines.length * 0.5) {
    const splitPipe = (line: string) =>
      line
        .replace(/^\|/, '')
        .replace(/\|$/, '')
        .split('|')
        .map((c) => sanitizeCellValue(c));
    const parsedRows = pipeLines.map(splitPipe).filter((r) => r.length >= 2);
    if (parsedRows.length >= 2) {
      return buildTableFromMatrix(parsedRows, 'Detected Pipe-Delimited Table');
    }
  }

  // Candidate 2: Tab-delimited lines
  const tabLines = nonEmptyLines.filter((l) => l.includes('\t'));
  if (tabLines.length >= 3 && tabLines.length >= nonEmptyLines.length * 0.5) {
    const parsedRows = tabLines
      .map((l) => l.split('\t').map((c) => sanitizeCellValue(c)))
      .filter((r) => r.length >= 2);
    if (parsedRows.length >= 2) {
      return buildTableFromMatrix(parsedRows, 'Detected Tab-Delimited Table');
    }
  }

  // Candidate 3: Multi-space aligned columns (common in PDF / fixed-width TXT reports)
  const spaceSplitRows = nonEmptyLines.map((l) => l.trim().split(/\s{2,}/).map((c) => sanitizeCellValue(c)));
  const multiColRows = spaceSplitRows.filter((r) => r.length >= 2);
  if (multiColRows.length >= 3 && multiColRows.length >= nonEmptyLines.length * 0.6) {
    const colCounts = new Map<number, number>();
    for (const r of multiColRows) {
      colCounts.set(r.length, (colCounts.get(r.length) ?? 0) + 1);
    }
    const [targetCols, targetFreq] = Array.from(colCounts.entries()).sort((a, b) => b[1] - a[1])[0];
    if (targetCols >= 2 && targetFreq >= 3) {
      const startIdx = multiColRows.findIndex((r) => r.length === targetCols);
      const consistent = multiColRows.slice(startIdx).filter((r) => r.length <= targetCols + 1);
      if (consistent.length >= 2) {
        return buildTableFromMatrix(consistent, 'Detected Column-Aligned Table');
      }
    }
  }

  return null;
}

function buildTableFromMatrix(
  matrix: string[][],
  strategy: string
): { headers: string[]; rows: Record<string, string>[]; strategy: string } {
  const maxCols = Math.max(...matrix.map((r) => r.length));
  const rawHeaderRow = [...matrix[0]];
  while (rawHeaderRow.length < maxCols) rawHeaderRow.push('');
  const headers = deduplicateHeaders(rawHeaderRow);

  const rows: Record<string, string>[] = [];
  for (let i = 1; i < matrix.length; i++) {
    const cells = matrix[i];
    if (cells.every((c) => sanitizeCellValue(c) === '')) continue;
    const obj: Record<string, string> = {};
    headers.forEach((h, colIdx) => {
      obj[h] = sanitizeCellValue(cells[colIdx]);
    });
    rows.push(obj);
  }
  return { headers, rows, strategy };
}

function tryParseKeyValueRecords(
  rawText: string
): { headers: string[]; rows: Record<string, string>[]; strategy: string } | null {
  const lines = rawText.split(/\r?\n/);
  const kvRegex = /^([A-Za-z][A-Za-z0-9\s/_#().-]{1,42})\s*[:=]\s*(.*)$/;

  // Collect blocks separated by blank lines or dashed dividers
  const blocks: { pairs: { key: string; value: string }[]; rawHeading?: string }[] = [];
  let currentPairs: { key: string; value: string }[] = [];
  let pendingHeading: string | undefined;

  const flushBlock = () => {
    if (currentPairs.length > 0) {
      blocks.push({ pairs: currentPairs, rawHeading: pendingHeading });
      currentPairs = [];
      pendingHeading = undefined;
    }
  };

  for (const rawLine of lines) {
    const trimmed = rawLine.trim();
    if (!trimmed || /^[-=_*]{3,}$/.test(trimmed)) {
      flushBlock();
      continue;
    }

    const match = trimmed.match(kvRegex);
    if (match) {
      const key = match[1].trim();
      const value = sanitizeCellValue(match[2]);
      // If the same key repeats inside a contiguous block without blank lines, start a new record
      if (currentPairs.some((p) => p.key.toLowerCase() === key.toLowerCase())) {
        flushBlock();
      }
      currentPairs.push({ key, value });
    } else if (currentPairs.length === 0 && trimmed.length <= 70) {
      pendingHeading = trimmed;
    }
  }
  flushBlock();

  if (blocks.length === 0) return null;

  const totalPairs = blocks.reduce((s, b) => s + b.pairs.length, 0);
  const totalNonEmptyLines = lines.filter((l) => l.trim().length > 0).length;
  if (totalPairs < 3 || totalPairs < totalNonEmptyLines * 0.3) return null;

  // Multi-record document (e.g. 2+ invoices, profiles, tickets) -> 1 row per record
  if (blocks.length >= 2) {
    const keyOrder: string[] = [];
    const normalizedMap = new Map<string, string>();
    const hasHeading = blocks.some((b) => Boolean(b.rawHeading));

    if (hasHeading) {
      keyOrder.push('Record_Heading');
    }
    for (const b of blocks) {
      for (const p of b.pairs) {
        const lower = p.key.toLowerCase();
        if (!normalizedMap.has(lower)) {
          normalizedMap.set(lower, p.key);
          keyOrder.push(p.key);
        }
      }
    }

    const headers = deduplicateHeaders(keyOrder);
    const headerByLower = new Map<string, string>();
    if (hasHeading) {
      keyOrder.slice(1).forEach((k, idx) => headerByLower.set(k.toLowerCase(), headers[idx + 1]));
    } else {
      keyOrder.forEach((k, idx) => headerByLower.set(k.toLowerCase(), headers[idx]));
    }

    const rows: Record<string, string>[] = blocks.map((b) => {
      const row: Record<string, string> = {};
      headers.forEach((h) => {
        row[h] = '';
      });
      if (hasHeading && b.rawHeading) {
        row[headers[0]] = sanitizeCellValue(b.rawHeading);
      }
      for (const p of b.pairs) {
        const h = headerByLower.get(p.key.toLowerCase());
        if (h) row[h] = p.value;
      }
      return row;
    });

    return {
      headers,
      rows,
      strategy: 'Structured Key-Value Records',
    };
  }

  // Single key-value block -> normalize as Field / Value / Numeric_Value / Category_Hint rows
  const singlePairs = blocks[0].pairs;
  if (singlePairs.length >= 2) {
    const headers = ['Field', 'Value', 'Numeric_Value'];
    const rows = singlePairs.map((p) => ({
      Field: p.key,
      Value: p.value,
      Numeric_Value: extractFirstNumericString(p.value),
    }));
    return {
      headers,
      rows,
      strategy: 'Single-Document Key-Value Attributes',
    };
  }

  return null;
}

function extractFirstNumericString(text: string): string {
  const numMatch = text.match(/(?:[$€£₹]\s*)?(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?)(?:\s*%)?/);
  if (!numMatch) return '';
  const cleaned = numMatch[1].replace(/,/g, '');
  const parsed = Number(cleaned);
  return Number.isFinite(parsed) ? String(parsed) : '';
}

function extractAllNumbersString(text: string): string {
  const matches = text.match(/(?:[$€£₹]\s*)?(?:-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?)(?:\s*%)?/g);
  if (!matches) return '';
  // Pick first standalone number cleanly
  return extractFirstNumericString(matches[0]);
}

function extractExplicitDate(text: string): string {
  const iso = text.match(/\b(\d{4}-\d{2}-\d{2})\b/);
  if (iso) return iso[1];
  const slash = text.match(/\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b/);
  if (slash) return slash[1];
  const month = text.match(
    /\b((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:,?\s+\d{4})?)\b/i
  );
  if (month) return month[1];
  const monthYear = text.match(
    /\b((?:Q[1-4]\s+\d{4})|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4})\b/i
  );
  return monthYear ? monthYear[1] : '';
}

function extractUnitOrCurrency(text: string): string {
  if (/[$]|USD\b/i.test(text)) return 'USD';
  if (/[€]|EUR\b/i.test(text)) return 'EUR';
  if (/[£]|GBP\b/i.test(text)) return 'GBP';
  if (/[₹]|INR\b/i.test(text)) return 'INR';
  if (/\d\s*%/.test(text)) return '%';
  const unitMatch = text.match(/\b\d+(?:\.\d+)?\s*(units?|orders?|items?|hours?|days?|kg|tons?|users?|customers?|tickets?)\b/i);
  return unitMatch ? unitMatch[1] : '';
}

function classifyTextCategory(text: string): string {
  const lower = text.toLowerCase();
  if (/\b(revenue|sales|profit|cost|expense|price|budget|invoice|payment|amount|margin|income)\b/.test(lower)) {
    return 'Financial';
  }
  if (/\b(customer|client|user|account|subscriber|retention|churn|segment|buyer)\b/.test(lower)) {
    return 'Customer';
  }
  if (/\b(product|sku|inventory|stock|shipment|delivery|warehouse|supply|category|order)\b/.test(lower)) {
    return 'Operations';
  }
  if (/\b(employee|staff|department|hr|hiring|salary|attendance|team|manager)\b/.test(lower)) {
    return 'Workforce';
  }
  if (/\b(error|issue|bug|incident|ticket|delay|alert|warning|risk)\b/.test(lower)) {
    return 'Risk_And_Support';
  }
  return '';
}

function extractCapitalizedEntities(text: string): string {
  const cleaned = text.replace(/^[-*•\d.)\s]+/, '');
  const matches = cleaned.match(/\b([A-Z][A-Za-z0-9&-]{1,}(?:\s+[A-Z][A-Za-z0-9&-]{1,}){0,2})\b/g);
  if (!matches) return '';
  const stopWords = new Set(['The', 'This', 'That', 'These', 'Those', 'Total', 'Average', 'Summary', 'Section', 'Note', 'In', 'On', 'At', 'For', 'By', 'With']);
  const filtered = matches.filter((m) => !stopWords.has(m));
  return filtered.length > 0 ? Array.from(new Set(filtered)).slice(0, 3).join('; ') : '';
}

function extractProseRows(rawText: string): { headers: string[]; rows: Record<string, string>[]; strategy: string } {
  const rawLines = rawText.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  const statements: { section: string; text: string }[] = [];
  let currentSection = '';

  for (const line of rawLines) {
    const looksLikeHeading =
      line.length <= 60 &&
      !/[.!?;]$/.test(line) &&
      !/[$€£₹%]/.test(line) &&
      !/\b\d{4}-\d{2}-\d{2}\b/.test(line) &&
      !/\b\d{2,}\b/.test(line) &&
      line.split(/\s+/).length <= 6 &&
      !line.startsWith('-') &&
      !line.startsWith('*') &&
      !line.startsWith('•');

    if (looksLikeHeading) {
      currentSection = line.replace(/:$/, '').trim();
      continue;
    }

    // Split multi-sentence lines cleanly while preserving bullet items as whole records
    if (/^[-*•\d]+[.)]\s+/.test(line)) {
      statements.push({
        section: currentSection,
        text: line.replace(/^[-*•]+\s+|^\d+[.)]\s+/, '').trim(),
      });
    } else {
      const sentences = line
        .split(/(?<=[.!?])\s+(?=[A-Z0-9])/)
        .map((s) => s.trim())
        .filter(Boolean);
      for (const s of sentences) {
        statements.push({ section: currentSection, text: s });
      }
    }
  }

  const headers = [
    'Row_Number',
    'Section',
    'Category',
    'Entity',
    'Date_Mentioned',
    'Numeric_Value',
    'Unit_Or_Currency',
    'Extracted_Text',
  ];

  const rows: Record<string, string>[] = statements.map((st, idx) => ({
    Row_Number: String(idx + 1),
    Section: st.section,
    Category: classifyTextCategory(st.text),
    Entity: extractCapitalizedEntities(st.text),
    Date_Mentioned: extractExplicitDate(st.text),
    Numeric_Value: extractAllNumbersString(st.text),
    Unit_Or_Currency: extractUnitOrCurrency(st.text),
    Extracted_Text: st.text,
  }));

  return {
    headers,
    rows,
    strategy: 'Entity & Fact Extraction from Unstructured Text',
  };
}

function structureExtractedContent(extracted: RawExtractedContent): {
  headers: string[];
  rows: Record<string, string>[];
  strategy: string;
  warnings: string[];
} {
  const warnings = [...extracted.warnings];

  // 1. Native Word tables extracted from DOCX XML
  if (extracted.tables.length > 0) {
    const bestTable = [...extracted.tables].sort((a, b) => b.length * b[0].length - a.length * a[0].length)[0];
    const tableRes = buildTableFromMatrix(bestTable, 'Extracted Native Document Table');
    if (tableRes.rows.length > 0) {
      if (extracted.tables.length > 1) {
        warnings.push(`Found ${extracted.tables.length} tables; converted the largest table (${tableRes.rows.length} rows).`);
      }
      return { ...tableRes, warnings };
    }
  }

  // 2. Pipe / Tab / Multi-Space aligned tables inside PDF or TXT
  const textTableRes = tryParseTextTable(extracted.rawText);
  if (textTableRes && textTableRes.rows.length > 0) {
    return { ...textTableRes, warnings };
  }

  // 3. Key-Value structured records / forms
  const kvRes = tryParseKeyValueRecords(extracted.rawText);
  if (kvRes && kvRes.rows.length > 0) {
    return { ...kvRes, warnings };
  }

  // 4. General prose / bulleted document entity & metric extraction
  const proseRes = extractProseRows(extracted.rawText);
  if (proseRes.rows.length === 0) {
    throw new UnstructuredConversionError(
      'Unable to organize any meaningful rows from the uploaded document.'
    );
  }
  return { ...proseRes, warnings };
}

export async function convertUnstructuredBuffer(
  filename: string,
  buffer: ArrayBuffer
): Promise<ConvertedDatasetResult> {
  if (!filename || !filename.trim()) {
    throw new UnstructuredConversionError('Missing filename for uploaded document.');
  }

  const ext = getFileExtension(filename);
  if (!SUPPORTED_UNSTRUCTURED_EXTENSIONS.includes(ext as SupportedUnstructuredExt)) {
    throw new UnstructuredConversionError(
      `Unsupported file format '${ext || 'unknown'}'. Please upload a .pdf, .docx, or .txt file.`
    );
  }

  if (buffer.byteLength === 0) {
    throw new UnstructuredConversionError(`Uploaded file '${filename}' is empty (0 bytes).`);
  }

  const maxBytes = MAX_UNSTRUCTURED_FILE_MB * 1024 * 1024;
  if (buffer.byteLength > maxBytes) {
    throw new UnstructuredConversionError(
      `Uploaded file '${filename}' exceeds the ${MAX_UNSTRUCTURED_FILE_MB} MB limit.`
    );
  }

  let extracted: RawExtractedContent;
  if (ext === '.txt') {
    const rawText = decodeTextBuffer(buffer);
    extracted = { fileType: 'TXT', rawText, tables: [], warnings: [] };
  } else if (ext === '.docx') {
    extracted = await extractDocxContent(buffer);
  } else {
    extracted = await extractPdfContent(buffer);
  }

  const structured = structureExtractedContent(extracted);

  let missingCellCount = 0;
  for (const row of structured.rows) {
    for (const h of structured.headers) {
      if (!row[h] || row[h].trim() === '') {
        missingCellCount++;
      }
    }
  }

  return {
    sourceFileName: filename,
    sourceFileType: extracted.fileType,
    fileSizeBytes: buffer.byteLength,
    extractionStrategy: structured.strategy,
    headers: structured.headers,
    rows: structured.rows,
    rawTextPreview: extracted.rawText.slice(0, 4000),
    rawCharacterCount: extracted.rawText.length,
    missingCellCount,
    warnings: structured.warnings,
    extractedAt: new Date().toISOString(),
  };
}

export async function convertUnstructuredFile(file: File): Promise<ConvertedDatasetResult> {
  const buffer = await file.arrayBuffer();
  return convertUnstructuredBuffer(file.name, buffer);
}

export function buildConvertedExcelBytes(result: ConvertedDatasetResult): Uint8Array {
  const wb = XLSX.utils.book_new();

  // Preserve numeric cells as numbers and keep empty strings as true Excel blank cells (null)
  const aoa: (string | number | null)[][] = [result.headers];
  for (const row of result.rows) {
    aoa.push(
      result.headers.map((h) => {
        const raw = (row[h] ?? '').trim();
        if (raw === '') return null;
        if (/^-?\d+(?:\.\d+)?$/.test(raw)) {
          const num = Number(raw);
          if (Number.isFinite(num)) return num;
        }
        return raw;
      })
    );
  }

  const wsData = XLSX.utils.aoa_to_sheet(aoa);
  wsData['!cols'] = result.headers.map((h) => {
    let maxLen = h.length;
    for (const r of result.rows.slice(0, 50)) {
      maxLen = Math.max(maxLen, (r[h] ?? '').length);
    }
    return { wch: Math.min(48, Math.max(12, maxLen + 3)) };
  });
  XLSX.utils.book_append_sheet(wb, wsData, 'Structured_Data');

  const metaRows = [
    { Property: 'Source File', Value: result.sourceFileName },
    { Property: 'Source Type', Value: result.sourceFileType },
    { Property: 'Extraction Strategy', Value: result.extractionStrategy },
    { Property: 'Extracted Rows', Value: result.rows.length },
    { Property: 'Extracted Columns', Value: result.headers.length },
    { Property: 'Blank / Missing Cells Preserved', Value: result.missingCellCount },
    { Property: 'Extracted At', Value: result.extractedAt },
  ];
  const wsMeta = XLSX.utils.json_to_sheet(metaRows);
  XLSX.utils.book_append_sheet(wb, wsMeta, 'Conversion_Summary');

  const out = XLSX.write(wb, { bookType: 'xlsx', type: 'array' });
  return out instanceof Uint8Array ? out : new Uint8Array(out as ArrayBuffer);
}
