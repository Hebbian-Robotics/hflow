import { autocompletion, closeBrackets, completionKeymap } from "@codemirror/autocomplete";
import { defaultKeymap, history, historyKeymap } from "@codemirror/commands";
import { PostgreSQL, sql } from "@codemirror/lang-sql";
import { bracketMatching, HighlightStyle, syntaxHighlighting } from "@codemirror/language";
import { Compartment, EditorState } from "@codemirror/state";
import { drawSelection, EditorView, keymap, lineNumbers, placeholder } from "@codemirror/view";
import { tags } from "@lezer/highlight";
import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";

// CodeMirror 6 SQL editor. Everything bundles locally (lezer grammars included),
// so the offline rule holds: no CDN, no fonts, no remote assets.

/** Table name -> column names, fed to SQL autocompletion. */
export type SqlSchema = Record<string, string[]>;

export interface SqlEditorHandle {
  /** Replace the whole document (loading a saved query). */
  setDocument(text: string): void;
  /** Insert at the cursor, replacing any selection (catalog-tree affordance). */
  insertText(text: string): void;
  /**
   * The SQL a run should use: the trimmed selection when one exists and
   * preferSelection is set, otherwise the trimmed full document.
   */
  currentSql(preferSelection: boolean): string;
}

interface SqlEditorProps {
  initialDoc: string;
  schema: SqlSchema;
  onDocChanged: (doc: string) => void;
  onSelectionChanged: (hasSelection: boolean) => void;
  /** Cmd/Ctrl+Enter. The parent decides what to run via the handle. */
  onRunShortcut: () => void;
}

// Colors ride the app's CSS custom properties so light/dark themes just work.
const editorTheme = EditorView.theme({
  "&": {
    height: "100%",
    fontSize: "12.5px",
    backgroundColor: "var(--surface)",
    color: "var(--ink)",
  },
  "&.cm-focused": { outline: "none" },
  ".cm-scroller": { fontFamily: "var(--font-mono)", lineHeight: "1.55" },
  ".cm-content": { caretColor: "var(--ink)", padding: "8px 0" },
  ".cm-cursor, .cm-dropCursor": { borderLeftColor: "var(--ink)" },
  "&.cm-focused > .cm-scroller > .cm-selectionLayer .cm-selectionBackground, .cm-selectionBackground":
    {
      backgroundColor: "var(--accent-soft)",
    },
  ".cm-gutters": {
    backgroundColor: "var(--surface-sunken)",
    color: "var(--ink-faint)",
    border: "none",
    borderRight: "1px solid var(--border)",
  },
  ".cm-placeholder": { color: "var(--ink-faint)" },
  ".cm-tooltip": {
    backgroundColor: "var(--surface)",
    border: "1px solid var(--border-strong)",
    color: "var(--ink)",
  },
  ".cm-tooltip.cm-tooltip-autocomplete > ul > li[aria-selected]": {
    backgroundColor: "var(--accent-soft)",
    color: "var(--accent-ink)",
  },
});

// Syntax highlighting under the app's neutral-chrome rule: the structural
// tokens separate by WEIGHT and ink step (keyword, type, operator, comment)
// and only the two token classes that are literal VALUES keep a hue — strings
// green, numbers amber, the same --ok / --warn that mean "a value" elsewhere.
// Keywords were accent-coloured while the accent was teal; a near-black accent
// would have made them indistinguishable from plain text, so the weight that
// was reinforcing the colour now carries the distinction on its own.
const sqlHighlightStyle = HighlightStyle.define([
  { tag: tags.keyword, color: "var(--ink)", fontWeight: "700" },
  { tag: tags.string, color: "var(--ok)" },
  { tag: tags.number, color: "var(--warn)" },
  { tag: tags.bool, color: "var(--warn)" },
  { tag: tags.comment, color: "var(--ink-faint)", fontStyle: "italic" },
  { tag: tags.operator, color: "var(--ink-muted)" },
  { tag: tags.punctuation, color: "var(--ink-muted)" },
  { tag: tags.typeName, color: "var(--ink)", fontWeight: "600" },
]);

function sqlLanguageExtension(schema: SqlSchema) {
  return sql({ dialect: PostgreSQL, schema, upperCaseKeywords: true });
}

export const SqlEditor = forwardRef<SqlEditorHandle, SqlEditorProps>(function SqlEditor(
  { initialDoc, schema, onDocChanged, onSelectionChanged, onRunShortcut },
  ref,
) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const viewRef = useRef<EditorView | null>(null);
  const languageCompartmentRef = useRef<Compartment | null>(null);
  if (languageCompartmentRef.current === null) languageCompartmentRef.current = new Compartment();
  const languageCompartment = languageCompartmentRef.current;

  // Latest callbacks live behind a ref so the (single) EditorView never holds
  // stale closures and never needs recreating on parent re-renders.
  const callbacksRef = useRef({ onDocChanged, onSelectionChanged, onRunShortcut });
  callbacksRef.current = { onDocChanged, onSelectionChanged, onRunShortcut };
  const initialDocRef = useRef(initialDoc);
  const schemaRef = useRef(schema);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const runShortcutBinding = {
      key: "Mod-Enter",
      run: () => {
        callbacksRef.current.onRunShortcut();
        return true;
      },
    };
    const updateListener = EditorView.updateListener.of((update) => {
      if (update.docChanged) callbacksRef.current.onDocChanged(update.state.doc.toString());
      if (update.selectionSet || update.docChanged) {
        callbacksRef.current.onSelectionChanged(!update.state.selection.main.empty);
      }
    });
    const state = EditorState.create({
      doc: initialDocRef.current,
      extensions: [
        lineNumbers(),
        drawSelection(),
        history(),
        bracketMatching(),
        closeBrackets(),
        autocompletion(),
        // The run shortcut goes first so nothing below can shadow Mod-Enter.
        keymap.of([runShortcutBinding, ...completionKeymap, ...defaultKeymap, ...historyKeymap]),
        languageCompartment.of(sqlLanguageExtension(schemaRef.current)),
        syntaxHighlighting(sqlHighlightStyle),
        editorTheme,
        EditorView.lineWrapping,
        placeholder("SELECT episode_id, task, success FROM episodes WHERE …"),
        updateListener,
      ],
    });
    const view = new EditorView({ state, parent: container });
    viewRef.current = view;
    return () => {
      view.destroy();
      viewRef.current = null;
    };
  }, [languageCompartment]);

  // The catalog tables load async; feed them into completion when they arrive.
  useEffect(() => {
    schemaRef.current = schema;
    const view = viewRef.current;
    if (!view) return;
    view.dispatch({ effects: languageCompartment.reconfigure(sqlLanguageExtension(schema)) });
  }, [schema, languageCompartment]);

  useImperativeHandle(
    ref,
    () => ({
      setDocument(text: string) {
        const view = viewRef.current;
        if (!view) return;
        view.dispatch({
          changes: { from: 0, to: view.state.doc.length, insert: text },
          selection: { anchor: text.length },
        });
        view.focus();
      },
      insertText(text: string) {
        const view = viewRef.current;
        if (!view) return;
        const range = view.state.selection.main;
        view.dispatch({
          changes: { from: range.from, to: range.to, insert: text },
          selection: { anchor: range.from + text.length },
        });
        view.focus();
      },
      currentSql(preferSelection: boolean) {
        const view = viewRef.current;
        if (!view) return "";
        const range = view.state.selection.main;
        if (preferSelection && !range.empty)
          return view.state.sliceDoc(range.from, range.to).trim();
        return view.state.doc.toString().trim();
      },
    }),
    [],
  );

  return <div className="sql-editor" ref={containerRef} />;
});
