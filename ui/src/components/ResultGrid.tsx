import { type ColumnDef, flexRender, tableFeatures, useTable } from "@tanstack/react-table";
import { useMemo } from "react";
import type { ColumnDescriptor, ResultRow } from "../api";
import { EmptyPanel } from "./QueryStates";
import { ValueCell } from "./ValueCell";

/**
 * Read-only result grid for curation previews — the Episodes table's visual
 * twin (same .data-table styling and value rendering), minus sorting and
 * pagination, which the studio doesn't do client-side (thin-client rule).
 */
// v9 registers features explicitly so unused ones tree-shake away. The
// preview grid renders exactly what the server returned -- no client-side
// sorting, filtering or pagination (the thin-client rule) -- so it registers
// none, and only the always-present core row model remains.
const previewGridFeatures = tableFeatures({});

export function ResultGrid({ columns, rows }: { columns: ColumnDescriptor[]; rows: ResultRow[] }) {
  const tableColumns = useMemo<ColumnDef<typeof previewGridFeatures, ResultRow, unknown>[]>(() => {
    return columns.map((column) => ({
      id: column.name,
      accessorFn: (row: ResultRow) => row[column.name],
      header: () => (
        <span className="th-label" title={`${column.name} · ${column.type}`}>
          {column.name}
        </span>
      ),
      cell: (cellContext) => <ValueCell columnName={column.name} value={cellContext.getValue()} />,
    }));
  }, [columns]);

  const table = useTable({
    features: previewGridFeatures,
    data: rows,
    columns: tableColumns,
  });

  if (rows.length === 0) {
    return <EmptyPanel title="The query returned no rows." hint="Loosen the WHERE clause?" />;
  }

  return (
    <div className="table-scroll">
      <table className="data-table">
        <thead>
          {table.getHeaderGroups().map((headerGroup) => (
            <tr key={headerGroup.id}>
              {headerGroup.headers.map((header) => (
                <th key={header.id} className="th-static">
                  {flexRender(header.column.columnDef.header, header.getContext())}
                </th>
              ))}
            </tr>
          ))}
        </thead>
        <tbody>
          {table.getRowModel().rows.map((tableRow) => (
            <tr key={tableRow.id}>
              {/* getAllCells, not getVisibleCells: the latter belongs to the
                  column-visibility feature, which this grid does not register. */}
              {tableRow.getAllCells().map((cell) => (
                <td key={cell.id}>{flexRender(cell.column.columnDef.cell, cell.getContext())}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
