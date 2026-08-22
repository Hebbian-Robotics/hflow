import { keepPreviousData, useQuery } from "@tanstack/react-query";
import {
  type ColumnDef,
  flexRender,
  getCoreRowModel,
  type SortingState,
  type Updater,
  useReactTable,
} from "@tanstack/react-table";
import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import {
  DEFAULT_ORDER_BY,
  type EpisodeColumnStats,
  type EpisodeRow,
  type EpisodeStatus,
  type EpisodesFilter,
  fetchEpisodeFacets,
  fetchEpisodeStats,
  fetchEpisodesPage,
  parseEpisodesQuery,
} from "../api";
import { FacetSidebar, type MultiValueFacetName } from "../components/FacetSidebar";
import { HeaderDistribution } from "../components/HeaderDistribution";
import { EmptyPanel, ErrorPanel, LoadingPanel } from "../components/QueryStates";
import { SqlFooter } from "../components/SqlFooter";
import { ValueCell } from "../components/ValueCell";
import { ChevronDownIcon, SearchIcon } from "../icons";

const SEARCH_DEBOUNCE_MS = 300;
const STATS_DEBOUNCE_MS = 250;
const PAGE_SIZE_CHOICES = [25, 50, 100, 250, 500];

// Header-popover clicks only map to columns with an existing structured filter
// param; other columns render their stats without a click-to-filter affordance.
const STATS_FILTERABLE_COLUMNS = new Set(["task", "operator", "embodiment", "status"]);

export function EpisodesPage() {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const query = useMemo(() => parseEpisodesQuery(searchParams), [searchParams]);

  const episodesQuery = useQuery({
    queryKey: ["episodes", query],
    queryFn: () => fetchEpisodesPage(query),
    placeholderData: keepPreviousData,
  });
  const facetsQuery = useQuery({ queryKey: ["episode-facets"], queryFn: fetchEpisodeFacets });

  // Column mini-distributions, computed server-side under the ACTIVE filters.
  // Debounced and independent of the table query — the table never waits on stats.
  const statsFilter = useMemo<EpisodesFilter>(
    () => ({
      task: query.task,
      operator: query.operator,
      embodiment: query.embodiment,
      status: query.status,
      success: query.success,
      search: query.search,
    }),
    [query],
  );
  const statsFilterKey = useMemo(() => JSON.stringify(statsFilter), [statsFilter]);
  const [debouncedStats, setDebouncedStats] = useState<{ key: string; filter: EpisodesFilter }>({
    key: statsFilterKey,
    filter: statsFilter,
  });
  useEffect(() => {
    if (statsFilterKey === debouncedStats.key) return;
    const debounceHandle = window.setTimeout(
      () => setDebouncedStats({ key: statsFilterKey, filter: statsFilter }),
      STATS_DEBOUNCE_MS,
    );
    return () => window.clearTimeout(debounceHandle);
  }, [statsFilterKey, statsFilter, debouncedStats.key]);
  const statsQuery = useQuery({
    queryKey: ["episode-stats", debouncedStats.key],
    queryFn: () => fetchEpisodeStats(debouncedStats.filter),
    placeholderData: keepPreviousData,
  });
  const statsByColumn = useMemo(() => {
    const byColumn = new Map<string, EpisodeColumnStats>();
    for (const columnStats of statsQuery.data?.columns ?? []) {
      byColumn.set(columnStats.name, columnStats);
    }
    return byColumn;
  }, [statsQuery.data]);

  // Any filter or sort change restarts pagination at the first page.
  const updateFilters = useCallback(
    (mutate: (params: URLSearchParams) => void) => {
      setSearchParams(
        (previous) => {
          const next = new URLSearchParams(previous);
          mutate(next);
          next.delete("offset");
          return next;
        },
        { replace: true },
      );
    },
    [setSearchParams],
  );

  // Debounced search box, kept in sync when the URL changes from elsewhere.
  const [searchDraft, setSearchDraft] = useState(query.search);
  useEffect(() => {
    setSearchDraft(query.search);
  }, [query.search]);
  useEffect(() => {
    if (searchDraft === query.search) return;
    const debounceHandle = window.setTimeout(() => {
      updateFilters((params) => {
        if (searchDraft) params.set("search", searchDraft);
        else params.delete("search");
      });
    }, SEARCH_DEBOUNCE_MS);
    return () => window.clearTimeout(debounceHandle);
  }, [searchDraft, query.search, updateFilters]);

  const toggleFacetValue = useCallback(
    (facet: MultiValueFacetName, value: string) => {
      updateFilters((params) => {
        const currentValues = params.getAll(facet);
        params.delete(facet);
        const nextValues = currentValues.includes(value)
          ? currentValues.filter((existing) => existing !== value)
          : [...currentValues, value];
        for (const entry of nextValues) params.append(facet, entry);
      });
    },
    [updateFilters],
  );

  const selectStatus = useCallback(
    (status: EpisodeStatus | null) => {
      updateFilters((params) => {
        if (status) params.set("status", status);
        else params.delete("status");
      });
    },
    [updateFilters],
  );

  const selectSuccess = (value: string) => {
    updateFilters((params) => {
      if (value === "true" || value === "false") params.set("success", value);
      else params.delete("success");
    });
  };

  // A popover value click applies the same structured params the sidebar uses.
  const applyStatsValueFilter = useCallback(
    (columnName: string, value: string) => {
      if (columnName === "status") {
        if (value === "ok" || value === "quarantined") {
          selectStatus(query.status === value ? null : value);
        }
        return;
      }
      if (columnName === "task" || columnName === "operator" || columnName === "embodiment") {
        toggleFacetValue(columnName, value);
      }
    },
    [query.status, selectStatus, toggleFacetValue],
  );

  const activeStatsValues = useCallback(
    (columnName: string): readonly string[] => {
      switch (columnName) {
        case "task":
          return query.task;
        case "operator":
          return query.operator;
        case "embodiment":
          return query.embodiment;
        case "status":
          return query.status ? [query.status] : [];
        default:
          return [];
      }
    },
    [query],
  );

  // Sorting maps 1:1 onto the API's order_by/order params; the server default
  // (recorded_at desc) is mirrored so the header indicator is honest.
  const sorting: SortingState = useMemo(
    () => [{ id: query.orderBy ?? DEFAULT_ORDER_BY, desc: query.order === "desc" }],
    [query.orderBy, query.order],
  );

  const handleSortingChange = useCallback(
    (updater: Updater<SortingState>) => {
      const nextSorting = typeof updater === "function" ? updater(sorting) : updater;
      const primarySort = nextSorting[0];
      updateFilters((params) => {
        if (primarySort) {
          params.set("order_by", primarySort.id);
          params.set("order", primarySort.desc ? "desc" : "asc");
        } else {
          params.delete("order_by");
          params.delete("order");
        }
      });
    },
    [sorting, updateFilters],
  );

  const episodesData = episodesQuery.data;
  const rows = useMemo(() => episodesData?.rows ?? [], [episodesData]);

  // Columns come from the server's DESCRIBE of the wide view, so the table
  // renders whatever the catalog holds without a hardcoded schema.
  const tableColumns = useMemo<ColumnDef<EpisodeRow, unknown>[]>(() => {
    return (episodesData?.columns ?? []).map((column) => ({
      id: column.name,
      accessorFn: (row: EpisodeRow) => row[column.name],
      header: () => (
        <span className="th-label" title={`${column.name} · ${column.type}`}>
          {column.name}
        </span>
      ),
      cell: (cellContext) => <ValueCell columnName={column.name} value={cellContext.getValue()} />,
    }));
  }, [episodesData]);

  const table = useReactTable({
    data: rows,
    columns: tableColumns,
    getCoreRowModel: getCoreRowModel(),
    manualSorting: true,
    manualPagination: true,
    enableMultiSort: false,
    enableSortingRemoval: false,
    state: { sorting },
    onSortingChange: handleSortingChange,
  });

  const total = episodesData?.total ?? 0;
  const pageStart = total === 0 ? 0 : query.offset + 1;
  const pageEnd = query.offset + rows.length;
  const canGoPrevious = query.offset > 0;
  const canGoNext = query.offset + rows.length < total;

  const goToOffset = (nextOffset: number) => {
    setSearchParams((previous) => {
      const next = new URLSearchParams(previous);
      if (nextOffset > 0) next.set("offset", String(nextOffset));
      else next.delete("offset");
      return next;
    });
  };

  const setPageSize = (nextLimit: number) => {
    updateFilters((params) => {
      params.set("limit", String(nextLimit));
    });
  };

  const openEpisode = (row: EpisodeRow) => {
    const episodeId = row.episode_id;
    if (typeof episodeId !== "string") return;
    navigate(`/episodes/${encodeURIComponent(episodeId)}`);
  };

  const hasActiveFilters =
    query.task.length > 0 ||
    query.operator.length > 0 ||
    query.embodiment.length > 0 ||
    query.status !== null ||
    query.success !== null ||
    query.search !== "";

  const clearFilters = () => {
    setSearchParams(new URLSearchParams(), { replace: true });
  };

  const pageSizeChoices = PAGE_SIZE_CHOICES.includes(query.limit)
    ? PAGE_SIZE_CHOICES
    : [...PAGE_SIZE_CHOICES, query.limit].sort((left, right) => left - right);

  const isRefreshing = episodesQuery.isFetching && !episodesQuery.isPending;

  return (
    <div className="episodes-page">
      <div className="toolbar">
        <div className="search-box">
          <SearchIcon className="search-icon" />
          <input
            type="search"
            className="input search-input"
            placeholder="Search id, task, operator…"
            value={searchDraft}
            onChange={(event) => setSearchDraft(event.target.value)}
            aria-label="Search episodes by id, task, or operator"
          />
        </div>
        <label className="toolbar-field">
          <span className="toolbar-field-label">Status</span>
          <select
            className="input"
            value={query.status ?? "any"}
            onChange={(event) => {
              const value = event.target.value;
              selectStatus(value === "ok" || value === "quarantined" ? value : null);
            }}
          >
            <option value="any">any</option>
            <option value="ok">ok</option>
            <option value="quarantined">quarantined</option>
          </select>
        </label>
        <label className="toolbar-field">
          <span className="toolbar-field-label">Success</span>
          <select
            className="input"
            value={query.success ?? "any"}
            onChange={(event) => selectSuccess(event.target.value)}
          >
            <option value="any">any</option>
            <option value="true">success</option>
            <option value="false">failure</option>
          </select>
        </label>
        <div className="toolbar-spacer" />
        {isRefreshing ? <span className="refresh-note">updating…</span> : null}
        <span className="row-count">
          {total} episode{total === 1 ? "" : "s"}
        </span>
      </div>

      <div className="episodes-body">
        <FacetSidebar
          facets={facetsQuery.data}
          isPending={facetsQuery.isPending}
          error={facetsQuery.isError ? facetsQuery.error : null}
          onRetry={() => {
            void facetsQuery.refetch();
          }}
          selectedValues={{
            task: query.task,
            operator: query.operator,
            embodiment: query.embodiment,
          }}
          selectedStatus={query.status}
          onToggleValue={toggleFacetValue}
          onSelectStatus={selectStatus}
        />
        <div className="table-pane">
          {episodesQuery.isPending ? (
            <LoadingPanel label="Loading episodes…" />
          ) : episodesQuery.isError ? (
            <ErrorPanel
              error={episodesQuery.error}
              onRetry={() => {
                void episodesQuery.refetch();
              }}
            />
          ) : rows.length === 0 ? (
            hasActiveFilters ? (
              <EmptyPanel title="No episodes match the current filters.">
                <button type="button" className="btn" onClick={clearFilters}>
                  Clear filters
                </button>
              </EmptyPanel>
            ) : (
              <EmptyPanel
                title="No episodes in this catalog yet."
                hint="Record one with app.test(record=True) or import recordings with hflow ingest."
              />
            )
          ) : (
            <div className={isRefreshing ? "table-scroll is-refreshing" : "table-scroll"}>
              <table className="data-table">
                <thead>
                  {table.getHeaderGroups().map((headerGroup) => (
                    <tr key={headerGroup.id}>
                      {headerGroup.headers.map((header) => {
                        const sortDirection = header.column.getIsSorted();
                        const columnStats = statsByColumn.get(header.column.id);
                        const isStatsFilterable =
                          columnStats?.kind === "categorical" &&
                          STATS_FILTERABLE_COLUMNS.has(header.column.id);
                        return (
                          <th
                            key={header.id}
                            aria-sort={
                              sortDirection === "asc"
                                ? "ascending"
                                : sortDirection === "desc"
                                  ? "descending"
                                  : undefined
                            }
                          >
                            <div className="th-inner">
                              <button
                                type="button"
                                className="th-sort"
                                onClick={header.column.getToggleSortingHandler()}
                              >
                                {flexRender(header.column.columnDef.header, header.getContext())}
                                {sortDirection ? (
                                  <ChevronDownIcon
                                    className={
                                      sortDirection === "asc" ? "sort-arrow is-asc" : "sort-arrow"
                                    }
                                  />
                                ) : null}
                              </button>
                              {columnStats ? (
                                <HeaderDistribution
                                  stats={columnStats}
                                  onSelectValue={
                                    isStatsFilterable
                                      ? (value) => applyStatsValueFilter(header.column.id, value)
                                      : null
                                  }
                                  activeValues={activeStatsValues(header.column.id)}
                                />
                              ) : null}
                            </div>
                          </th>
                        );
                      })}
                    </tr>
                  ))}
                </thead>
                <tbody>
                  {table.getRowModel().rows.map((tableRow) => (
                    <tr
                      key={tableRow.id}
                      className="episode-row"
                      tabIndex={0}
                      onClick={() => openEpisode(tableRow.original)}
                      onKeyDown={(event) => {
                        if (event.key === "Enter") openEpisode(tableRow.original);
                      }}
                    >
                      {tableRow.getVisibleCells().map((cell) => (
                        <td key={cell.id}>
                          {flexRender(cell.column.columnDef.cell, cell.getContext())}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>

      <div className="pagination-bar">
        <button
          type="button"
          className="btn"
          disabled={!canGoPrevious}
          onClick={() => goToOffset(Math.max(0, query.offset - query.limit))}
        >
          Previous
        </button>
        <button
          type="button"
          className="btn"
          disabled={!canGoNext}
          onClick={() => goToOffset(query.offset + query.limit)}
        >
          Next
        </button>
        <span className="page-range">
          {pageStart}–{pageEnd} of {total}
        </span>
        <div className="toolbar-spacer" />
        <label className="toolbar-field">
          <span className="toolbar-field-label">Rows</span>
          <select
            className="input"
            value={String(query.limit)}
            onChange={(event) => setPageSize(Number.parseInt(event.target.value, 10))}
          >
            {pageSizeChoices.map((size) => (
              <option key={size} value={String(size)}>
                {size}
              </option>
            ))}
          </select>
        </label>
      </div>

      <SqlFooter sql={episodesData?.sql} />
    </div>
  );
}
