import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import { AppShell } from "./AppShell";
import { EmptyPanel } from "./components/QueryStates";
import { EpisodeDetailPage } from "./pages/EpisodeDetailPage";
import { EpisodesPage } from "./pages/EpisodesPage";
import "./styles.css";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      staleTime: 15_000,
      refetchOnWindowFocus: false,
    },
  },
});

const router = createBrowserRouter([
  {
    path: "/",
    element: <AppShell />,
    children: [
      { index: true, element: <EpisodesPage /> },
      { path: "episodes/:episodeId", element: <EpisodeDetailPage /> },
      {
        path: "*",
        element: (
          <div className="episode-page">
            <EmptyPanel title="Page not found." hint="Use Episodes in the nav rail." />
          </div>
        ),
      },
    ],
  },
]);

const rootElement = document.getElementById("root");
if (!rootElement) throw new Error("Root element #root is missing from index.html");

createRoot(rootElement).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  </StrictMode>,
);
