import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
// React Flow ships its own stylesheet and it has to load before ours, which
// overrides parts of it.
import "@xyflow/react/dist/style.css";
import "./styles.css";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // This server is on localhost or a private network, so a failed request
      // is a real refusal (no runtime, no catalog) far more often than it is a
      // blip. Retrying would only delay showing the reason.
      retry: false,
      refetchOnWindowFocus: false,
    },
  },
});

const container = document.getElementById("root");
if (container === null) throw new Error("index.html is missing its #root element");

createRoot(container).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </StrictMode>,
);
