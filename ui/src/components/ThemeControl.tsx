import { Monitor, Moon, Sun } from "lucide-react";
import { RadioGroup } from "radix-ui";
import type { ComponentType } from "react";
import { type ThemePreference, useThemePreference } from "../theme";

// Theme control, in the nav rail's footer next to the rest of the workspace
// readout.
//
// Three buttons rather than one cycling button: every state is one keystroke
// or one click away, the active one is visible without being pressed, and
// nobody has to learn the cycle order. Radix RadioGroup is the primitive that
// fits — pick one of three, exactly one always chosen — and it brings the
// radio pattern with it: one Tab stop for the group, arrow keys that move AND
// choose, and role/aria-checked that match what the arrow keys actually do.
// (ToggleGroup also announces `radiogroup`, but its arrows only move focus.)

const THEME_OPTIONS: readonly {
  value: ThemePreference;
  label: string;
  Icon: ComponentType<{ className?: string }>;
}[] = [
  { value: "system", label: "System", Icon: Monitor },
  { value: "light", label: "Light", Icon: Sun },
  { value: "dark", label: "Dark", Icon: Moon },
];

export function ThemeControl() {
  const { preference, resolvedTheme, setPreference } = useThemePreference();
  return (
    <RadioGroup.Root
      className="theme-control"
      value={preference}
      onValueChange={(next) => {
        if (next === "system" || next === "light" || next === "dark") setPreference(next);
      }}
      aria-label="Colour theme"
    >
      {THEME_OPTIONS.map(({ value, label, Icon }) => {
        const title =
          value === "system" ? `Follow the system theme (currently ${resolvedTheme})` : label;
        return (
          <RadioGroup.Item
            key={value}
            value={value}
            className="theme-control-item"
            aria-label={title}
            title={title}
          >
            <Icon />
          </RadioGroup.Item>
        );
      })}
    </RadioGroup.Root>
  );
}
