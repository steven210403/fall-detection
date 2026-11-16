import js from "@eslint/js";
import globals from "globals";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";
import tseslint from "typescript-eslint";
import { defineConfig, globalIgnores } from "eslint/config";

export default defineConfig([
  // ``._*`` are AppleDouble sidecars that an exFAT workspace leaves beside every file it
// writes. They are not source and cannot be parsed, so they are ignored rather than
// reported as a parse error on every lint run.
  globalIgnores(["dist", "coverage", "**/._*"]),
  {
    files: ["**/*.{ts,tsx}"],
    extends: [
      js.configs.recommended,
      tseslint.configs.recommended,
      reactHooks.configs.flat["recommended-latest"],
      reactRefresh.configs.vite,
    ],
    languageOptions: {
      ecmaVersion: 2023,
      globals: globals.browser,
    },
  },
  {
    files: ["vite.config.ts", "tests/**/*.{ts,tsx}"],
    languageOptions: {
      globals: { ...globals.node },
    },
  },
]);
