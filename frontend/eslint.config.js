import js from "@eslint/js";
import reactHooks from "eslint-plugin-react-hooks";
import globals from "globals";
import tseslint from "typescript-eslint";

export default tseslint.config(
  { ignores: ["node_modules", "playwright-report", "test-results"] },
  {
    files: ["**/*.{ts,tsx}"],
    extends: [js.configs.recommended, ...tseslint.configs.strict],
    languageOptions: { ecmaVersion: 2023, globals: { ...globals.browser, ...globals.node } },
    plugins: { "react-hooks": reactHooks },
    rules: {
      ...reactHooks.configs.recommended.rules,
      "no-restricted-syntax": [
        "error",
        { selector: "JSXAttribute[name.name='dangerouslySetInnerHTML']", message: "Render model metadata as text (SEC-09)." },
      ],
      "no-restricted-properties": [
        "error",
        { object: "document", property: "cookie", message: "WeightLab uses no cookies." },
      ],
    },
  },
  {
    // Tests index recorded calls and read document.cookie to prove nothing is stored there.
    files: ["src/**/*.test.{ts,tsx}", "src/test/**", "tests/**"],
    rules: { "@typescript-eslint/no-non-null-assertion": "off", "no-restricted-properties": "off" },
  },
);
