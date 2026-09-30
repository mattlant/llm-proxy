const tseslint = require("@typescript-eslint/eslint-plugin");
const parser = require("@typescript-eslint/parser");

module.exports = [{
  files: ["**/*.{ts,tsx}"],
  ignores: ["**/dist/**", "**/coverage/**"],
  languageOptions: { parser, parserOptions: { ecmaVersion: 2022, sourceType: "module", ecmaFeatures: { jsx: true } } },
  plugins: { "@typescript-eslint": tseslint },
  rules: { ...tseslint.configs.recommended.rules }
}];
