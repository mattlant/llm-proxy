const path = require("path");
const HtmlWebpackPlugin = require("html-webpack-plugin");
const { aliases } = require("../../webpack/shared.cjs");
const AssetManifestPlugin = require("../../webpack/asset-manifest.cjs");

module.exports = {
  entry: "./src/index.ts",
  output: { path: path.resolve(__dirname, "dist"), publicPath: "/management/", clean: true, filename: "[name].[contenthash].js" },
  devServer: { devMiddleware: { publicPath: "/management/" }, port: 0 },
  resolve: { extensions: [".tsx", ".ts", ".js"], alias: aliases },
  module: { rules: [{ test: /\.tsx?$/, loader: "ts-loader", options: { transpileOnly: true }, exclude: /node_modules/ }] },
  plugins: [new HtmlWebpackPlugin({ title: "LLM Proxy management", templateContent: "<!doctype html><html><head><meta charset=\"utf-8\"><title>LLM Proxy management</title></head><body><div id=\"root\"></div></body></html>" }), new AssetManifestPlugin()]
};
