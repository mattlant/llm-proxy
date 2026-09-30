class AssetManifestPlugin {
  apply(compiler) {
    compiler.hooks.thisCompilation.tap("AssetManifestPlugin", (compilation) => {
      compilation.hooks.processAssets.tap(
        { name: "AssetManifestPlugin", stage: compiler.webpack.Compilation.PROCESS_ASSETS_STAGE_SUMMARIZE },
        () => {
          const assets = Object.keys(compilation.assets).filter((asset) => asset !== "asset-manifest.json").sort();
          compilation.emitAsset("asset-manifest.json", new compiler.webpack.sources.RawSource(`${JSON.stringify({ assets }, null, 2)}\n`));
        }
      );
    });
  }
}
module.exports = AssetManifestPlugin;
